#!/usr/bin/env python3
"""Audit real outgoing Manager packets and actual Isaac feedback offline.

Produces an immutable compact replay archive using the original monotonic
timestamps. Raw translations were not present in old packets; do not invent
them from root-relative SMPL joints.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def collect(run):
    files = sorted((run / "pico_input").glob("pose_*.npz"))
    if not files:
        raise ValueError("No recorded Manager packets")
    fields = ("smpl_pose", "smpl_joints", "body_quat_w", "joint_pos")
    result = {k: [] for k in fields}
    result.update(packet_time=[], source_frame=[])
    optional = ("left_hand_joints", "right_hand_joints", "left_trigger", "right_trigger",
                "left_grip", "right_grip", "timestamp_send_monotonic",
                "timestamp_realtime",
                "vr_position", "vr_orientation", "heading_increment",
                "body_timestamp_ns", "source_timestamp_ns", "reference_timestamp_ns")
    extras = {k: [] for k in optional}
    for file in files:
        with np.load(file, allow_pickle=False) as p:
            for k in fields:
                result[k].append(p[k][-1])
            result["packet_time"].append(float(p["timestamp_monotonic"][0]))
            result["source_frame"].append(int(p["frame_index"][-1]))
            for k in optional:
                if k in p:
                    extras[k].append(p[k])
    for k, rows in extras.items():
        if rows:
            if len(rows) != len(files):
                raise ValueError(f"partially present packet field: {k}")
            result[k] = rows
    return {k: np.asarray(v) for k, v in result.items()}


def audit(run, packets):
    t = packets["packet_time"]
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError("Manager packet timestamps are not increasing")
    j = packets["smpl_joints"]
    relative = j - j[:, :1]
    # SMPL left/right ankles are 7/8; local coordinates already use Z-up.
    summary = {"run": str(run), "packet_count": len(t),
        "duration_seconds": float(np.ptp(t)),
        "median_packet_period_ms": float(np.median(dt)*1000),
        "p99_packet_period_ms": float(np.quantile(dt, .99)*1000),
        "max_packet_gap_seconds": float(dt.max()),
        "packet_gaps_over_100ms": int(np.count_nonzero(dt > .1)),
        "translation_available": False,
        "ankle_relative_range_m": np.ptp(relative[:, [7, 8]], axis=0).tolist(),
        "windows": []}
    meta = json.loads((run/"bridge.json").read_text())
    with np.load(run/"bridge.npz") as robot:
        bt = meta["started_monotonic"] + robot["wall_time"]
        for start in np.arange(t[0], t[-1]-5, 5):
            end = start+5
            pm = (t >= start) & (t < end)
            bm = (bt >= start) & (bt < end)
            if pm.sum() < 20 or bm.sum() < 20:
                continue
            ankles = relative[pm][:, [7, 8]]
            root = robot["root_pos"][bm]
            summary["windows"].append({"start_from_first_packet_s": float(start-t[0]),
                "packet_count": int(pm.sum()),
                "ankle_x_range_m": np.ptp(ankles[..., 0], axis=0).tolist(),
                "ankle_z_range_m": np.ptp(ankles[..., 2], axis=0).tolist(),
                "robot_pelvis_displacement_m": (root[-1]-root[0]).tolist(),
                "robot_leg_range_deg": float(np.rad2deg(np.ptp(robot["joint_pos"][bm, :12], axis=0)).max())})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    p = collect(args.run)
    summary = audit(args.run, p)
    with (args.run/"pico_response_audit.json").open("x") as stream:
        json.dump(summary, stream, indent=2)
    with (args.run/"pico_replay_archive.npz").open("xb") as stream:
        np.savez_compressed(stream, **p)
    print(json.dumps({k: v for k, v in summary.items() if k != "windows"}, indent=2))
    strongest = sorted(summary["windows"], key=lambda w: max(w["ankle_z_range_m"]), reverse=True)[:6]
    print("Most active ankle windows:", json.dumps(strongest, indent=2))


if __name__ == "__main__":
    main()
