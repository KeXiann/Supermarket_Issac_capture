from project_paths import SONIC_ROOT
#!/usr/bin/env python3
"""Export actual G1 state, references and conservative QC without running DDS.

Outputs are development samples. This does not approve task success or promote
data into a formal dataset. Original recordings are never modified.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from capture_utils import (HUMAN_REFERENCE_MAX_AGE_S, MAX_HUMAN_PACKET_GAP_S, human_packet_gap_summary,
                           human_reference_valid)
from sonic_contract import SONIC_CPP_MOTOR_ORDER

ROOT = Path(__file__).resolve().parents[1]
PROJECT = SONIC_ROOT
DEFAULT_URDF = PROJECT / "gear_sonic/data/assets/robot_description/urdf/g1/main.urdf"
PHASES = {0: "warmup", 1: "pose", 2: "tail", 3: "outside_publication_or_unknown", 4: "pose_mode_entry_transition"}
ROBOT_FIELDS = {
    "joint_pos": ("q_rad", 29), "joint_vel": ("dq_rad_s", 29),
    "root_pos": ("root_pos_w_m", 3), "root_quat_wxyz": ("root_quat_wxyz", 4),
    "root_lin_vel_w": ("root_lin_vel_w_m_s", 3),
    "root_ang_vel_w": ("root_ang_vel_w_rad_s", 3),
    "target_joint_pos": ("pd_equilibrium_q_rad", 29),
    "target_joint_vel": ("pd_target_dq_rad_s", 29),
    "applied_torque_cmd": ("effort_command_or_estimate_nm", 29),
    "external_effort_cmd": ("external_effort_cmd_nm", 29),
    "drive_kp": ("drive_kp_nm_rad", 29), "drive_kd": ("drive_kd_nm_s_rad", 29),
}
HUMAN_FIELDS = ("smpl_pose", "smpl_joints", "body_quat_w", "joint_pos")


def file_info(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def validate_fk_manifest(captured, urdf):
    """Identify the actual deployed body URDF among additional hand assets."""
    assets = [f for f in captured['files'] if f.get('role') == 'body_fk_urdf']
    if not assets:
        assets = [f for f in captured['files']
                  if f['path'].endswith('/robot_description/urdf/g1/main.urdf')]
    if not assets:
        assets = [f for f in captured['files'] if f['path'].endswith('.urdf')]
    if len(assets) != 1 or assets[0]['sha256'] != file_info(urdf)['sha256']:
        raise ValueError('FK asset differs from the capture-time URDF; provide the captured asset')
    return 'FK URDF SHA256 matches the capture-time bridge manifest'


def load_robot(run, *, apply_recording_window=False):
    metadata = json.loads((run / "bridge.json").read_text())
    if tuple(metadata["motor_joint_names"]) != SONIC_CPP_MOTOR_ORDER:
        raise ValueError("unexpected DDS motor joint order; refusing an ambiguous trajectory")
    isaac = metadata["isaac_joint_names"]
    permutation = metadata["motor_to_isaac_joint_index"]
    # Revo2 can share the main articulation: 29 body + 22 hand joints.
    # The DDS body arrays remain 29-wide; validate their exact named mapping,
    # rather than requiring the entire articulation to contain only 29 joints.
    if (len(isaac) < 29 or len(set(isaac)) != len(isaac)
            or len(permutation) != 29
            or any(type(i) is not int or i < 0 or i >= len(isaac) for i in permutation)
            or len(set(permutation)) != 29
            or [isaac[i] for i in permutation] != list(SONIC_CPP_MOTOR_ORDER)):
        raise ValueError("inconsistent motor-to-Isaac permutation")
    with np.load(run / "bridge.npz", allow_pickle=False) as archive:
        raw = {k: archive[k] for k in archive.files}
    times = raw["time"]
    if times.ndim != 1 or len(times) < 2:
        raise ValueError("need at least two robot samples")
    n = len(times)
    for key, (_, width) in ROBOT_FIELDS.items():
        if raw[key].shape != (n, width):
            raise ValueError(f"invalid shape for {key}: {raw[key].shape}")
    if metadata.get("hand_joint_names"):
        for side in ("left", "right"):
            for suffix, width in (("joint_pos", 11), ("joint_vel", 11), ("dds_target", 7)):
                key = f"{side}_hand_{suffix}"
                if key not in raw or raw[key].shape != (n, width) or not np.isfinite(raw[key]).all():
                    raise ValueError(f"invalid Revo2 field: {key}")
    scalar = ("wall_time", "command_age", "base_tilt_rad", "controller_active",
              "raw_command_valid", "fault_latched", "startup_waiting",
              "dds_accepted_count", "dds_rejected_count")
    for key in scalar:
        if raw[key].shape != (n,):
            raise ValueError(f"invalid shape for {key}")
    for key in ("startup_waiting", "fault_latched", "controller_active", "raw_command_valid"):
        if not np.isin(raw[key], [0, 1]).all():
            raise ValueError(f"invalid boolean field {key}")
    startup = raw["startup_waiting"].astype(bool)
    if np.any(np.diff(startup.astype(int)) > 0):
        raise ValueError("startup waiting resumed after physics release")
    indices = np.flatnonzero(~startup)
    if len(indices) < 2:
        raise ValueError("need at least two physical samples after startup")
    data = {k: v[indices] for k, v in raw.items()}
    if apply_recording_window:
        from recording_window import recording_mask
        clocks = (data['state_monotonic_ns'] if 'state_monotonic_ns' in data else
                  ((metadata['started_monotonic'] + data['wall_time']) * 1e9).astype(np.int64))
        keep = recording_mask(run, clocks)
        indices = indices[keep]
        data = {k: v[keep] for k, v in data.items()}
        if len(indices) < 2:
            raise ValueError('explicit recording interval contains fewer than two physical samples')
    # Nonfinite startup command ages are legitimate; physical state is not.
    for key in ("time", "wall_time", "base_tilt_rad", *ROBOT_FIELDS):
        if not np.isfinite(data[key]).all():
            raise ValueError(f"nonfinite physical data: {key}")
    if np.any(np.diff(data["time"]) <= 0) or np.any(np.diff(data["wall_time"]) <= 0):
        raise ValueError("physical simulation and host timestamps must strictly increase")
    for key, width in (("velocity_limits_motor_order", 29),):
        values = np.asarray(metadata[key])
        if values.shape != (width,) or not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError(f"invalid {key}")
    lo = np.asarray(metadata["position_limits_motor_order"]["low"])
    hi = np.asarray(metadata["position_limits_motor_order"]["high"])
    if lo.shape != (29,) or hi.shape != (29,) or not np.isfinite([lo, hi]).all() or np.any(lo >= hi):
        raise ValueError("invalid recorded position limits")
    return metadata, data, indices, int(startup.sum())


def load_human(source):
    """Retain source clock and optional raw tracking; no fabricated translations."""
    archive = source / "pico_replay_archive.npz"
    files = sorted((source / "pico_input").glob("pose_*.npz"))
    if archive.exists():
        with np.load(archive, allow_pickle=False) as data:
            human = {k: data[k] for k in data.files}
    else:
        from audit_pico_response import collect
        human = collect(source)
    n = len(human["packet_time"])
    for key, shape in zip(HUMAN_FIELDS, [(n, 21, 3), (n, 24, 3), (n, 4), (n, 29)]):
        if human[key].shape != shape or not np.isfinite(human[key]).all():
            raise ValueError(f"invalid human reference: {key}")
    t = human["packet_time"]
    if n < 2 or t.shape != (n,) or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("human packet timestamps must strictly increase")
    # Only join raw tracking after verifying the entire row correspondence.
    raw_keys = ("body_poses_raw", "headset_pose", "body_timestamp_ns",
                "source_timestamp_ns", "timestamp_send_monotonic", "reference_timestamp_ns",
                "left_hand_joints", "right_hand_joints", "left_trigger", "right_trigger", "left_grip", "right_grip",
                "vr_position", "vr_orientation", "heading_increment", "timestamp_realtime")
    raw_note = "raw tracking unavailable in the supplied source"
    if files:
        if len(files) != n:
            raise ValueError("source archive/raw packet count mismatch; refusing an unsafe join")
        values = {k: [] for k in raw_keys}
        for i, path in enumerate(files):
            with np.load(path, allow_pickle=False) as packet:
                if (abs(float(packet["timestamp_monotonic"][0]) - t[i]) > 1e-6
                        or int(packet["frame_index"][-1]) != int(human["source_frame"][i])):
                    raise ValueError("source archive/raw packet clock or frame mismatch")
                for key in raw_keys:
                    if key in packet:
                        values[key].append(packet[key])
        for key, rows in values.items():
            if rows and len(rows) != n:
                raise ValueError(f"partially present raw tracking field: {key}")
            if rows:
                human[key] = np.stack(rows)
                if not np.isfinite(human[key]).all():
                    raise ValueError(f"nonfinite raw tracking: {key}")
        if "body_poses_raw" in human:
            raw_note = "raw OpenXR body positions/quaternions preserved in their original frame; not robot/world coordinates"
    human["time_from_first_packet_s"] = t - t[0]
    return human, raw_note


def mode_samples(path, host_time, max_age=.1):
    modes = np.full(len(host_time), -1, dtype=np.int16)
    if not path.exists():
        return modes
    with path.open() as stream:
        header = stream.readline().strip().split(",")
    cols = [header.index("time_monotonic_ms"), header.index("encoder_mode_0")]
    values = np.loadtxt(path, delimiter=",", skiprows=1, usecols=cols, ndmin=2)
    if not len(values) or not np.isfinite(values).all() or np.any(np.diff(values[:, 0]) <= 0):
        raise ValueError("invalid SONIC mode clock")
    if not np.isin(values[:, 1], [0, 1, 2]).all():
        raise ValueError("unexpected encoder mode value")
    mt = values[:, 0] / 1000
    row = np.searchsorted(mt, host_time, side="right") - 1
    safe_row = np.maximum(row, 0)
    valid = (row >= 0) & (host_time - mt[safe_row] <= max_age)
    modes[valid] = values[safe_row[valid], 1].astype(np.int16)
    return modes


def segment_ids(valid, times, expected_dt):
    """Do not join across inactive/fault periods or missing recording intervals."""
    result = np.full(len(times), -1, dtype=np.int32)
    segment = -1
    for i in range(len(times)):
        if valid[i]:
            if i == 0 or not valid[i - 1] or times[i] - times[i - 1] > 1.5 * expected_dt:
                segment += 1
            result[i] = segment
    return result


def align_reference(host_time, summary, human, modes):
    phase = np.full(len(host_time), 3, dtype=np.int8)
    reference_time = np.full(len(host_time), np.nan)
    duration = float(human["time_from_first_packet_s"][-1])
    if summary.get("test") == "pose":
        # Recorded human capture clock and replay publication clock are distinct.
        origin = float(summary["publication_start_monotonic"])
        end = float(summary["publication_end_monotonic"])
        start = float(summary["pose_start_seconds"])
        warmup = float(summary["planner_warmup_seconds"])
        if (not np.isfinite([origin, end, start, warmup]).all() or end <= origin
                or start < 0 or start >= duration or warmup < 0):
            raise ValueError("invalid replay timing metadata")
        if abs(float(summary["source_duration_seconds"]) - duration) > 1e-6:
            raise ValueError("replay source duration does not match the supplied human archive")
        elapsed = host_time - origin
        inside = (host_time >= origin) & (host_time <= end)
        phase[inside & (elapsed < warmup)] = 0
        phase[inside & (elapsed >= warmup) & (elapsed <= warmup + duration - start)] = 1
        phase[inside & (elapsed > warmup + duration - start)] = 2
        active = (phase == 1) | (phase == 2)
        reference_time[active] = start + elapsed[active] - warmup
        # Both processes run at 50 Hz. Preserve and explicitly label the short
        # mode-entry interval; later unknown/wrong modes remain QC failures.
        transition = (phase == 1) & (elapsed < warmup + .1) & (modes != 2)
        phase[transition] = 4
        alignment = "replay publication monotonic clock + warmup + source offset; no original capture-clock join"
    else:
        inside = (host_time >= human["packet_time"][0]) & (host_time <= human["packet_time"][-1])
        phase[(modes == 0) | (modes == 1)] = 0
        phase[inside & (modes == 2)] = 1
        reference_time[inside] = host_time[inside] - human["packet_time"][0]
        alignment = "live same-host Manager capture clock and Bridge clock; requires same host/boot"
    rows = np.full(len(host_time), -1, dtype=np.int64)
    finite = np.isfinite(reference_time)
    rows[finite] = np.clip(np.searchsorted(human["time_from_first_packet_s"], reference_time[finite],
                                          side="right") - 1, 0, len(human["packet_time"]) - 1)
    return phase, reference_time, rows, alignment


def forward_kinematics(q, root_pos, root_quat, urdf):
    import pinocchio as pin
    from scipy.spatial.transform import Rotation
    model = pin.buildModelFromUrdf(str(urdf))
    if model.nq != 29:
        raise ValueError("FK URDF does not have the expected 29 scalar joints")
    order = []
    links = []
    joint_children = {j.attrib["name"]: j.find("child").attrib["link"]
                      for j in ET.parse(urdf).getroot().findall("joint")}
    for name in SONIC_CPP_MOTOR_ORDER:
        joint = model.getJointId(name)
        if joint >= model.njoints or model.joints[joint].nq != 1:
            raise ValueError(f"FK URDF missing expected scalar joint: {name}")
        order.append(model.joints[joint].idx_q)
        links.append(joint_children[name])
    frame_names = links + ["left_ankle_roll_link", "right_ankle_roll_link",
                           "left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link"]
    frames = [model.getFrameId(name) for name in frame_names]
    if max(frames) >= model.nframes:
        raise ValueError("FK URDF missing an expected link frame")
    kin = model.createData()
    local = np.empty((len(q), len(frames), 3))
    for i, row in enumerate(q):
        state = pin.neutral(model)
        state[order] = row
        pin.framesForwardKinematics(model, kin, state)
        local[i] = [kin.oMf[f].translation for f in frames]
    if np.any(np.linalg.norm(root_quat, axis=1) < 1e-8):
        raise ValueError("zero root quaternion")
    rotation = Rotation.from_quat(root_quat[:, [1, 2, 3, 0]]).as_matrix()
    world = np.einsum("nij,nkj->nki", rotation, local) + root_pos[:, None, :]
    return {"joint_link_pos_w_m": world[:, :29], "ankle_pos_root_m": local[:, 29:31],
            "ankle_pos_w_m": world[:, 29:31], "wrist_link_pos_w_m": world[:, 31:33],
            "torso_link_pos_w_m": world[:, 33]}, links


def evaluate_quality(metadata, data, result, human, summary, excluded, recording_bounds=None):
    checks = []
    def check(name, passed, detail):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})
    t = result["time_s"]
    expected = float(metadata["sim_dt"] * metadata["record_every"])
    dt = np.diff(t)
    check("recording_cadence", np.all(np.abs(dt - expected) <= expected * .5),
          {"expected_s": expected, "min_s": float(dt.min()), "max_s": float(dt.max())})
    from upstream_contract import check_physics_metadata
    check("free_base_sonic_profile", not metadata["fixed_base"] and metadata["target_policy"] == "sonic-pd"
          and check_physics_metadata(metadata) and metadata["extra_waist_damping"] == 0
          and metadata["pd_target_slew"] == "none", "free base / versioned PD and model contract / no extra damping or slew")
    check("all_physical_samples_controlled", result["sample_valid"].all(),
          {"invalid_samples": int((~result["sample_valid"]).sum()), "fault_reason": metadata["fault_reason"]})
    check("no_fault_or_execution_error", not metadata["fault_reason"] and not summary.get("execution_error"),
          {"execution_error": summary.get("execution_error"), "fault_reason": metadata["fault_reason"]})
    check("no_rejected_dds_commands", metadata["dds_rejected_count"] == 0,
          {"count": metadata["dds_rejected_count"]})
    lo = np.asarray(metadata["position_limits_motor_order"]["low"])
    hi = np.asarray(metadata["position_limits_motor_order"]["high"])
    violation = np.maximum(np.maximum(lo - result["q_rad"], result["q_rad"] - hi), 0)
    check("actual_position_limits", violation.max() <= .01,
          {"max_violation_rad": float(violation.max()), "solver_tolerance_rad": .01})
    ratio = np.abs(result["dq_rad_s"]) / metadata["velocity_limits_motor_order"]
    worst = np.unravel_index(ratio.argmax(), ratio.shape)
    check("actual_velocity_limits", ratio.max() <= 1.01,
          {"max_limit_ratio": float(ratio[worst]), "tolerance_ratio": 1.01,
           "worst_joint": SONIC_CPP_MOTOR_ORDER[worst[1]], "time_s": float(t[worst[0]])})
    norm_error = np.abs(np.linalg.norm(result["root_quat_wxyz"], axis=1) - 1)
    check("root_quaternion_norm", norm_error.max() <= .001,
          {"max_norm_error": float(norm_error.max()), "tolerance": .001})
    # These are existing guard boundaries, not a task-quality or comfort threshold.
    tilt = np.rad2deg(data["base_tilt_rad"])
    height = result["root_pos_w_m"][:, 2]
    min_height = metadata.get("fall_guard_min_height_m", .38)
    check("fall_guard_envelope", height.min() >= min_height and tilt.max() < metadata["max_base_tilt_deg"],
          {"min_height_m": float(height.min()), "max_height_m": float(height.max()),
           "max_tilt_deg": float(tilt.max()), "guard_height_m": min_height,
           "guard_tilt_deg": metadata["max_base_tilt_deg"]})
    pose = result["phase_id"] == 1
    check("pose_mode_coverage", pose.any() and np.all(result["encoder_mode"][pose] == 2),
          {"pose_samples": int(pose.sum()), "pose_mode_mismatches": int(np.count_nonzero(pose & (result["encoder_mode"] != 2))),
           "entry_transition_samples_excluded": int((result["phase_id"] == 4).sum()),
           "entry_transition_window_s": .1,
           "mode_clock_max_age_s": .1})
    window = recording_bounds
    packet_times = human['packet_time']
    if window is not None:
        packet_times = packet_times[(packet_times >= window['start_monotonic_ns'] / 1e9)
                                    & (packet_times < window['end_monotonic_ns'] / 1e9)]
    gap_summary = human_packet_gap_summary(packet_times)
    check("human_reference_gaps", gap_summary["capture_gap_gate_passed"],
          {"max_gap_s": gap_summary["max_packet_gap_s"],
           "gaps_over_100ms": gap_summary["packet_gaps_over_100ms"],
           "gaps_over_gate": gap_summary["packet_gaps_over_gate"], **gap_summary})
    if summary.get("test") == "pose":
        complete = summary.get("pose_last_source_frame") == len(human["packet_time"]) - 1
        actual_duration = summary["publication_end_monotonic"] - summary["publication_start_monotonic"]
        expected_duration = (summary["planner_warmup_seconds"] + summary["source_duration_seconds"]
                             - summary["pose_start_seconds"] + summary["replay_tail_seconds"])
        check("whole_selected_source_published", complete and actual_duration >= expected_duration - .04,
              {"last_source_row": summary.get("pose_last_source_frame"), "expected_last_row": len(human["packet_time"]) - 1,
               "publication_s": actual_duration, "expected_including_tail_s": expected_duration})
    check("graceful_physics_stop", "before policy shutdown" in metadata.get("stop_reason", ""),
          {"reason": metadata.get("stop_reason"), "native_cleanup": summary.get("cleanup")})
    pose_valid = result["pose_valid"].astype(bool)
    return {"schema_version": 1, "stage": "development", "formal_collection_approved": False,
            "status": "development_checks_pass" if all(c["passed"] for c in checks) else "development_review_required",
            "checks": checks, "sample_count": len(t), "startup_samples_excluded": excluded,
            "physical_duration_s": float(np.ptp(t)),
            "source_duration_s": float(human["time_from_first_packet_s"][-1]),
            "phase_samples": {name: int((result["phase_id"] == i).sum()) for i, name in PHASES.items()},
            "valid_pose_samples": int(pose_valid.sum()),
            "valid_pose_duration_s": float(np.ptp(t[pose_valid])) if pose_valid.any() else 0.,
            "valid_pose_segments": int(len(np.unique(result["pose_segment_id"][pose_valid]))),
            "max_actual_joint_speed_rad_s": float(np.abs(result["dq_rad_s"]).max()),
            "max_pelvis_horizontal_displacement_m": float(np.linalg.norm(result["root_pos_w_m"][:, :2] - result["root_pos_w_m"][0, :2], axis=1).max()),
            "limitations": [
                "Checks are initial engineering thresholds, not a formal task-success or dataset-acceptance standard.",
                "Recorded joint limits and velocities are checked at 50 Hz; sub-sample peaks are unobserved.",
                "No contact forces or object/task annotations were recorded; ankle FK is not a contact measurement.",
                "PD equilibrium targets are virtual spring targets, not human/robot anatomical motion ground truth.",
                "Implicit-PD effort is a command-side estimate, not a measured actuator torque.",
                ("State monotonic timestamps are sampled before state reads; camera/human/token streams remain independently sampled and use causal host-clock association."
                 if "state_monotonic_ns" in data else
                 "Legacy host timestamps are sampled after the physics step while states are read before it; alignment is approximate within a loop period."),
                "Formal assets, task definitions, contact validation and end-effector calibration are pending.",
                "Native Kit cleanup can time out after recording/physics have stopped; see cleanup detail.",
            ]}


def write_csv(path, data):
    keys = ["time_s", "sim_time_s", "host_monotonic_s", "source_recording_row", "phase_id", "encoder_mode",
            "sample_valid", "segment_id", "pose_valid", "pose_segment_id", "reference_frame_index"]
    columns = [data[k] for k in keys]
    header = keys.copy()
    for key in ("root_pos_w_m", "root_quat_wxyz", "root_lin_vel_w_m_s", "root_ang_vel_w_rad_s"):
        labels = ["w", "x", "y", "z"] if "quat" in key else ["x", "y", "z"]
        for i, label in enumerate(labels):
            header.append(f"{key}_{label}")
            columns.append(data[key][:, i])
    for key in ("q_rad", "dq_rad_s", "pd_equilibrium_q_rad"):
        for i, name in enumerate(SONIC_CPP_MOTOR_ORDER):
            header.append(f"{key}/{name}")
            columns.append(data[key][:, i])
    np.savetxt(path, np.column_stack(columns), delimiter=",", header=",".join(header), comments="", fmt="%.10g")


def plot_trajectory(path, data, quality):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = data["time_s"]
    fig, axes = plt.subplots(4, 2, figsize=(15, 13), layout="constrained")
    fig.suptitle("G1 / SONIC free-base POSE - development trajectory\nActual state; no task/contact qualification", fontsize=16)
    axes[0, 0].plot(t, data["root_pos_w_m"])
    axes[0, 0].legend(["world x", "world y", "world z"], ncol=3)
    axes[0, 0].set_ylabel("Pelvis position (m)")
    axes[0, 1].plot(t, data["base_tilt_deg"])
    axes[0, 1].set_ylabel("Base tilt (deg)")
    axes[1, 0].plot(t, data["q_rad"][:, :12], lw=.6, alpha=.7)
    axes[1, 0].set_ylabel("Actual leg joints (rad)")
    axes[1, 1].plot(t, data["q_rad"][:, 15:], lw=.6, alpha=.7)
    axes[1, 1].set_ylabel("Actual arm joints (rad)")
    speed_ratio = np.max(np.abs(data["dq_rad_s"]) / data["velocity_limits_rad_s"], axis=1)
    axes[2, 0].plot(t, speed_ratio)
    axes[2, 0].axhline(1., color="red", ls="--", label="recorded limit")
    axes[2, 0].set_ylabel("Max joint speed / limit")
    axes[2, 0].legend()
    axes[2, 1].plot(t, data["ankle_pos_w_m"][:, :, 2])
    axes[2, 1].legend(["left ankle", "right ankle"])
    axes[2, 1].set_ylabel("Ankle link origin height (m)\nFK only; not sole/contact")
    axes[3, 0].plot(t, data["command_age_s"] * 1000)
    axes[3, 0].set_ylabel("Latest DDS command age (ms)")
    axes[3, 1].step(t, data["encoder_mode"], where="post", label="SONIC encoder mode")
    axes[3, 1].step(t, data["phase_id"], where="post", label="phase 0 warmup / 1 pose / 2 tail", alpha=.65)
    axes[3, 1].set_ylabel("Mode / scheduled phase")
    axes[3, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.set_xlim(t[0], t[-1])
        ax.set_xlabel("Physical time from first exported sample (s)")
        for phase, color in [(0, "#cccccc"), (2, "#e8c56b")]:
            mask = data["phase_id"] == phase
            if mask.any():
                ax.axvspan(t[mask][0], t[mask][-1], color=color, alpha=.18)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def prepare(run, source, out, urdf=DEFAULT_URDF, preview=False):
    run, source, out, urdf = map(lambda p: Path(p).resolve(), [run, source, out, urdf])
    if out.exists():
        raise FileExistsError(f"output already exists: {out}; use a fresh path")
    # Validate completely before creating output; malformed state must fail closed.
    metadata, data, indices, excluded = load_robot(run, apply_recording_window=True)
    capture_path = run / "capture_bridge.json"
    fk_identity = "historical capture-time asset hash unavailable"
    if capture_path.exists():
        captured = json.loads(capture_path.read_text())
        fk_identity = validate_fk_manifest(captured, urdf)
    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    human, raw_note = load_human(source)
    if summary.get("pose_source_sha256") and summary["pose_source_sha256"] != file_info(source / "pico_replay_archive.npz")["sha256"]:
        raise ValueError("replay source SHA256 mismatch")
    host_time = (data["state_monotonic_ns"] / 1e9 if "state_monotonic_ns" in data
                 else metadata["started_monotonic"] + data["wall_time"])
    mode_path = run / "sonic_state/encoder_mode.csv"
    modes = mode_samples(mode_path, host_time)
    phase, ref_time, ref_rows, alignment = align_reference(host_time, summary, human, modes)
    expected_dt = metadata["sim_dt"] * metadata["record_every"]
    valid = (data["controller_active"].astype(bool) & data["raw_command_valid"].astype(bool)
             & ~data["fault_latched"].astype(bool) & np.isfinite(data["command_age"])
             & (data["command_age"] >= 0) & (data["command_age"] <= metadata["watchdog"]))
    ref_age = ref_time - human["time_from_first_packet_s"][np.maximum(ref_rows, 0)]
    pose_valid = valid & (phase == 1) & human_reference_valid(ref_rows, ref_age, modes)
    result = {name: data[key] for key, (name, _) in ROBOT_FIELDS.items()}
    result.update(time_s=data["time"] - data["time"][0], sim_time_s=data["time"],
                  host_monotonic_s=host_time, source_recording_row=indices, phase_id=phase,
                  encoder_mode=modes, sample_valid=valid, segment_id=segment_ids(valid, data["time"], expected_dt),
                  pose_valid=pose_valid, pose_segment_id=segment_ids(pose_valid, data["time"], expected_dt),
                  reference_time_s=ref_time, reference_frame_index=ref_rows, reference_age_s=ref_age,
                  command_age_s=data["command_age"], base_tilt_deg=np.rad2deg(data["base_tilt_rad"]),
                  joint_names=np.array(SONIC_CPP_MOTOR_ORDER), velocity_limits_rad_s=np.array(metadata["velocity_limits_motor_order"]),
                  controller_active=data["controller_active"], raw_command_valid=data["raw_command_valid"],
                  fault_latched=data["fault_latched"], dds_accepted_count=data["dds_accepted_count"],
                  dds_rejected_count=data["dds_rejected_count"])
    fk, links = forward_kinematics(result["q_rad"], result["root_pos_w_m"], result["root_quat_wxyz"], urdf)
    result.update(fk, joint_link_names=np.array(links))
    if metadata.get("hand_joint_names"):
        for side in ("left", "right"):
            for suffix in ("joint_pos", "joint_vel", "target_joint_pos", "dds_target", "command_valid", "command_count",
                           "command_monotonic_s", "command_unix_ns"):
                key = f"{side}_hand_{suffix}"
                if key in data:
                    result[key] = data[key]
            result[f"{side}_hand_joint_names"] = np.array(metadata["hand_joint_names"][side])
    from recording_window import recording_window
    quality = evaluate_quality(metadata, data, result, human, summary, excluded, recording_window(run))
    quality["human_raw_tracking"] = raw_note
    quality["reference_alignment"] = alignment
    quality["fk_asset_identity"] = fk_identity
    input_paths = [run / "bridge.npz", run / "bridge.json", urdf]
    input_paths += [p for p in [summary_path, mode_path, source / "pico_replay_archive.npz"] if p.exists()]
    input_paths += sorted(run.glob("capture_*.json"))
    manifest = {"schema_version": 1, "stage": "development", "formal_asset_pending": True,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "run": str(run), "source_run": str(source), "inputs": [file_info(p) for p in input_paths],
                "analysis_code": file_info(Path(__file__)),
                "analysis_dependencies": [file_info(ROOT / "bridge/capture_utils.py")],
                "historical_provenance": "Current FK asset hash is recorded at export. Historical runs without capture_*.json cannot prove capture-time code/asset/model identity.",
                "capture_metadata": metadata, "replay_summary": summary}
    out.mkdir(parents=True, exist_ok=False)
    (out / "exporter_source.py").write_bytes(Path(__file__).read_bytes())
    (out / "capture_utils_source.py").write_bytes((ROOT / "bridge/capture_utils.py").read_bytes())
    manifest["analysis_source_snapshot"] = "exporter_source.py"
    manifest["analysis_dependency_snapshots"] = ["capture_utils_source.py"]
    np.savez_compressed(out / "robot_trajectory.npz", **result)
    np.savez_compressed(out / "human_reference.npz", **human)
    segment_dir = out / "pose_segments"
    segment_dir.mkdir()
    static_fields = {"joint_names", "joint_link_names", "velocity_limits_rad_s",
                     "left_hand_joint_names", "right_hand_joint_names"}
    quality["pose_segment_files"] = []
    for segment in np.unique(result["pose_segment_id"][result["pose_valid"]]):
        mask = result["pose_segment_id"] == segment
        part = {k: v if k in static_fields else v[mask] for k, v in result.items()}
        part["time_in_segment_s"] = part["time_s"] - part["time_s"][0]
        destination = segment_dir / f"segment_{segment:03d}.npz"
        np.savez_compressed(destination, **part)
        quality["pose_segment_files"].append({"path": str(destination.relative_to(out)),
                                              "samples": int(mask.sum()),
                                              "duration_s": float(part["time_in_segment_s"][-1])})
    write_csv(out / "robot_trajectory.csv", result)
    (out / "quality_report.json").write_text(json.dumps(quality, indent=2, allow_nan=False))
    (out / "trajectory_schema.json").write_text(json.dumps({
        "schema_version": 1, "stage": "development", "phase_ids": PHASES,
        "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in result.items()},
        "conventions": {
            "time_s": "simulation time relative to first post-startup sample; samples are not resampled or gap-filled",
            "sim_time_s": "original physical simulation clock; startup waiting is excluded",
            "root_pos_w_m": "Isaac world, metres, Z up; original origin retained",
            "root_quat_wxyz": "root orientation in Isaac world, scalar first, original values retained",
            "joint_names": "SONIC C++ DDS motor order for all 29-wide motor arrays",
            "joint_link_pos_w_m": "URDF joint child-link origins from FK, ordered by joint_link_names; not measured contact",
            "pd_equilibrium_q_rad": "applied SONIC virtual PD equilibrium; can exceed physical limits; not the anatomical reference",
            "effort_command_or_estimate_nm": metadata["torque_recording_semantics"],
            "drive_kp_nm_rad": "actual PD gains after bridge scaling; backend is recorded in bridge.json",
            "sample_valid": "active, fresh and valid DDS command, no latched fault; not a task-success label",
            "segment_id": "-1 invalid; new ID across inactive/fault samples or cadence gaps; do not concatenate gaps",
            "pose_valid": "sample_valid AND scheduled pose phase AND observed SONIC encoder mode=2 AND causal human reference age <=100ms; segments split across stale human rows",
            "phase_id": "phase 4 explicitly excludes a wrong/unknown encoder mode only within the first 100 ms of scheduled POSE; later mismatches fail QC",
            "reference_time_s": "time into original human archive; NaN when no pose/tail reference applies",
            "reference_frame_index": "row in human_reference.npz, -1 when absent; tail repeats final source row",
            "reference_age_s": "reference_time_s minus selected human row time; NaN when absent; >100ms is stale and excluded from pose_valid",
            "human_reference": "original source timeline and SMPL fields, unresampled; joint_pos is a Manager field, not actual robot state",
            "hands": "{left,right}_hand_joint_pos/vel: 11 physical joints in recorded hand_joint_names order, rad/rad_s; dds_target: 7-slot Dex3 synergy command, not physical joint targets; command_valid means ever received, not freshness or grasp success",
            "raw_tracking": raw_note, "alignment": alignment,
            "csv": "flattened convenience subset; NPZ and this schema define the full trajectory",
        }}, indent=2))
    plot_trajectory(out / "trajectory_plots.png", result, quality)
    if preview:
        from render_trajectory_preview import render
        render(out / "robot_trajectory.npz", out / "actual_state_preview.mp4", urdf)
    manifest["outputs"] = [file_info(p) for p in sorted(out.rglob("*")) if p.is_file()]
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    return quality


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--source-run", type=Path, help="original human run; defaults to live run itself")
    parser.add_argument("--out", type=Path, required=True, help="new output directory; never overwrites")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--preview", action="store_true", help="also render an actual-state skeleton MP4; requires ffmpeg")
    args = parser.parse_args()
    quality = prepare(args.run, args.source_run or args.run, args.out, args.urdf, args.preview)
    print(json.dumps({k: quality[k] for k in ["status", "sample_count", "physical_duration_s",
                                             "source_duration_s", "valid_pose_samples", "valid_pose_duration_s"]}, indent=2))
    print(f"Output: {args.out.resolve()}")
    if quality["status"] != "development_checks_pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
