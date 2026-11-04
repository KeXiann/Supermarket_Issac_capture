#!/usr/bin/env python3
"""Export Bridge/PICO evidence using the pinned Kinetra configs/data_standard schema.

Schema compatibility is distinct from hardware equivalence. Revo2 normalization,
sampled DDS timing and unavailable hardware fields are explicit in meta/info.json.
Raw observations and source recordings are never rewritten.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from capture_utils import (HUMAN_REFERENCE_MAX_AGE_S, atomic_json,
                           human_reference_valid, sync_directory)
from prepare_trajectory import DEFAULT_URDF, load_human, load_robot, mode_samples
import data_standard

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "assets/collection_reference_2609"
VIDEO_KEY = "observation.images.ego_view"
SCALARS = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
TRAINING_ONLY_FIELDS = ("observation.previous_hand_action_joint_position",
                        "action.aligned_hand_joint_position")


def validate_startup_window(config, rows):
    """Do not publish startup-reference adaptation as human task demonstration."""
    if not config or not config.get('simulation_startup_reference'):
        return
    cycles = config.get('simulation_startup_reference_cycles')
    if not isinstance(cycles, int) or cycles < 10:
        raise ValueError('Missing or invalid simulation startup reference contract')
    if any(int(row['index']) < cycles for row in rows):
        raise ValueError('Recording includes SONIC startup reference transition; wait for [PERF] READY before recording')


def load_sonic_debug(run):
    """Read the optional patched-C++ causal stream without trusting its rows."""
    path = Path(run) / "sonic_debug.msgpack"
    if not path.exists():
        return None
    from sonic_debug_recorder import validate_recording
    _, records = validate_recording(path)
    rows = []
    config = None
    for record in records:
        payload = record.get("payload", {})
        if record.get("topic") == "robot_config":
            config = payload
            continue
        if (record.get("topic") == "g1_debug" and isinstance(payload, dict)
                and payload.get("body_action_alignment_version") == 2):
            rows.append(payload)
    if not rows:
        raise ValueError("SONIC debug stream has no protocol-v2 state/action rows")
    rows.sort(key=lambda row: int(row.get("state_monotonic_ns", 0)))
    times = np.asarray([int(row["state_monotonic_ns"]) for row in rows], dtype=np.int64)
    if np.any(times <= 0) or np.any(np.diff(times) <= 0):
        raise ValueError("SONIC debug state clocks must strictly increase")
    return rows, times, config


def valid_debug_row(row):
    """Apply the pinned reference body's causal and shape checks to one C++ row."""
    reference_data = str(REFERENCE / "gear_sonic/data")
    if reference_data not in sys.path:
        sys.path.insert(0, reference_data)
    from body_alignment import aligned_body_frame
    try:
        _, fields = aligned_body_frame(row)
        return bool(fields["observation.body_source_age_s"][0] <= .1)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def template():
    return data_standard.template()


def csv_state(path, host, width):
    with path.open() as stream:
        names = stream.readline().strip().split(",")
    values = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    times = values[:, names.index("time_monotonic_ms")] / 1000.
    if len(times) < 2 or np.any(np.diff(times) <= 0) or not np.isfinite(values).all():
        raise ValueError(f"invalid SONIC CSV: {path}")
    rows = np.searchsorted(times, host, side="right") - 1
    safe = np.maximum(rows, 0)
    age = host - times[safe]
    if np.any(rows < 0) or np.any(age > .1) or values.shape[1] - 5 != width:
        raise ValueError(f"missing/stale/wrong-width SONIC CSV: {path}")
    return values[safe, 5:], rows, age


def wrist_fk(q, urdf=DEFAULT_URDF):
    """Reference exporter uses wrist FK in the base frame, scalar-first quat."""
    import pinocchio as pin
    from scipy.spatial.transform import Rotation
    from sonic_contract import SONIC_CPP_MOTOR_ORDER
    model = pin.buildModelFromUrdf(str(urdf))
    order = [model.joints[model.getJointId(name)].idx_q for name in SONIC_CPP_MOTOR_ORDER]
    frames = [model.getFrameId(f"{side}_wrist_yaw_link") for side in ("left", "right")]
    data = model.createData()
    result = np.empty((len(q), 14))
    for i, row in enumerate(q):
        state = pin.neutral(model)
        state[order] = row
        pin.framesForwardKinematics(model, data, state)
        for side, frame in enumerate(frames):
            pose = data.oMf[frame]
            quat = Rotation.from_matrix(pose.rotation).as_quat()[[3, 0, 1, 2]]
            result[i, side * 7:(side + 1) * 7] = np.r_[pose.translation, quat]
    return result


def mentor_rot6d(quaternions):
    """Match the reference quat_to_rot6d: column 0 followed by column 1."""
    from scipy.spatial.transform import Rotation
    q = np.asarray(quaternions)
    if q.shape[-1] != 4:
        raise ValueError("rotation input must be scalar-first quaternions")
    matrix = Rotation.from_quat(q.reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix()
    return matrix[:, :, :2].transpose(0, 2, 1).reshape(*q.shape[:-1], 6)


def mentor_target_orientation(quaternions, heading, valid):
    """Use the reference accumulated-heading and initial-yaw normalization."""
    from scipy.spatial.transform import Rotation
    q = np.asarray(quaternions, dtype=np.float64)
    valid_rows = np.flatnonzero(valid)
    output = np.zeros((len(q), 6), dtype=np.float64)
    if not len(valid_rows):
        return output
    body = Rotation.from_quat(q[valid_rows][:, [1, 2, 3, 0]])
    target = Rotation.from_euler("z", np.asarray(heading)[valid_rows]) * body
    euler = target.as_euler("ZYX")
    euler[:, 0] -= euler[0, 0]
    normalized = Rotation.from_euler("ZYX", euler).as_quat()[:, [3, 0, 1, 2]]
    output[valid_rows] = mentor_rot6d(normalized)
    return output


def build_arrays(run):
    from scipy.spatial.transform import Rotation
    meta, robot, indices, excluded = load_robot(run, apply_recording_window=True)
    sonic_debug = load_sonic_debug(run)
    bridge_join_age = None
    if sonic_debug is not None:
        debug_rows, debug_times, debug_config = sonic_debug
        bridge_source = robot
        bridge_times = robot["state_monotonic_ns"].astype(np.int64)
        from recording_window import recording_mask
        keep = (debug_times >= bridge_times[0]) & recording_mask(run, debug_times)
        debug_rows = [row for row, use in zip(debug_rows, keep) if use]
        debug_times = debug_times[keep]
        if not len(debug_times):
            raise ValueError("SONIC protocol-v2 stream predates Bridge physical recording")
        validate_startup_window(debug_config, debug_rows)
        bridge_rows = np.searchsorted(bridge_times, debug_times, side="right") - 1
        bridge_join_age = (debug_times - bridge_times[bridge_rows]).astype(np.float64) / 1e9
        robot = {key: value[bridge_rows] for key, value in robot.items()}
        indices = indices[bridge_rows]
    else:
        debug_rows, debug_times, debug_config = None, None, None
        bridge_source = robot
        bridge_times = robot["state_monotonic_ns"].astype(np.int64)
    human, _ = load_human(run)
    if not meta.get("recording_finalized"):
        raise ValueError("refusing a running checkpoint")
    n = len(indices)
    info = template()
    fields = {k: v for k, v in info["features"].items() if v["dtype"] != "video"}
    arrays = {k: np.full((n, *f["shape"]), -1 if f["shape"] == [1] else 0, dtype=f["dtype"])
              for k, f in fields.items()}
    availability = {k: {"available": False, "reason": "not produced by the simulation adapter; exclude from training"}
                    for k in fields}
    def put(key, value, source):
        value = np.asarray(value, dtype=fields[key]["dtype"]).reshape(n, *fields[key]["shape"])
        if not np.isfinite(value).all():
            raise ValueError(f"nonfinite export field: {key}")
        arrays[key] = value
        availability[key] = {"available": True, "source": source}
    if sonic_debug is not None:
        host = debug_times.astype(np.float64) / 1e9
    else:
        host = robot["state_monotonic_ns"].astype(np.float64) / 1e9
    sim = robot["time"]
    if debug_times is None and not np.allclose(np.diff(sim), .02, atol=1e-8, rtol=0):
        raise ValueError("50Hz export requires contiguous 50Hz physical samples; no filling gaps")
    for key in ("timestamp", "frame_index", "index", "episode_index", "task_index"):
        value = (np.arange(n) / 50 if key == "timestamp" else
                 np.arange(n) if key in ("frame_index", "index") else np.zeros(n))
        put(key, value, "episode-local nominal 50Hz row grid; original clocks retained in alignment; independent session episode=0")
    put("observation.state", robot["joint_pos"], "measured pre-step Bridge state, SONIC C++ motor order, rad")
    put("action.wbc", robot["target_joint_pos"], "Bridge applied PD equilibrium for this physics interval, rad; not the reference fork's C++ s_t/a_t logger pair")
    put("observation.joint_velocity", robot["joint_vel"], "PhysX measured rad/s")
    put("observation.eef_state", wrist_fk(robot["joint_pos"]), "G1 URDF wrist-link FK in robot base frame, XYZ metres + wxyz per side")
    put("observation.root_orientation", robot["root_quat_wxyz"], "Isaac world, wxyz")
    rotations = Rotation.from_quat(robot["root_quat_wxyz"][:, [1, 2, 3, 0]])
    put("observation.projected_gravity", rotations.inv().apply(np.tile([0., 0., -1.], (n, 1))), "unit gravity in pelvis frame")
    put("observation.base_angular_velocity", robot["root_ang_vel_b"], "pelvis-frame angular velocity, rad/s")
    put("observation.base_acceleration", robot["root_acc_b"], "pelvis-frame finite-difference specific-force estimate, m/s^2; not a physical IMU")
    for key, raw in (("observation.body_monotonic_ns", "state_monotonic_ns"),
                     ("observation.body_timestamp_ns", "state_unix_ns"),
                     ("observation.capture_timestamp_ns", "state_unix_ns"),
                     ("action.body_monotonic_ns", "action_monotonic_ns"),
                     ("observation.control_cycle_index", "control_cycle_index")):
        put(key, robot[raw], f"Bridge {raw}; same host boot")
    valid = (robot["controller_active"].astype(bool) & ~robot["fault_latched"].astype(bool)
             & robot["raw_command_valid"].astype(bool) & (robot["command_age"] <= meta["watchdog"]))
    put("observation.body_valid", valid, "Bridge validity; does not assert the reference C++ body_alignment_version=2 contract")
    # If the patched SONIC binary was used, replace the proxy Bridge action and
    # timing fields with the exact C++ StateLogger snapshot/queued-action pair.
    # This join is causal and bounded; old runs without this stream stay clearly
    # marked as development captures.
    debug_join = None
    if sonic_debug is not None:
        debug_safe = np.arange(n, dtype=np.int64)
        debug_age_s = np.zeros(n, dtype=np.float64)
        debug_ok = np.asarray([valid_debug_row(row) for row in debug_rows], dtype=bool)
        debug_ok &= bridge_join_age <= .1
        if not np.any(debug_ok):
            raise ValueError("SONIC debug stream is present but has no causal action rows")
        def debug_matrix(name, width, dtype=np.float64):
            value = np.zeros((n, width), dtype=dtype)
            for row_number, source_row in enumerate(debug_safe):
                raw = np.asarray(debug_rows[source_row].get(name, []), dtype=dtype).reshape(-1)
                if raw.size == width:
                    value[row_number] = raw
            return value
        def debug_scalar(name, dtype=np.int64, default=-1):
            return np.asarray([debug_rows[i].get(name, default) for i in debug_safe], dtype=dtype)
        debug_action = debug_matrix("body_action", 29)
        debug_previous = debug_matrix("previous_body_action", 29)
        debug_quat = debug_matrix("base_quat", 4)
        debug_gyro = debug_matrix("base_ang_vel", 3)
        debug_accel = debug_matrix("base_accel", 3)
        debug_state = debug_matrix("body_q", 29)
        debug_dq = debug_matrix("body_dq", 29)
        # The reference StateLogger wire arrays use the same named 29-joint
        # order as the pinned dataset schema. Resolve by feature names so a
        # future schema/order change fails visibly instead of silently shifting.
        feature_names = fields["observation.state"]["names"]
        canonical_names = fields["action.wbc"]["names"]
        permutation = [feature_names.index(name) for name in canonical_names]
        debug_state = debug_state[:, permutation]
        debug_dq = debug_dq[:, permutation]
        debug_action = debug_action[:, permutation]
        debug_previous = debug_previous[:, permutation]
        debug_quat = debug_matrix("base_quat", 4)
        for key, value, source in (
            ("observation.state", debug_state, "patched SONIC StateLogger body_q, exact C++ snapshot"),
            ("action.wbc", debug_action, "patched SONIC StateLogger exact queued MotorCommand.q_target"),
            ("observation.joint_velocity", debug_dq, "patched SONIC LowState MotorState.dq at StateLogger snapshot"),
            ("observation.root_orientation", debug_quat, "patched SONIC LowState.imu_state quaternion"),
            ("observation.base_angular_velocity", debug_gyro, "patched SONIC LowState.imu_state gyroscope"),
            ("observation.base_acceleration", debug_accel, "patched SONIC LowState.imu_state accelerometer"),
        ):
            value[~debug_ok] = arrays[key][~debug_ok]
            put(key, value, source)
        rotations = Rotation.from_quat(arrays["observation.root_orientation"][:, [1, 2, 3, 0]])
        put("observation.projected_gravity",
            rotations.inv().apply(np.tile([0., 0., -1.], (n, 1))),
            "projected gravity derived from patched SONIC LowState quaternion")
        put("observation.eef_state", wrist_fk(debug_state),
            "G1 URDF wrist-link FK from exact SONIC StateLogger q snapshot")
        put("observation.body_monotonic_ns", debug_scalar("state_monotonic_ns"), "patched SONIC StateLogger steady-clock snapshot")
        put("observation.body_timestamp_ns", debug_scalar("state_timestamp_ns"), "patched SONIC StateLogger Unix snapshot")
        put("action.body_monotonic_ns", debug_scalar("body_action_monotonic_ns"), "patched SONIC exact queue timestamp")
        put("observation.control_cycle_index", debug_scalar("index"), "patched SONIC StateLogger cycle index")
        put("observation.lowstate_receive_monotonic_ns", debug_scalar("lowstate_receive_monotonic_ns"), "patched SONIC LowState DataBuffer receipt clock")
        put("observation.lowstate_tick", debug_scalar("lowstate_tick"), "patched SONIC raw LowState tick")
        put("observation.previous_body_action", debug_previous, "patched SONIC preceding exact queued target")
        put("observation.previous_body_action_index", debug_scalar("previous_body_action_index"), "patched SONIC preceding logger index")
        put("observation.previous_body_action_monotonic_ns", debug_scalar("previous_body_action_monotonic_ns"), "patched SONIC preceding action clock")
        put("observation.previous_body_action_valid", debug_scalar("previous_body_action_valid", np.int32, 0), "patched SONIC causal validity")
        source_age = (debug_scalar("state_monotonic_ns") - debug_scalar("lowstate_receive_monotonic_ns")).astype(np.float64) / 1e9
        put("observation.body_source_age_s", source_age, "patched SONIC StateLogger snapshot minus LowState receipt")
        put("observation.body_capture_age_s", bridge_join_age, "same-host C++ StateLogger snapshot minus preceding Bridge sample")
        valid = (valid & debug_ok & (source_age >= 0) & (source_age <= .1))
        put("observation.body_valid", valid, "patched SONIC protocol-v2 causal validity")
        mode = debug_scalar("encoder_mode", np.int32, -2)
        put("teleop.stream_mode", (mode == 2).astype(np.int32), "patched SONIC StateLogger encoder mode; POSE maps to reference stream mode 1")
        token_valid = np.zeros(n, dtype=bool)
        for row_number, source_row in enumerate(debug_safe):
            token = np.asarray(debug_rows[source_row].get("token_state", []), dtype=np.float64).reshape(-1)
            if token.size == 64 and np.isfinite(token).all():
                arrays["action.motion_token"][row_number] = token
                token_valid[row_number] = True
        availability["action.motion_token"] = {
            "available": bool(token_valid.all()),
            "source": "patched SONIC StateLogger post-state token on exact C++ cycle" if token_valid.all() else None,
            "reason": None if token_valid.all() else f"missing exact token on {int((~token_valid).sum())} rows; see alignment mask",
        }
        token_rows = np.arange(n, dtype=np.int64)
        token_ages = np.zeros(n, dtype=np.float64)
        modes = np.where(mode == 2, 2, 0).astype(np.int32)
        debug_join = {"rows": debug_safe, "valid": debug_ok, "age_s": debug_age_s,
                      "token_valid": token_valid, "config": debug_config or {}}
        init_quat = np.zeros((n, 4), dtype=np.float64)
        have_init = np.zeros(n, dtype=bool)
        for row_number, source_row in enumerate(debug_safe):
            raw = np.asarray(debug_rows[source_row].get("init_base_quat", []), dtype=np.float64).reshape(-1)
            if raw.size == 4 and np.isfinite(raw).all() and np.linalg.norm(raw) > .9:
                init_quat[row_number] = raw
                have_init[row_number] = True
        if np.any(have_init):
            init_quat[~have_init, 0] = 1
            put("observation.init_base_quat", init_quat, "SONIC heading-state initial base quaternion")
        heading_available = all("delta_heading" in row for row in debug_rows)
        if heading_available:
            put("teleop.delta_heading", debug_scalar("delta_heading", np.float64),
                "exact C++ accumulated heading state; not the per-packet heading increment")
    if sonic_debug is None:
        tokens, token_rows, token_ages = csv_state(run / "sonic_state/token_state.csv", host, 64)
        put("action.motion_token", tokens, "latest preceding recorded SONIC token_state.csv; <=100ms host-clock association, not proven exact DDS-command/token pairing")
        modes = mode_samples(run / "sonic_state/encoder_mode.csv", host)
        token_valid = np.ones(n, dtype=bool)
    else:
        tokens = arrays["action.motion_token"]
    rows = np.searchsorted(human["packet_time"], host, side="right") - 1
    safe = np.maximum(rows, 0)
    age = host - human["packet_time"][safe]
    human_valid = human_reference_valid(rows, age, modes)
    put("teleop.stream_mode", np.where(modes == 2, 1, 0), "reference StreamMode.POSE=1; source SONIC encoder mode=2; other mode=OFF placeholder")
    for field in ("smpl_pose", "smpl_joints", "body_quat_w"):
        values = human[field][safe].reshape(n, -1).copy()
        values[~human_valid] = 0
        if field == "body_quat_w":
            values[~human_valid, 0] = 1
        put("teleop." + field, values, "causal Manager packet-time association; valid only where bridge_alignment.human_valid=true")
    frame = human["source_frame"][safe].copy()
    frame[~human_valid] = -1
    put("teleop.smpl_frame_index", frame, "original Manager stream frame; -1 outside fresh POSE")
    # Recover pose-derived orientation and wrist targets from the original
    # packet row, keeping the same causal/freshness mask as the SMPL fields.
    body_quat = human["body_quat_w"][safe].reshape(n, 4).astype(np.float64)
    if availability["teleop.delta_heading"]["available"]:
        rot6 = mentor_target_orientation(body_quat, arrays["teleop.delta_heading"][:, 0], human_valid)
        put("teleop.target_body_orientation", rot6,
            "reference initial-yaw normalization after exact C++ accumulated heading; column-wise rotation 6D")
    else:
        availability["teleop.target_body_orientation"] = {
            "available": False,
            "reason": "reference yaw-normalized target needs C++ accumulated heading; raw PICO quaternion remains available"}
    human_q = human["joint_pos"][safe].reshape(n, 29).astype(np.float64)
    wrists = np.stack([human_q[:, [19, 20, 21]], human_q[:, [26, 27, 28]]], axis=1).reshape(n, 6)
    wrists[~human_valid] = 0
    put("teleop.left_wrist_joints", wrists[:, :3], "causal PICO SONIC motor target, left wrist roll/pitch/yaw")
    put("teleop.right_wrist_joints", wrists[:, 3:], "causal PICO SONIC motor target, right wrist roll/pitch/yaw")
    if "vr_position" in human:
        vr_position = np.asarray(human["vr_position"])[safe].reshape(n, 9).copy()
        vr_position[~human_valid] = 0
        put("teleop.vr_3pt_position", vr_position, "causal PICO left/right wrist and headset position packet")
    if "vr_orientation" in human:
        vr_quat = np.asarray(human["vr_orientation"])[safe].reshape(n, 3, 4).astype(np.float64)
        vr_rot6 = mentor_rot6d(vr_quat).reshape(n, 18)
        vr_rot6[~human_valid] = 0
        put("teleop.vr_3pt_orientation", vr_rot6, "causal PICO three quaternion orientations converted to reference column-wise rotation 6D")
    # Local raw ordering has thumb opposition then flexion. Reference six-motor
    # order is flexion, opposition, index, middle, ring, little.
    permutation = [1, 0, 2, 3, 4, 5]
    observed, velocity = [], []
    for side in ("left", "right"):
        limits = np.asarray(meta["hand_position_limits"][side]["high"][:6])[permutation]
        if np.any(limits <= 0):
            raise ValueError("hand normalization needs positive recorded active-joint limits")
        observed.append(robot[f"{side}_hand_joint_pos"][:, permutation] / limits)
        velocity.append(robot[f"{side}_hand_joint_vel"][:, permutation] / limits)
    qh, dqh = map(lambda parts: np.concatenate(parts, axis=1), (observed, velocity))
    state_time_ns = (debug_times if debug_times is not None else robot["state_monotonic_ns"])
    action_time_ns = (np.asarray([int(row.get("body_action_monotonic_ns", 0)) for row in debug_rows], dtype=np.int64)
                      if debug_times is not None else state_time_ns.copy())
    hand_state_rows = np.searchsorted(bridge_times, state_time_ns, side="right") - 1
    hand_action_rows = np.searchsorted(bridge_times, action_time_ns, side="right") - 1
    hand_state_safe = np.clip(hand_state_rows, 0, len(bridge_times)-1)
    hand_action_safe = np.clip(hand_action_rows, 0, len(bridge_times)-1)
    def hand_event_rows(side, target_ns, latest_bridge_rows):
        event_ns = np.asarray(bridge_source[f"{side}_hand_command_monotonic_s"], dtype=np.float64) * 1e9
        rows = np.searchsorted(event_ns, target_ns.astype(np.float64), side="right") - 1
        rows = np.minimum(rows, latest_bridge_rows)
        valid = (rows >= 0) & (event_ns[np.clip(rows, 0, len(event_ns)-1)] > 0)
        return np.clip(rows, 0, len(event_ns)-1), valid
    def bridge_hand_values(side, row_index):
        limits = np.asarray(meta["hand_position_limits"][side]["high"][:6])[permutation]
        return bridge_source[f"{side}_hand_target_joint_pos"][row_index][:, permutation] / limits
    # Build true as-of command snapshots at the body snapshot and exact action
    # queue times. Never shift rows and call that command history.
    previous_hand_parts, action_hand_parts, state_hand_valid, state_hand_mono, state_hand_unix, state_hand_count = [], [], [], [], [], []
    action_hand_valid, action_hand_unix, action_hand_mono, action_hand_count = [], [], [], []
    state_event_rows, action_event_rows = [], []
    for side in ("left", "right"):
        sr, sv = hand_event_rows(side, state_time_ns, hand_state_safe)
        ar, av = hand_event_rows(side, action_time_ns, hand_action_safe)
        state_event_rows.append(sr); action_event_rows.append(ar)
        previous_hand_parts.append(bridge_hand_values(side, sr))
        action_hand_parts.append(bridge_hand_values(side, ar))
        state_hand_valid.append(bridge_source[f"{side}_hand_command_valid"][sr].astype(bool) & sv)
        state_hand_mono.append(bridge_source[f"{side}_hand_command_monotonic_s"][sr])
        state_hand_unix.append(bridge_source[f"{side}_hand_command_unix_ns"][sr])
        state_hand_count.append(bridge_source[f"{side}_hand_command_count"][sr])
        action_hand_valid.append(bridge_source[f"{side}_hand_command_valid"][ar].astype(bool) & av)
        action_hand_mono.append(bridge_source[f"{side}_hand_command_monotonic_s"][ar])
        action_hand_unix.append(bridge_source[f"{side}_hand_command_unix_ns"][ar])
        action_hand_count.append(bridge_source[f"{side}_hand_command_count"][ar])
    previous_hand_joint = np.concatenate(previous_hand_parts, axis=1)
    action_hand_joint = np.concatenate(action_hand_parts, axis=1)
    body_hand_valid = np.stack(state_hand_valid, axis=1)
    aligned_hand_valid = np.stack(action_hand_valid, axis=1)
    state_hand_age = host[:, None] - np.stack(state_hand_mono, axis=1)
    action_host = action_time_ns.astype(np.float64) / 1e9
    action_hand_age = action_host[:, None] - np.stack(action_hand_mono, axis=1)
    hand_max_age_s = .1
    body_hand_valid &= (state_hand_age >= 0) & (state_hand_age <= hand_max_age_s)
    aligned_hand_valid &= (action_hand_age >= 0) & (action_hand_age <= hand_max_age_s)
    commanded = [action_hand_joint[:, :6], action_hand_joint[:, 6:]]
    hand_valid = [body_hand_valid[:, 0], body_hand_valid[:, 1]]
    for side, values in zip(("left", "right"), commanded):
        put(f"teleop.{side}_hand_joints", values, "causal Bridge Revo2 target as of exact SONIC action queue time; six normalized active joints")
    for key, value in (("observation.hand_joint_position", qh), ("observation.hand_joint_velocity", dqh),
                       ("action.hand_joint_position", action_hand_joint)):
        put(key, value, "Revo2 six-active-joint normalized simulation values, left then right; not hardware-calibrated BrainCo q")
    put("observation.hand_binary_state", np.stack([x.mean(axis=1) >= .5 for x in observed], axis=1), "derived threshold of measured normalized state; not a grasp-success signal")
    put("action.hand_binary", np.stack([x.mean(axis=1) >= .5 for x in commanded], axis=1), "derived threshold of analog command; original control remains continuous, not binary")
    put("observation.hand_valid", np.stack(hand_valid, axis=1), "physical states sampled locally; command ever-received validity")
    put("action.hand_valid", aligned_hand_valid.all(axis=1), "both sides have causal Bridge command snapshots at C++ action time")
    put("action.hand_enabled", aligned_hand_valid.all(axis=1), "Bridge applied both hand targets at C++ action time")
    # Revo2 has no BrainCo command-history acknowledgement.  We still retain
    # the actual causal Bridge snapshots in the reference-shaped columns so a
    # downstream Revo2-aware trainer can use them, while metadata keeps the
    # distinction explicit and the BrainCo validator remains fail-closed.
    previous_binary = np.stack([x.mean(axis=1) >= .5 for x in previous_hand_parts], axis=1).astype(np.int32)
    hand_binary = np.stack([x.mean(axis=1) >= .5 for x in commanded], axis=1).astype(np.int32)
    previous_valid = body_hand_valid.all(axis=1).astype(np.int32)
    put("observation.previous_hand_action", previous_binary, "Bridge Revo2 target selected as of body state time, binary threshold; not BrainCo acknowledgement")
    put("observation.previous_hand_action_joint_position", previous_hand_joint, "Bridge Revo2 normalized analog target selected as of body state time")
    put("action.aligned_hand", hand_binary, "Bridge Revo2 target selected as of exact SONIC action queue time, binary threshold")
    put("action.aligned_hand_joint_position", action_hand_joint, "Bridge Revo2 normalized analog target selected as of exact SONIC action queue time")
    put("observation.previous_hand_action_valid", previous_valid, "both Bridge Revo2 snapshots available at body state time")
    put("action.aligned_hand_valid", aligned_hand_valid.all(axis=1).astype(np.int32), "both Bridge Revo2 snapshots available at action queue time")
    state_hand_mono_s = np.stack(state_hand_mono, axis=1)
    state_hand_mono_ns = np.rint(state_hand_mono_s * 1e9).astype(np.int64)
    state_hand_unix = np.stack(state_hand_unix, axis=1)
    state_hand_count = np.stack(state_hand_count, axis=1)
    action_hand_mono_s = np.stack(action_hand_mono, axis=1)
    action_hand_mono_ns = np.rint(action_hand_mono_s * 1e9).astype(np.int64)
    action_hand_unix = np.stack(action_hand_unix, axis=1)
    action_hand_count = np.stack(action_hand_count, axis=1)
    previous_mono = np.max(state_hand_mono_ns, axis=1)
    previous_unix = np.max(state_hand_unix, axis=1)
    aligned_mono = np.max(action_hand_mono_ns, axis=1)
    aligned_unix = np.max(action_hand_unix, axis=1)
    put("observation.hand_received_timestamp_ns", state_hand_unix, "Bridge workstation receipt time for Revo2 callbacks, per side")
    put("observation.hand_age_s", np.where(state_hand_mono_s > 0, state_hand_age, -1.), "same-host body snapshot minus Revo2 command receipt; Revo2 validity gate 100 ms")
    put("action.hand_timestamp_ns", aligned_unix, "latest Bridge Revo2 callback timestamp as of SONIC action queue")
    if "timestamp_realtime" in human:
        pico_realtime = np.asarray(human["timestamp_realtime"], dtype=np.float64).reshape(-1)
        if len(pico_realtime) != len(human["packet_time"]):
            raise ValueError("PICO realtime timestamp count differs from packet count")
        pico_ns = np.full(n, -1, dtype=np.int64)
        pico_time_valid = np.isfinite(pico_realtime[safe]) & (pico_realtime[safe] > 0)
        usable_pico_time = human_valid & pico_time_valid
        pico_ns[usable_pico_time] = np.rint(pico_realtime[safe][usable_pico_time] * 1e9).astype(np.int64)
        put("teleop.hand_pico_timestamp_ns", pico_ns, "PICO packet Unix timestamp; input-packet time, not a hand write acknowledgement")
    else:
        arrays["teleop.hand_pico_timestamp_ns"][:] = -1
        availability["teleop.hand_pico_timestamp_ns"] = {
            "available": False, "reason": "source Manager packets lack timestamp_realtime; -1 is an explicit sentinel"}
    put("observation.previous_hand_action_monotonic_ns", previous_mono, "latest Bridge Revo2 callback clock as of body state")
    put("observation.previous_hand_action_timestamp_ns", previous_unix, "latest Bridge Revo2 callback Unix clock as of body state")
    put("observation.previous_hand_action_sequence", np.max(state_hand_count, axis=1), "Bridge callback count snapshot, not BrainCo sequence")
    put("action.aligned_hand_monotonic_ns", aligned_mono, "latest Bridge Revo2 callback clock as of SONIC action queue")
    put("action.aligned_hand_timestamp_ns", aligned_unix, "latest Bridge Revo2 callback Unix clock as of SONIC action queue")
    put("action.aligned_hand_sequence", np.max(action_hand_count, axis=1), "Bridge callback count snapshot, not BrainCo sequence")
    session_id = json.loads((run / "session.json").read_text()).get("session_id", "")
    session_number = int(session_id[:15], 16) if len(session_id) >= 15 else -1
    if session_number == 0:
        session_number = 1
    put("observation.previous_hand_action_session_id", np.full(n, session_number, dtype=np.int64), "capture session identifier hash; not BrainCo DDS session")
    put("action.aligned_hand_session_id", np.full(n, session_number, dtype=np.int64), "capture session identifier hash; not BrainCo DDS session")
    with (run / "ego_view.timestamps.csv").open() as stream:
        camera = list(csv.DictReader(stream))
    ct = np.array([float(r["host_monotonic_s"]) for r in camera])
    from upstream_camera import select_camera_frames
    cr = select_camera_frames(camera, host, sim, meta.get('ego_render_schedule'))
    put("observation.camera_timestamp", [float(camera[i]["unix_time_s"]) for i in cr], "Unix seconds of selected original simulated-camera frame")
    # Rotation offset is not emitted by the current Bridge. Keep an explicit
    # structural sentinel; measured PICO orientation and (when available) the
    # SONIC initial quaternion were populated above.
    arrays["observation.cpp_rotation_offset"][:, 0] = 1
    if not availability["observation.init_base_quat"]["available"]:
        arrays["observation.init_base_quat"][:, 0] = 1
    alignment = {"host_monotonic_s": host, "sim_time_s": sim, "source_bridge_row": indices,
                 "human_row": rows, "human_valid": human_valid, "human_age_s": age,
                 "camera_source_frame": cr, "camera_age_s": host - ct[cr],
                 "sonic_token_row": token_rows, "sonic_token_age_s": token_ages,
                 "sonic_token_valid": token_valid,
                 "source_encoder_mode": modes,
                 "sonic_debug_join": debug_join}
    exact_body = debug_join is not None
    semantics = {"reference": json.loads((REFERENCE / "reference.json").read_text()),
                 "field_availability": availability,
                 "body_alignment": ("patched SONIC C++ StateLogger protocol v2: exact s_t/a_t pair, preceding action, LowState receipt clock and raw tick"
                                    if exact_body else
                                    "sampled Bridge physical state and actually applied PD target; not reference C++ same-cycle protocol v2; no exact latent decoder replay claim"),
                 "hand_recording": {"schema": "revo2_sim_6_active_normalized",
                     "reference_feature_names": "brainco_motor_0..5 retained for structural compatibility; these are Revo2 mappings, not BrainCo hardware",
                     "source_active_permutation": permutation, "limits": meta["hand_position_limits"],
                     "full_11_joint_rad": "bridge.npz / prepared/robot_trajectory.npz; no information lost",
                     "control": "capture-time mapping recorded in source provenance; current main trigger controls full-hand grasp and side grip is recording-only; action.hand_binary is derived only",
                     "history": "Bridge command snapshots, not BrainCo successful-write acknowledgement",
                     "max_asof_age_s": hand_max_age_s,
                     "pico_timestamp": "source Manager packet Unix timestamp; not a hand command acknowledgement"},
                 "camera_alignment": ("upstream pre-action physics grid, native 25Hz frame held into 50fps encoded stream; source state and render-completion clocks retained"
                                      if meta.get('ego_render_schedule') == 'upstream' else
                                      "last actually captured frame at/before state monotonic time, held into 50fps encoded stream; original timestamps retained"),
                 "camera_geometry": meta.get("ego_camera"),
                 "task_outcome": "unreviewed; task instruction does not certify successful execution",
                 "world_recording": {"available_in_raw_capture": bool(meta.get("world_trace_required")),
                     "path": "world_trace.npz" if meta.get("world_trace_required") else None,
                     "sampling_phase": "pre_action" if meta.get('ego_render_schedule') == 'upstream' else "post_action",
                     "semantics": "actual PhysX rigid-body poses and foot contact forces; explicit sampling-phase simulation clock; no automatic task-success label"},
                 "hand_appearance": {side: setup.get("appearance") for side, setup in meta.get("hand_setup", {}).items()
                                     if setup.get("appearance") is not None},
                 "timestamp_domains": "timestamp: episode-local nominal 50Hz row grid; bridge_alignment.sim_time_s: original physics time; *_monotonic_ns: host boot; *_timestamp_ns: Unix when available; unavailable scalars=-1",
                 "human_alignment_policy": {"max_reference_age_s": HUMAN_REFERENCE_MAX_AGE_S,
                     "stale_rows": "human_valid=false in bridge_alignment.npz; teleop fields use sentinels, smpl_frame_index=-1"},
                 "source_run": str(run), "startup_samples_excluded": excluded,
                 "physics_contract": meta.get("physics_contract"),
                 "task_initialization": meta.get("startup_alignment"),
                 "training_ready": False, "body_protocol_v2": exact_body,
                 "training_compatibility": {
                     "lerobot_v2_1_layout": True,
                     "revo2_aware_imitation": "candidate; review task labels and validity masks before training",
                     "mentor_semanticdpc_converter": "blocked: historical capture lacks the C++ same-cycle body protocol v2"
                         if not exact_body else "requires the pinned Revo2 hand-domain adapter; not a BrainCo-equivalent episode",
                 },
                 "training_blockers": ([] if exact_body else ["sonic_debug_protocol_v2_missing"])
                 + ["revo2_hand_history_not_brainco_write_ack"]
                 + (["task_success_not_annotated"] if meta.get('world_trace_required') else
                    ["task_success_and_object_state_not_recorded"]),
                 "stage": "development"}
    semantics["lerobot_schema_mode"] = "mentor_2609_exact"
    semantics["lerobot_training_only_fields"] = list(TRAINING_ONLY_FIELDS)
    if exact_body:
        config = debug_config or {}
        if str(config.get("body_action_alignment_version", "")) != "2":
            raise ValueError("SONIC protocol-v2 stream is missing its matching robot_config contract")
        for key in ("body_recording_default_q_hardware_json", "body_recording_action_scale_hardware_json"):
            if key not in config:
                raise ValueError(f"SONIC robot_config is missing {key}")
            values = np.asarray(json.loads(config[key]), dtype=np.float64)
            if values.shape != (29,) or not np.isfinite(values).all():
                raise ValueError(f"invalid SONIC robot_config array: {key}")
        semantics["body_action_alignment_version"] = 2
        if config.get('simulation_startup_reference'):
            semantics['simulation_startup'] = {
                'method': 'measured initial-pose reference, ten frozen policy cycles, reference blend to live input',
                'reference_cycles': config['simulation_startup_reference_cycles'],
                'startup_rows_in_export': False, 'initial_pose_changed': False}
        semantics["body_recording_default_q_hardware_json"] = config["body_recording_default_q_hardware_json"]
        semantics["body_recording_action_scale_hardware_json"] = config["body_recording_action_scale_hardware_json"]
    return info, arrays, alignment, semantics


def encode_video(source, selected, destination, width=640, height=360):
    import cv2
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError(f"cannot decode {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-v", "error", "-n", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
               "-r", "50", "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination)]
    # The finalizer defers terminal exit signals while persisting saved data.
    # Keep the encoder outside that terminal's process group as well.
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=True)
    frame = None
    index = -1
    try:
        for wanted in selected:
            while index < wanted:
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f"video ended before original frame {wanted}")
                index += 1
            if frame.shape[:2] != (height, width):
                raise ValueError(f"camera dimensions {frame.shape[:2]} differ from reference {(height, width)}")
            encoder.stdin.write(frame.tobytes())
        encoder.stdin.close()
        error = encoder.stderr.read().decode(errors="replace")
        code = encoder.wait(timeout=60)
        if code:
            raise RuntimeError(f"ffmpeg failed {code}: {error}")
    finally:
        cap.release()
        if encoder.poll() is None:
            encoder.kill()
            encoder.wait()
        if not encoder.stdin.closed:
            encoder.stdin.close()
        encoder.stderr.close()


def build_training_artifact(destination, arrays, alignment, availability):
    """Write a causally filtered Revo2-aware BC pack alongside LeRobot data."""
    import hashlib
    n = len(arrays["timestamp"])
    masks = {
        "body_valid": arrays["observation.body_valid"][:, 0].astype(bool),
        "human_valid": np.asarray(alignment["human_valid"], dtype=bool),
        "pose_mode": arrays["teleop.stream_mode"][:, 0] == 1,
        "hand_state_valid": arrays["observation.hand_valid"].astype(bool).all(axis=1),
        "hand_action_valid": arrays["action.hand_valid"][:, 0].astype(bool),
        "camera_fresh": np.asarray(alignment["camera_age_s"]) <= .1,
    }
    valid = np.logical_and.reduce(list(masks.values()))
    selected = np.flatnonzero(valid)
    if not len(selected):
        raise ValueError("no rows satisfy body, fresh PICO POSE, Revo2, and camera validity gates")
    cycle = arrays["observation.control_cycle_index"][:, 0].astype(np.int64)
    mono = arrays["observation.body_monotonic_ns"][:, 0].astype(np.int64)
    exact_body = alignment.get("sonic_debug_join") is not None
    continuity_time = mono.astype(np.float64) / 1e9 if exact_body else np.asarray(alignment["sim_time_s"], dtype=np.float64)
    continuity_clock = "SONIC StateLogger same-host monotonic clock" if exact_body else "original Bridge physics clock"
    cycle_diffs = np.diff(cycle[selected])
    positive_diffs = cycle_diffs[cycle_diffs > 0]
    expected_cycle_step = int(np.median(positive_diffs)) if len(positive_diffs) else 1
    segment = np.full(n, -1, dtype=np.int64)
    segment_id = -1
    prev = None
    for row in selected:
        contiguous = (prev is not None and row == prev + 1
                     and cycle[row] - cycle[prev] == expected_cycle_step
                     and .01 <= continuity_time[row] - continuity_time[prev] <= .03
                     and mono[row] > mono[prev])
        if not contiguous:
            segment_id += 1
        segment[row] = segment_id
        prev = row
    keys = {
        "observation.state": "robot_q_rad",
        "observation.joint_velocity": "robot_dq_rad_s",
        "observation.root_orientation": "robot_root_quat_wxyz",
        "observation.projected_gravity": "robot_projected_gravity",
        "observation.base_angular_velocity": "robot_base_gyro_rad_s",
        "observation.base_acceleration": "robot_base_acceleration_sim_estimate",
        "teleop.smpl_pose": "pico_smpl_pose",
        "teleop.smpl_joints": "pico_smpl_joints",
        "teleop.body_quat_w": "pico_body_quat_wxyz",
        "teleop.target_body_orientation": "pico_target_body_rotation_6d",
        "teleop.vr_3pt_position": "pico_vr_3pt_position",
        "teleop.vr_3pt_orientation": "pico_vr_3pt_rotation_6d",
        "teleop.left_wrist_joints": "pico_left_wrist_target_rad",
        "teleop.right_wrist_joints": "pico_right_wrist_target_rad",
        "action.wbc": "robot_body_target_rad",
        "action.hand_joint_position": "robot_revo2_target_normalized",
        "teleop.left_hand_joints": "pico_left_hand_input_normalized",
        "teleop.right_hand_joints": "pico_right_hand_input_normalized",
        "observation.previous_hand_action_joint_position": "previous_revo2_target_normalized",
        "action.aligned_hand_joint_position": "revo2_action_time_target_normalized",
    }
    selected_keys = {key: name for key, name in keys.items()
                     if key in arrays and availability.get(key, {}).get("available", False)}
    required_inputs = ("observation.state", "observation.joint_velocity", "teleop.smpl_joints",
                       "action.wbc", "action.hand_joint_position")
    missing = [key for key in required_inputs if key not in selected_keys]
    if missing:
        raise ValueError(f"required training signals unavailable: {missing}")
    training = destination / "training"
    training.mkdir()
    pack = training / "episode_000000_revo2_pose_bc.npz"
    pack_arrays = {name: arrays[key][selected] for key, name in selected_keys.items()}
    pack_arrays.update(source_row_index=selected,
                       source_control_cycle_index=cycle[selected],
                       source_body_monotonic_ns=mono[selected],
                       source_human_frame_index=arrays["teleop.smpl_frame_index"][selected, 0],
                       source_camera_frame_index=np.asarray(alignment["camera_source_frame"])[selected],
                       segment_id=segment[selected])
    with pack.open("xb") as stream:
        np.savez_compressed(stream, **pack_arrays)
        stream.flush()
        os.fsync(stream.fileno())
    counts = {name: int(mask.sum()) for name, mask in masks.items()}
    digest = hashlib.sha256(pack.read_bytes()).hexdigest()
    manifest = {
        "format": "g1_pico_revo2_pose_bc_v1",
        "source_run": str(alignment.get("source_run", "")),
        "source_lerobot_directory": "../",
        "source_video": "../videos/chunk-000/observation.images.ego_view/episode_000000.mp4",
        "source_rows": n,
        "selected_rows": int(len(selected)),
        "validity_gates": counts,
        "continuous_segments": int(segment_id + 1),
        "expected_bridge_cycle_step": expected_cycle_step,
        "validity_policy": "all gates required; continuous source rows, stable control-cycle step, 10-30ms on the declared continuity clock, and increasing same-host monotonic time",
        "continuity_clock": continuity_clock,
        "training_scope": "PICO POSE to simulated G1/Revo2 behavior cloning; not task-success, contact, or object-state supervision",
        "body_action_semantics": (
            "patched SONIC protocol-v2 exact StateLogger action paired to current state"
            if availability.get("observation.previous_body_action_valid", {}).get("available", False)
            else "Bridge PD target sampled for this physical interval; not the later SONIC C++ protocol-v2 exact queue pairing"),
        "hand_semantics": "Revo2 normalized analog targets; Bridge as-of snapshots, not BrainCo DDS write acknowledgements",
        "fields": selected_keys,
        "artifact": pack.name,
        "sha256": digest,
        "ready_for_revo2_aware_pose_bc": bool(len(selected) >= 10),
        "direct_mentor_semanticdpc_conversion": False,
        "limitations": ["not a BrainCo hand dataset", "no task-success label; any recorded object trajectories remain in raw world_trace.npz, outside this BC tensor pack",
                        "does not contain world-space PICO root translation", "simulated base acceleration is an estimate"],
    }
    atomic_json(training / "manifest.json", manifest)
    if not manifest["ready_for_revo2_aware_pose_bc"]:
        raise ValueError(f"only {len(selected)} rows passed BC gates; at least 10 required")
    return manifest


def validate_training_artifact(destination):
    import hashlib
    destination = Path(destination)
    training = destination / "training"
    manifest = json.loads((training / "manifest.json").read_text())
    path = training / manifest["artifact"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != manifest["sha256"]:
        raise ValueError("training artifact SHA256 mismatch")
    with np.load(path, allow_pickle=False) as pack:
        arrays = {key: pack[key] for key in pack.files}
    rows = arrays["source_row_index"]
    if rows.ndim != 1 or len(rows) != manifest["selected_rows"] or np.any(np.diff(rows) <= 0):
        raise ValueError("training source-row index is invalid")
    n = len(rows)
    for key, value in arrays.items():
        if value.ndim == 0 or value.shape[0] != n:
            raise ValueError(f"training array row count mismatch: {key}")
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise ValueError(f"training array has non-finite values: {key}")
    seg = arrays["segment_id"]
    if np.any(seg < 0) or np.any(np.diff(seg) < 0) or int(seg[-1] + 1) != manifest["continuous_segments"]:
        raise ValueError("training segments are invalid")
    return {"rows": n, "segments": int(seg[-1] + 1), "sha256": digest,
            "format": manifest["format"], "ready_for_revo2_aware_pose_bc": manifest["ready_for_revo2_aware_pose_bc"]}


def export_semanticdpc_artifact(destination, info, table):
    """Run the pinned tensor converter only when body protocol v2 is present."""
    import hashlib
    training = Path(destination) / "training/semanticdpc_0826"
    training.mkdir(parents=True, exist_ok=True)
    config = info.get("script_config", {})
    contract_path = REFERENCE / "gear_sonic/data/semanticdpc_contract.json"
    contract = json.loads(contract_path.read_text())
    base = {
        "format": "semanticdpc_0826_tensor_layout_revo2_adapter_v1",
        "reference_commit": contract["semanticdpc_commit"],
        "contract_sha256": __import__("hashlib").sha256(contract_path.read_bytes()).hexdigest(),
        "hand_domain": "Revo2 simulator commands thresholded to the contract's two binary hand channels",
        "brainco_equivalent": False,
        "task_success_supervision": False,
    }
    if config.get("body_action_alignment_version") != 2:
        manifest = {**base, "status": "blocked", "reason": "historical capture lacks C++ body protocol v2",
                    "valid_rows": 0, "full_history_rows": 0, "segments": 0,
                    "artifact": None, "sha256": None,
                    "ready_for_revo2_semanticdpc_finetuning": False}
        atomic_json(training / "manifest.json", manifest)
        return manifest
    reference_data = str(REFERENCE / "gear_sonic/data")
    if reference_data not in sys.path:
        sys.path.insert(0, reference_data)
    from semanticdpc_conversion import convert_episode
    converted = convert_episode(table.to_pandas(), info)
    valid_rows = int(converted["sample_valid"].sum())
    full_history_rows = int(converted["history_mask"].all(axis=1).sum())
    path = training / "episode_000000.npz"
    with path.open("xb") as stream:
        np.savez_compressed(stream, **converted)
        stream.flush()
        os.fsync(stream.fileno())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {**base, "status": "complete", "reason": None,
                "source_rows": len(table), "valid_rows": valid_rows,
                "full_history_rows": full_history_rows,
                "segments": len(set(converted["segment_id"][converted["segment_id"] >= 0].tolist())),
                "tensor_shapes": {key: list(value.shape) for key, value in converted.items()
                                  if hasattr(value, "shape")},
                "artifact": path.name, "sha256": digest,
                "ready_for_revo2_semanticdpc_finetuning": valid_rows >= 10,
                "ready_for_brainco_checkpoint_without_adaptation": False,
                "limitations": ["Revo2 hand-domain adapter; not BrainCo successful-write history",
                                "task/object success is not labeled"]}
    atomic_json(training / "manifest.json", manifest)
    return manifest


def validate_semanticdpc_artifact(destination):
    import hashlib
    path = Path(destination) / "training/semanticdpc_0826/manifest.json"
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text())
    if manifest["status"] == "blocked":
        if manifest["artifact"] is not None or manifest["valid_rows"] != 0:
            raise ValueError("blocked SemanticDPC artifact contains training data")
        return {"status": "blocked", "reason": manifest["reason"]}
    artifact = path.parent / manifest["artifact"]
    if hashlib.sha256(artifact.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("SemanticDPC tensor artifact SHA256 mismatch")
    with np.load(artifact, allow_pickle=False) as tensors:
        shapes = {key: list(tensors[key].shape) for key in tensors.files}
        valid_rows = int(tensors["sample_valid"].sum())
        full_history = int(tensors["history_mask"].all(axis=1).sum())
        if shapes != manifest["tensor_shapes"] or valid_rows != manifest["valid_rows"]:
            raise ValueError("SemanticDPC tensor dimensions or validity count changed")
        if shapes.get("state64", [None, None])[1] != 64 or shapes.get("history10", [None, None, None])[1:] != [10, 95] or shapes.get("action31", [None, None])[1] != 31:
            raise ValueError("SemanticDPC tensors do not match the pinned contract")
        if full_history != manifest["full_history_rows"]:
            raise ValueError("SemanticDPC full-history count changed")
    return {"status": manifest["status"], "valid_rows": valid_rows,
            "full_history_rows": full_history, "segments": manifest["segments"],
            "sha256": manifest["sha256"],
            "ready_for_revo2_semanticdpc_finetuning": manifest["ready_for_revo2_semanticdpc_finetuning"]}


def export(run, task="PICO SONIC POSE development capture", output=None, schema_mode="mentor"):
    import pyarrow as pa
    import pyarrow.parquet as pq
    run = Path(run).resolve()
    destination = Path(output).expanduser().resolve() if output else run
    if destination == run:
        for name in ("data", "meta", "videos", "video"):
            if (destination / name).exists():
                raise FileExistsError(f"refusing to overwrite dataset: {destination / name}")
    elif destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty destination: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    info, arrays, alignment, semantics = build_arrays(run)
    semantics['task_instruction'] = task
    training_availability = dict(semantics["field_availability"])
    if schema_mode not in ("mentor", "bridge"):
        raise ValueError(f"unsupported LeRobot schema mode: {schema_mode}")
    semantics = data_standard.capture_semantics(semantics)
    semantics['dataset_statistics_format'] = data_standard.DATASET_STATS_FORMAT
    semantics['dataset_statistics_method'] = {'quantiles': 'linear, all dataset rows',
        'fingerprint': 'SHA256 of compact JSON [field,dtype,shape], NUL, little-endian C-order column bytes'}
    n = len(arrays["timestamp"])
    meta = destination / "meta"
    meta.mkdir()
    atomic_json(meta / "export_status.json", {"complete": False, "stage": "development"})
    data = destination / info["data_path"].format(episode_chunk=0, episode_index=0)
    data.parent.mkdir(parents=True)
    columns = {}
    hf_features = {}
    stats = {}
    export_arrays = {key: values for key, values in arrays.items() if key in info["features"]}
    for key, values in export_arrays.items():
        dtype = pa.from_numpy_dtype(values.dtype)
        value_feature = {"dtype": str(values.dtype), "_type": "Value"}
        if values.shape[1] == 1:
            columns[key] = pa.array(values[:, 0], type=dtype)
            hf_features[key] = value_feature
        else:
            columns[key] = pa.array(values.tolist(), type=pa.list_(dtype, values.shape[1]))
            hf_features[key] = {"feature": value_feature, "length": values.shape[1], "_type": "Sequence"}
        v = values.astype(np.float64)
        stats[key] = {"min": v.min(axis=0).tolist(), "max": v.max(axis=0).tolist(),
                      "mean": v.mean(axis=0).tolist(), "std": v.std(axis=0).tolist(), "count": [n]}
    table = pa.table(columns).replace_schema_metadata({
        b"huggingface": json.dumps({"info": {"features": hf_features}}).encode()})
    pq.write_table(table, data, compression="snappy")
    video = destination / info["video_path"].format(episode_chunk=0, episode_index=0, video_key=VIDEO_KEY)
    encode_video(run / "ego_view.mp4", alignment["camera_source_frame"], video)
    info["features"][VIDEO_KEY]["info"] = {"video.height": 360, "video.width": 640,
        "video.codec": "h264", "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
        "video.fps": 50, "video.channels": 3, "has_audio": False}
    stats[VIDEO_KEY] = data_standard.video_statistics(video)
    if stats[VIDEO_KEY]["count"] != [n]:
        raise ValueError("video/data count mismatch during statistics")
    info.update(robot_type="g1_revo2_isaac_sim", total_episodes=1, total_frames=n,
                total_tasks=1, total_videos=1, total_chunks=1, splits={"train": "0:1"},
                script_config=semantics, discarded_episode_indices=[])
    atomic_json(meta / "info.json", info)
    from recording_window import recording_window
    window = recording_window(run)
    if window is not None:
        atomic_json(meta / 'recording_control.json', window)
    atomic_json(meta / "modality.json", data_standard.modality())
    atomic_json(meta / "stats.json", data_standard.dataset_statistics(export_arrays))
    atomic_json(meta / "relative_stats.json", {"__fingerprints__": {}})
    for filename, record in (("tasks.jsonl", {"task_index": 0, "task": task}),
                             ("episodes.jsonl", {"episode_index": 0, "tasks": [task], "length": n}),
                             ("episodes_stats.jsonl", {"episode_index": 0, "stats": stats})):
        with (meta / filename).open("x") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    with (meta / "bridge_alignment.npz").open("xb") as stream:
        np.savez_compressed(stream, **alignment)
        stream.flush()
        os.fsync(stream.fileno())
    # The fork's canonical directory is plural. A zero-copy relative alias also
    # supports the user's requested singular spelling, with no duplicate movies.
    (destination / "video").symlink_to("videos", target_is_directory=True)
    result = validate(destination)
    training_manifest = build_training_artifact(destination, arrays,
        {**alignment, "source_run": str(run)}, training_availability)
    training_validation = validate_training_artifact(destination)
    result["training_validation"] = training_validation
    semanticdpc_manifest = export_semanticdpc_artifact(destination, info, table)
    semanticdpc_validation = validate_semanticdpc_artifact(destination)
    result["semanticdpc_0826"] = semanticdpc_validation
    info["script_config"]["training_artifacts"] = {
        "manifest": "../training/manifest.json",
        "revo2_pose_behavior_cloning_rows": training_manifest["selected_rows"],
        "ready_for_revo2_aware_pose_bc": training_manifest["ready_for_revo2_aware_pose_bc"],
        "direct_mentor_semanticdpc_conversion": False,
        "semanticdpc_0826_revo2_adapter": semanticdpc_validation,
    }
    atomic_json(meta / "info.json", info)
    for path in (data, video):
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
        sync_directory(path.parent)
    atomic_json(meta / "export_status.json", {"complete": True, "stage": "development",
                "training_ready": False, "training_artifacts": training_manifest,
                "training_validation": training_validation,
                "semanticdpc_0826": semanticdpc_manifest,
                "validation": result})
    return result


def validate(run):
    import cv2
    import pyarrow as pa
    import pyarrow.parquet as pq
    run = Path(run)
    info = json.loads((run / "meta/info.json").read_text())
    expected = data_standard.expected_template(info)
    for key in ("codebase_version", "fps", "chunks_size", "data_path", "video_path"):
        if info.get(key) != expected[key]:
            raise ValueError(f"reference metadata differs: {key}")
    for key, feature in expected["features"].items():
        actual_feature = dict(info.get("features", {}).get(key, {}))
        actual_feature.pop("info", None)
        expected_feature = {k: v for k, v in feature.items() if k != "info"}
        if actual_feature != expected_feature:
            raise ValueError(f"reference feature metadata differs: {key}")
    if set(info["features"]) != set(expected["features"]):
        raise ValueError("reference feature metadata has extra keys")
    modality = json.loads((run / "meta/modality.json").read_text())
    if modality != data_standard.modality():
        raise ValueError("reference modality metadata differs")
    # Source feature names, shapes and scalar/vector dtypes are authoritative.
    fields = {k: v for k, v in expected["features"].items() if v["dtype"] != "video"}
    table = pq.read_table(run / info["data_path"].format(episode_chunk=0, episode_index=0))
    if set(table.column_names) != set(fields) or len(table) != info["total_frames"]:
        raise ValueError("Parquet columns/count differ from reference schema")
    for key, feature in fields.items():
        dtype = pa.from_numpy_dtype(np.dtype(feature["dtype"]))
        actual = table[key].type
        if actual != (dtype if feature["shape"] == [1] else pa.list_(dtype, feature["shape"][0])):
            raise ValueError(f"reference dtype differs for {key}: {actual}")
        v = np.asarray(table[key].to_pylist(), dtype=feature["dtype"]).reshape(len(table), -1)
        if v.shape[1:] != tuple(feature["shape"]) or not np.isfinite(v).all():
            raise ValueError(f"invalid field {key}")
    hf_features = json.loads(table.schema.metadata[b"huggingface"])["info"]["features"]
    for key, feature in fields.items():
        value_feature = {"dtype": feature["dtype"], "_type": "Value"}
        expected_hf = value_feature if feature["shape"] == [1] else {
            "feature": value_feature, "length": feature["shape"][0], "_type": "Sequence"}
        if hf_features[key] != expected_hf:
            raise ValueError(f"Hugging Face feature metadata differs: {key}")
    for key in ("frame_index", "index"):
        if not np.array_equal(np.array(table[key]), np.arange(len(table))):
            raise ValueError(f"invalid {key}")
    if any(np.any(np.array(table[key]) != 0) for key in ("episode_index", "task_index")):
        raise ValueError("single-session episode/task indices must be zero")
    def jsonl(name):
        return [json.loads(line) for line in (run / "meta" / name).read_text().splitlines() if line]
    episodes, tasks, episode_stats = [jsonl(name) for name in ("episodes.jsonl", "tasks.jsonl", "episodes_stats.jsonl")]
    if (info["total_episodes"] != 1 or info["total_tasks"] != 1 or info["total_videos"] != 1
            or len(episodes) != 1 or episodes[0]["episode_index"] != 0 or episodes[0]["length"] != len(table)
            or len(tasks) != 1 or tasks[0]["task_index"] != 0 or episodes[0]["tasks"] != [tasks[0]["task"]]
            or len(episode_stats) != 1 or episode_stats[0]["episode_index"] != 0):
        raise ValueError("episode/task metadata count mismatch")
    aggregate = json.loads((run / "meta/stats.json").read_text())
    stats = episode_stats[0]['stats']
    expected_stats = set(expected["features"]) if data_standard.current(info) else set(fields)
    if set(stats) != expected_stats:
        raise ValueError("statistics metadata differs")
    if info['script_config'].get('dataset_statistics_format') == data_standard.DATASET_STATS_FORMAT:
        data_standard.validate_dataset_statistics(aggregate, table)
    elif aggregate != stats:
        raise ValueError('Historical aggregate statistics metadata differs')
    for key, feature in fields.items():
        for statistic in ("min", "max", "mean", "std"):
            values = np.asarray(stats[key][statistic])
            if values.shape != tuple(feature["shape"]) or not np.isfinite(values).all():
                raise ValueError(f"invalid statistics: {key}/{statistic}")
        if stats[key]["count"] != [len(table)]:
            raise ValueError(f"invalid statistics count: {key}")
    if not (run / "video").is_symlink() or (run / "video").resolve() != (run / "videos").resolve():
        raise ValueError("missing relative video -> videos compatibility alias")
    if not np.array_equal(np.array(table["timestamp"]), (np.arange(len(table)) / 50).astype(np.float32)):
        raise ValueError("invalid 50Hz timestamp grid")
    path = run / info["video_path"].format(episode_chunk=0, episode_index=0, video_key=VIDEO_KEY)
    cap = cv2.VideoCapture(str(path))
    count = 0
    rgb_stats = data_standard.RGBStatistics()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape != (360, 640, 3):
                raise ValueError("encoded frame dimensions differ")
            rgb_stats.add_bgr(frame)
            count += 1
    finally:
        cap.release()
    if count != len(table):
        raise ValueError(f"video/data count mismatch {count}/{len(table)}")
    if data_standard.current(info):
        data_standard.validate_image_statistics(stats[VIDEO_KEY], rgb_stats.result(), len(table))
    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,pix_fmt,r_frame_rate,nb_frames", "-of", "json", str(path)], text=True))["streams"][0]
    if probe["codec_name"] != "h264" or probe["pix_fmt"] != "yuv420p" or probe["r_frame_rate"] != "50/1":
        raise ValueError(f"video codec/FPS mismatch: {probe}")
    training_validation = validate_training_artifact(run) if (run / "training/manifest.json").exists() else None
    semanticdpc_validation = validate_semanticdpc_artifact(run)
    return {"rows": len(table), "numeric_columns": len(fields), "decoded_video_frames": count,
            "fps": 50, "canonical_video_directory": "videos", "video_alias": "video -> videos",
            "parquet_storage": "Hugging Face fixed-size Sequence vectors + scalar Value fields",
            "training_validation": training_validation,
            "semanticdpc_0826": semanticdpc_validation}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, help="write a new standalone dataset directory; never overwrite")
    parser.add_argument("--task", default="PICO SONIC POSE development capture")
    parser.add_argument("--schema", choices=("mentor", "bridge"), default="mentor",
                        help="both modes export the current configs/data_standard contract, including analog hand history")
    args = parser.parse_args()
    print(json.dumps(validate(args.run) if args.check else export(args.run, args.task, args.output, args.schema), indent=2))


if __name__ == "__main__":
    main()
