"""BrainCo dataset contract and read-only manager telemetry decoding."""

import json
import time

import numpy as np

from gear_sonic.utils.teleop.brainco_poses import (
    SIDES, JOINT_ORDER, load_poses, validate_poses, pose_targets,
)
MOTOR_NAMES = [f"{side}_brainco_motor_{i}" for side in SIDES for i in range(6)]


def configure_brainco_schema(features, modality, robot_model):
    """Keep body FK in radians; record BrainCo normalized motors separately.

    The stock FK model has Dex3 fingers. Those fingers must not appear as
    observations or actions of the BrainCo robot. Wrist FK is unchanged.
    """
    indices = robot_model.get_joint_group_indices("body")
    names = [robot_model.joint_names[i] for i in indices]
    for key in ("observation.state", "action.wbc"):
        features[key] = {"dtype": "float64", "shape": (len(indices),), "names": names}
    for group in ("left_leg", "right_leg", "waist", "left_arm", "right_arm"):
        group_indices = [indices.index(i) for i in robot_model.get_joint_group_indices(group)]
        modality["state"][group] = {"start": min(group_indices), "end": max(group_indices) + 1}
    for side in SIDES:
        offset = SIDES.index(side) * 6
        modality["state"][f"{side}_hand"] = {
            "start": offset, "end": offset + 6, "original_key": "observation.hand_joint_position",
        }
        features[f"teleop.{side}_hand_joints"] = {
            "dtype": "float32", "shape": (6,), "names": MOTOR_NAMES[offset:offset + 6],
        }
        modality["action"][f"{side}_hand_joints"]["end"] = 6
    definitions = {
        "observation.hand_joint_position": ("float32", MOTOR_NAMES),
        "observation.hand_joint_velocity": ("float32", MOTOR_NAMES),
        "observation.hand_binary_state": ("int32", list(SIDES)),
        "observation.hand_valid": ("int32", list(SIDES)),
        "observation.hand_received_timestamp_ns": ("int64", list(SIDES)),
        "observation.hand_age_s": ("float64", list(SIDES)),
        "action.hand_joint_position": ("float32", MOTOR_NAMES),
        "action.hand_binary": ("int32", list(SIDES)),
        "action.hand_valid": ("int32", ["command_valid"]),
        "action.hand_enabled": ("int32", ["command_sent_this_update"]),
        "action.hand_timestamp_ns": ("int64", ["last_successful_dds_write"]),
        "teleop.hand_pico_timestamp_ns": ("int64", ["last_command_pico_input"]),
        "observation.capture_timestamp_ns": ("int64", ["workstation_sample_time"]),
        "observation.camera_timestamp": ("float64", ["robot_camera_time_s"]),
    }
    for prefix in ('observation.previous_hand_action', 'action.aligned_hand'):
        definitions[prefix] = ('int32', list(SIDES))
        definitions[f'{prefix}_joint_position'] = ('float32', MOTOR_NAMES)
        for suffix in ('monotonic_ns', 'timestamp_ns', 'sequence', 'session_id'):
            definitions[f'{prefix}_{suffix}'] = ('int64', [suffix])
        definitions[f'{prefix}_valid'] = ('int32', ['valid'])
    for key, (dtype, names_) in definitions.items():
        features[key] = {"dtype": dtype, "shape": (len(names_),), "names": names_}
    for group, key in (("state", "observation.hand_binary_state"),
                       ("action", "action.hand_binary"),
                       ("action", "action.hand_joint_position")):
        modality[group][key.split(".")[-1]] = {
            "start": 0, "end": features[key]["shape"][0], "original_key": key,
        }


def brainco_metadata(pose_config=""):
    poses = load_poses(pose_config)
    return {
        "schema": "brainco_binary_v3", "motor_order": MOTOR_NAMES,
        "joint_order_per_hand": list(JOINT_ORDER), "pose_config": poses,
        "joint_action": "actual successfully written normalized DDS targets; never inferred from binary",
        "position_unit": "normalized service q: 0=open, 1=closed; not radians",
        "velocity_unit": "raw DDS MotorState.dq as reported by hand service",
        "binary_action": "last successfully written command; 0=open, 1=closed",
        "aligned_commands": "command_history selected at body snapshot and action queue times; monotonic clocks require SAME workstation; not physical execution acknowledgement",
        "binary_state": "derived per hand: measured q nearest to configured open/closed pose (ties closed); not a hardware switch or grasp success",
        "feedback_timestamps": "workstation DDS receipt time, Unix nanoseconds",
        "command_timestamp": "workstation time after both DDS writes succeed, Unix nanoseconds",
        "body_state": "29 body joints in FK model body order, radians; no Dex3 finger placeholders",
        "wrist_fk": "stock G1 wrist frames; no BrainCo fingertip kinematics",
    }


class BraincoRecordingState:
    def __init__(self, max_age_s=0.25, pose_config=""):
        self.poses = load_poses(pose_config)
        self.message = None
        self.max_age_s = max_age_s
        self.error = "waiting for brainco_state from PICO manager"

    def update(self, raw):
        try:
            message = json.loads(raw[len(b"brainco_state"):])
            if not isinstance(message, dict):
                raise ValueError("telemetry must be an object")
            if message.get("version") != 3:
                raise ValueError("unsupported BrainCo telemetry version")
            if validate_poses(message.get('pose_config')) != self.poses:
                raise ValueError('control/recording BrainCo pose configuration differs; restart both processes')
            events = message.get('command_history')
            if not isinstance(events, list) or len(events) > 128:
                raise ValueError('missing/oversized command history; restart PICO manager')
            coverage = message['monotonic_ns']
            session = message['command_session_id']
            if any(type(v) is not int or v <= 0 for v in (coverage, session)):
                raise ValueError('invalid command source clock/session')
            previous_seq = previous_stamp = 0
            for event in events:
                seq, stamp, wall = event['sequence'], event['monotonic_ns'], event['timestamp_ns']
                if any(type(v) is not int or v <= 0 for v in (seq, stamp, wall)) or seq <= previous_seq or not previous_stamp < stamp <= coverage:
                    raise ValueError('non-monotonic command history')
                if len(event['binary']) != 2 or any(v not in (0, 1) for v in event['binary']):
                    raise ValueError('invalid historical binary command')
                self.command_positions(event.get("q_command"), event["binary"])
                previous_seq, previous_stamp = seq, stamp
            self.message = message
            # Validate payload immediately, not only after the recording starts.
            self.frame()
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            self.message = None
            self.error = f"invalid BrainCo telemetry: {error}"

    def command_positions(self, command, binary):
        if not isinstance(command, dict) or set(command) != set(SIDES):
            raise ValueError("missing actual six-motor command")
        expected = pose_targets(self.poses, binary)
        values = []
        for side in SIDES:
            q = np.asarray(command[side], dtype=np.float64)
            if q.shape != (6,) or not np.isfinite(q).all() or not ((0 <= q) & (q <= 1)).all():
                raise ValueError("invalid actual six-motor command")
            if not np.allclose(q, expected[side], rtol=0, atol=1e-7):
                raise ValueError("actual command differs from configured pose")
            values.extend(q)
        return np.asarray(values, dtype=np.float32)

    def aligned_commands(self, state_ns, action_ns):
        """Latest successfully queued command at each body time, on the SAME host.

        Telemetry must cover the queried instant. Missing history is invalid,
        never replaced by a later command or inferred from measured hand q.
        """
        m = self.message or {}
        result = {}
        for prefix, cutoff in (('observation.previous_hand_action', state_ns),
                               ('action.aligned_hand', action_ns)):
            candidates = [e for e in m.get('command_history', []) if e['monotonic_ns'] <= cutoff]
            event = candidates[-1] if candidates else {}
            valid = bool(event) and m.get('monotonic_ns', 0) >= cutoff and not m.get('dry_run', True) and bool(m.get('command_valid'))
            result[prefix] = np.array(event.get('binary', [0, 0]), dtype=np.int32)
            result[f'{prefix}_joint_position'] = (
                self.command_positions(event.get('q_command'), event['binary'])
                if event else np.zeros(12, dtype=np.float32))
            for suffix in ('monotonic_ns', 'timestamp_ns', 'sequence'):
                result[f'{prefix}_{suffix}'] = np.array([event.get(suffix, 0)], dtype=np.int64)
            result[f'{prefix}_session_id'] = np.array([m.get('command_session_id', 0)], dtype=np.int64)
            result[f'{prefix}_valid'] = np.array([valid], dtype=np.int32)
        return result

    def frame(self, now_ns=None):
        now_ns = time.time_ns() if now_ns is None else now_ns
        m = self.message or {}
        q, dq, valid, stamps, ages = [], [], [], [], []
        telemetry_age = (now_ns - int(m.get("timestamp_ns", 0))) / 1e9
        for side in SIDES:
            state = m.get("feedback", {}).get(side, {})
            pos = np.asarray(state.get("q", [0.] * 6), dtype=np.float32)
            vel = np.asarray(state.get("dq", [0.] * 6), dtype=np.float32)
            if pos.shape != (6,) or vel.shape != (6,):
                raise ValueError("each BrainCo hand must contain six q/dq values")
            numeric_ok = np.isfinite(pos).all() and np.isfinite(vel).all() and ((0 <= pos) & (pos <= 1)).all()
            stamp = int(state.get("received_timestamp_ns", 0))
            age = (now_ns - stamp) / 1e9 if stamp else -1.
            ok = numeric_ok and bool(state.get("valid")) and 0 <= age <= self.max_age_s
            ok = ok and 0 <= telemetry_age <= self.max_age_s
            q.extend(pos); dq.extend(vel); valid.append(int(ok)); stamps.append(stamp); ages.append(age)
        command = m.get("binary_command")
        command_valid = command is not None and bool(m.get("command_valid")) and not m.get("dry_run", True)
        command = [0, 0] if command is None else command
        if len(command) != 2 or any(v not in (0, 1) for v in command):
            raise ValueError("binary command must be [left, right] with values 0 or 1")
        command_valid = command_valid and 0 <= telemetry_age <= self.max_age_s
        binary = np.asarray(command, dtype=np.int32)
        actual_command = (self.command_positions(m.get("q_command"), command)
                          if m.get("binary_command") is not None else np.zeros(12, dtype=np.float32))
        positions = np.asarray(q, dtype=np.float32)
        measured = positions.reshape(2, 6)
        opened = np.asarray([self.poses[s]["open"] for s in SIDES])
        closed = np.asarray([self.poses[s]["closed"] for s in SIDES])
        binary_state = (((measured - closed) ** 2).sum(axis=1)
                        <= ((measured - opened) ** 2).sum(axis=1)).astype(np.int32)
        result = {
            "observation.hand_joint_position": positions,
            "observation.hand_joint_velocity": np.asarray(dq, dtype=np.float32),
            "observation.hand_binary_state": binary_state,
            "observation.hand_valid": np.asarray(valid, dtype=np.int32),
            "observation.hand_received_timestamp_ns": np.asarray(stamps, dtype=np.int64),
            "observation.hand_age_s": np.asarray(ages, dtype=np.float64),
            "action.hand_binary": binary,
            "action.hand_joint_position": actual_command,
            "action.hand_valid": np.array([command_valid], dtype=np.int32),
            "action.hand_enabled": np.array([bool(m.get("command_enabled")) and command_valid], dtype=np.int32),
            "action.hand_timestamp_ns": np.array([m.get("command_timestamp_ns", 0)], dtype=np.int64),
            "teleop.hand_pico_timestamp_ns": np.array([m.get("pico_timestamp_ns", 0)], dtype=np.int64),
            "observation.capture_timestamp_ns": np.array([now_ns], dtype=np.int64),
        }
        for i, side in enumerate(SIDES):
            result[f"teleop.{side}_hand_joints"] = result["action.hand_joint_position"][i * 6:(i + 1) * 6].copy()
        ready = all(valid) and command_valid
        reasons = []
        if not ready:
            if self.message is None:
                reasons.append(self.error if self.error.startswith("invalid BrainCo telemetry:")
                               else "no valid brainco_state received")
            else:
                if not 0 <= telemetry_age <= self.max_age_s:
                    reasons.append(f"BrainCo manager telemetry age={telemetry_age * 1000:.0f}ms (limit {self.max_age_s * 1000:.0f}ms)")
                for i, side in enumerate(SIDES):
                    if not valid[i]:
                        state = m.get("feedback", {}).get(side, {})
                        age_text = f"{ages[i] * 1000:.0f}ms" if stamps[i] else "missing"
                        reasons.append(f"BrainCo {side}: feedback age={age_text}, service_valid={bool(state.get('valid'))}, limit={self.max_age_s * 1000:.0f}ms; q/dq must be finite, q in [0,1]")
                if m.get("dry_run", True):
                    reasons.append("BrainCo dry-run: no real hand command")
                if m.get("binary_command") is None or not m.get("command_valid"):
                    reasons.append("BrainCo no successful hand command yet; check policy/PICO and release both hand controls to arm")
        self.error = "; ".join(reasons)
        return result, ready
