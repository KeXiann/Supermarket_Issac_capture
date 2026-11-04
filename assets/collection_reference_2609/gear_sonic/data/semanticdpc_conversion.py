"""Raw real-recording tensors matching semanticdpc_0826 04eead3 causal.py.

This is a tensor adapter, not a synthetic simulator reset archive.
"""
import json
from pathlib import Path
import numpy as np

CONTRACT_PATH = Path(__file__).with_name('semanticdpc_contract.json')


def load_contract():
    return json.loads(CONTRACT_PATH.read_text())


def quat_features(quaternion):
    q = np.asarray(quaternion, dtype=np.float32)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    w, x, y, z = (float(v) for v in q)
    matrix = np.asarray([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ], dtype=np.float32)
    return matrix[:, :2].reshape(6), matrix.T @ np.array([0, 0, -1], np.float32)


def convert_episode(df, info, max_gap_ms=30.0):
    if not 20 <= max_gap_ms <= 40:
        raise ValueError('max_gap_ms must be in [20,40] for the 50 Hz contract')
    contract = load_contract()
    if float(info['fps']) != contract['control_hz']:
        raise ValueError('Recorded 50 Hz data required; implicit resampling is forbidden')
    if info.get('script_config', {}).get('body_action_alignment_version') != 2:
        raise ValueError('Body protocol v2 required; do not fabricate missing qd/IMU/history')
    n = len(df)
    if n == 0:
        raise ValueError('Empty episode')
    def matrix(key, width, dtype=np.float32):
        if key not in df:
            raise ValueError(f'Missing required column: {key}')
        values = np.asarray(df[key].tolist(), dtype=dtype)
        if values.shape == (n,) and width == 1:
            values = values[:, None]
        if values.shape != (n, width):
            raise ValueError(f'{key}: expected {(n,width)}, got {values.shape}')
        return values
    def scalar(key):
        raw = matrix(key, 1, dtype=None).ravel()
        if raw.dtype.kind not in 'iu':
            raise ValueError(f'Integer dtype required: {key}')
        return raw.astype(np.int64)
    def joints(key):
        names = info['features'].get(key, {}).get('names')
        if not isinstance(names, list) or len(names) != 29 or len(set(names)) != 29 or set(names) != set(contract['canonical_joint_names']):
            raise ValueError(f'Explicit 29-joint names required: {key}')
        return matrix(key, 29)[:, [names.index(name) for name in contract['canonical_joint_names']]]
    q, dq = joints('observation.state'), joints('observation.joint_velocity')
    target, previous = joints('action.wbc'), joints('observation.previous_body_action')
    quat = matrix('observation.root_orientation', 4)
    gyro = matrix('observation.base_angular_velocity', 3)
    accel = matrix('observation.base_acceleration', 3)
    hand_previous = matrix('observation.previous_hand_action', 2)
    hand_target = matrix('action.aligned_hand', 2)
    index = scalar('observation.control_cycle_index')
    state_ns = scalar('observation.body_monotonic_ns')
    action_ns = scalar('action.body_monotonic_ns')
    received_ns = scalar('observation.lowstate_receive_monotonic_ns')
    source_tick = scalar('observation.lowstate_tick')
    previous_index = scalar('observation.previous_body_action_index')
    previous_ns = scalar('observation.previous_body_action_monotonic_ns')
    hand_state_ns = scalar('observation.previous_hand_action_monotonic_ns')
    hand_action_ns = scalar('action.aligned_hand_monotonic_ns')
    session = scalar('observation.previous_hand_action_session_id')
    action_session = scalar('action.aligned_hand_session_id')
    flags = [scalar(k) for k in ('observation.body_valid', 'observation.previous_body_action_valid',
                                  'observation.previous_hand_action_valid', 'action.aligned_hand_valid')]
    if any(not np.isin(flag, [0, 1]).all() for flag in flags):
        raise ValueError('Validity flags must be 0 or 1')
    valid = np.logical_and.reduce([flag == 1 for flag in flags])
    valid &= (matrix('observation.hand_valid', 2, np.int64) == 1).all(axis=1)
    capture_age = matrix('observation.body_capture_age_s', 1).ravel()
    valid &= np.isfinite(capture_age) & (capture_age >= 0) & (capture_age <= .1)
    for values in (q, dq, target, previous, quat, gyro, accel, hand_previous, hand_target):
        valid &= np.isfinite(values).all(axis=1)
    valid &= np.isin(hand_previous, [0, 1]).all(axis=1) & np.isin(hand_target, [0, 1]).all(axis=1)
    valid &= (np.linalg.norm(quat, axis=1) >= .9) & (np.linalg.norm(quat, axis=1) <= 1.1)
    valid &= (index > 0) & (previous_index == index-1) & (previous_ns > 0) & (previous_ns <= state_ns)
    valid &= (received_ns > 0) & (received_ns <= state_ns) & ((state_ns-received_ns) <= 100_000_000)
    valid &= (source_tick >= 0) & (action_ns >= state_ns)
    valid &= (hand_state_ns > 0) & (hand_state_ns <= state_ns) & (hand_action_ns >= hand_state_ns) & (hand_action_ns <= action_ns)
    valid &= (session > 0) & (session == action_session)
    default = np.asarray(contract['default_q'], np.float32)
    scale = np.asarray(contract['action_scale'], np.float32)
    state64 = np.zeros((n,64), np.float32)
    frames = np.zeros((n,95), np.float32)
    actions = np.zeros((n,31), np.float32)
    for i in np.flatnonzero(valid):
        rot6d, gravity = quat_features(quat[i])
        state64[i] = np.concatenate((rot6d, q[i], dq[i]))
        frames[i] = np.concatenate((gyro[i], q[i]-default, dq[i], (previous[i]-default)/scale, gravity, hand_previous[i]))
        actions[i] = np.concatenate(((target[i]-default)/scale, hand_target[i]))
    histories = np.zeros((n,10,95), np.float32)
    masks = np.zeros((n,10), bool)
    reset = np.ones(n, bool)
    segments = np.full(n, -1, np.int64)
    reasons = np.full(n, 'invalid_sample', dtype='<U40')
    window, segment = [], -1
    for i in range(n):
        if not valid[i]:
            window = []
            continue
        reason = 'episode_start' if i == 0 else 'after_invalid'
        contiguous = False
        if i > 0 and valid[i-1]:
            delta = state_ns[i]-state_ns[i-1]
            if index[i] != index[i-1]+1:
                reason = 'cycle_gap_or_reset'
            elif not 10_000_000 <= delta <= max_gap_ms*1e6:
                reason = 'clock_gap_or_reset'
            elif received_ns[i] <= received_ns[i-1] or source_tick[i] <= source_tick[i-1]:
                reason = 'source_repeat_or_reset'
            elif session[i] != session[i-1]:
                reason = 'hand_session_reset'
            elif previous_ns[i] != action_ns[i-1] or not np.allclose(previous[i], target[i-1], rtol=0, atol=1e-6):
                reason = 'previous_action_mismatch'
            else:
                contiguous = True
                reason = 'continuous'
        if not contiguous:
            window = []
            segment += 1
        reset[i] = not contiguous
        reasons[i] = reason
        segments[i] = segment
        window.append(frames[i].copy())
        window = window[-10:]
        count = len(window)
        histories[i, -count:] = np.stack(window)
        histories[i, :-count] = window[0]
        masks[i, -count:] = True
    return {'state64': state64, 'history10': histories, 'history_mask': masks,
            'action31': actions, 'sample_valid': valid, 'reset_before': reset,
            'segment_id': segments, 'boundary_reason': reasons,
            'source_row_index': np.arange(n, dtype=np.int64), 'source_control_cycle_index': index,
            'source_state_monotonic_ns': state_ns, 'source_action_monotonic_ns': action_ns,
            'source_lowstate_tick': source_tick,
            'base_acceleration': np.where(valid[:,None], accel, 0).astype(np.float32)}
