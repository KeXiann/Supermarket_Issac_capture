"""Versioned state/action recording contract; source frames are never synthesized."""
import numpy as np

BODY_ALIGNMENT_VERSION = 2
BODY_JOINT_NAMES = [
    *[f'{side}_{name}_joint' for side in ('left', 'right')
      for name in ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll')],
    'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
    *[f'{side}_{name}_joint' for side in ('left', 'right')
      for name in ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll', 'wrist_pitch', 'wrist_yaw')],
]
BODY_ALIGNMENT_METADATA = {
    'version': BODY_ALIGNMENT_VERSION, 'pairing': 's_t,a_t',
    'action_source': 'body_action: exact queued MotorCommand.q_target, hardware order, radians',
    'previous_action_source': 'last preceding logger cycle with an exact queued target; invalid if that cycle has no target; not robot execution acknowledgement',
    'cycle_index': 'C++ StateLogger index; resets on deployment restart; repeats/skips allowed',
    'state_timestamp_ns': 'workstation Unix time at state snapshot logging (not sensor exposure time)',
    'monotonic_timestamps': 'workstation steady clock nanoseconds; all producing processes must run on the SAME host',
    'source_clock': 'lowstate_receive_monotonic_ns: workstation DataBuffer receipt; lowstate_tick: raw device tick, not Unix time',
    'base_imu': 'LowState.imu_state, NOT torso IMU; angular velocity rad/s; raw accelerometer m/s^2 in sensor frame (not gravity-compensated world acceleration)',
    'joint_velocity': '29 measured MotorState.dq, hardware order, rad/s; see feature names',
    'source_max_age_s': 0.1,
    'scope': 'body pair only; hand commands selected as-of body times; camera exposure remains independent',
}


class BodyAlignmentError(ValueError):
    """The publisher cannot supply a valid recording snapshot."""


def require_body_alignment_version(message):
    if message.get('body_action_alignment_version') != BODY_ALIGNMENT_VERSION:
        raise BodyAlignmentError('Body recording protocol v2 required. Rebuild/restart C++ deployment and use a NEW dataset directory.')


def aligned_body_frame(message):
    require_body_alignment_version(message)
    try:
        def integer(key, minimum=0):
            value = message[key]
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not minimum <= value <= np.iinfo(np.int64).max:
                raise ValueError(f'invalid {key}')
            return int(value)
        def vector(key, size):
            value = np.asarray(message[key], dtype=np.float64)
            if value.shape != (size,) or not np.isfinite(value).all():
                raise ValueError(f'invalid {key}')
            return value
        index = integer('index')
        action_index = integer('body_action_index')
        wall = integer('state_timestamp_ns', 1)
        state_ns = integer('state_monotonic_ns', 1)
        action_ns = integer('body_action_monotonic_ns', 1)
        receipt = integer('lowstate_receive_monotonic_ns', 1)
        tick = integer('lowstate_tick')
        if index != action_index or action_ns < state_ns or receipt > state_ns:
            raise ValueError('cycle/source timestamp mismatch')
        action = vector('body_action', 29)
        vector('body_q', 29)
        dq = vector('body_dq', 29)
        gyro = vector('base_ang_vel', 3)
        accel = vector('base_accel', 3)
        quat = vector('base_quat', 4)
        if not 0.9 <= np.linalg.norm(quat) <= 1.1:
            raise ValueError('invalid base quaternion norm')
        previous = vector('previous_body_action', 29)
        previous_index = integer('previous_body_action_index', -1)
        previous_ns = integer('previous_body_action_monotonic_ns')
        valid = message['previous_body_action_valid']
        if not isinstance(valid, (bool, np.bool_)):
            raise ValueError('previous action validity must be bool')
        if valid and (previous_index != index - 1 or previous_index < 0 or not 0 < previous_ns <= state_ns):
            raise ValueError('previous action is not causal/immediately preceding')
        if not valid and (previous_index != -1 or previous_ns != 0):
            raise ValueError('invalid previous action must have sentinel index/time')
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise BodyAlignmentError(f'Invalid body recording snapshot: {exc}') from exc
    scalar = lambda x, dtype=np.int64: np.array([x], dtype=dtype)
    age = (state_ns - receipt) / 1e9
    return action, {
        'observation.control_cycle_index': scalar(index),
        'observation.body_timestamp_ns': scalar(wall),
        'observation.body_monotonic_ns': scalar(state_ns),
        'action.body_monotonic_ns': scalar(action_ns),
        'observation.joint_velocity': dq,
        'observation.base_angular_velocity': gyro,
        'observation.base_acceleration': accel,
        'observation.lowstate_receive_monotonic_ns': scalar(receipt),
        'observation.lowstate_tick': scalar(tick),
        'observation.body_source_age_s': scalar(age, np.float64),
        'observation.body_valid': scalar(age <= 0.1, np.int32),
        'observation.previous_body_action': previous,
        'observation.previous_body_action_index': scalar(previous_index),
        'observation.previous_body_action_monotonic_ns': scalar(previous_ns),
        'observation.previous_body_action_valid': scalar(valid, np.int32),
    }


def configure_body_alignment_schema(features):
    for key in ('observation.control_cycle_index', 'observation.body_timestamp_ns',
                'observation.body_monotonic_ns', 'action.body_monotonic_ns',
                'observation.lowstate_receive_monotonic_ns', 'observation.lowstate_tick',
                'observation.previous_body_action_index', 'observation.previous_body_action_monotonic_ns'):
        features[key] = {'dtype': 'int64', 'shape': (1,), 'names': [key.split('.')[-1]]}
    for key in ('observation.body_source_age_s', 'observation.body_capture_age_s'):
        features[key] = {'dtype': 'float64', 'shape': (1,), 'names': [key.split('.')[-1]]}
    for key in ('observation.body_valid', 'observation.previous_body_action_valid'):
        features[key] = {'dtype': 'int32', 'shape': (1,), 'names': [key.split('.')[-1]]}
    for key in ('observation.joint_velocity', 'observation.previous_body_action'):
        features[key] = {'dtype': 'float64', 'shape': (29,), 'names': BODY_JOINT_NAMES.copy()}
    for key in ('observation.base_angular_velocity', 'observation.base_acceleration'):
        features[key] = {'dtype': 'float64', 'shape': (3,), 'names': ['x', 'y', 'z']}
