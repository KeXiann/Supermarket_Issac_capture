"""Pure-Python safety and control math used by the live Isaac/DDS adapter."""
from dataclasses import dataclass, replace

import numpy as np


MOTOR_COUNT = 29


class CommandRejected(ValueError):
    """A DDS command failed protocol or safety validation."""


@dataclass(frozen=True)
class MotorCommand:
    q: np.ndarray
    dq: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    tau: np.ndarray


def vector(values, name):
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (MOTOR_COUNT,):
        raise CommandRejected(f"{name}: expected {MOTOR_COUNT} motors, got {result.shape}")
    if not np.isfinite(result).all():
        raise CommandRejected(f"{name}: non-finite value")
    return result


def decode_lowcmd(msg, position_low, position_high, velocity_limits,
                  position_policy="strict"):
    motors = msg.motor_cmd
    if len(motors) < MOTOR_COUNT:
        raise CommandRejected(f"motor_cmd has {len(motors)} entries; expected at least {MOTOR_COUNT}")
    q = vector([m.q for m in motors[:MOTOR_COUNT]], "q")
    dq = vector([m.dq for m in motors[:MOTOR_COUNT]], "dq")
    kp = vector([m.kp for m in motors[:MOTOR_COUNT]], "kp")
    kd = vector([m.kd for m in motors[:MOTOR_COUNT]], "kd")
    tau = vector([m.tau for m in motors[:MOTOR_COUNT]], "tau")
    try:
        modes = [int(m.mode) for m in motors[:MOTOR_COUNT]]
    except AttributeError as exc:
        raise CommandRejected("LowCmd motor mode field is missing") from exc
    if any(mode != 1 for mode in modes):
        raise CommandRejected("LowCmd contains disabled/nonstandard motor mode")
    if position_policy not in ("strict", "sonic-pd"):
        raise ValueError(f"unknown position policy: {position_policy}")
    # SONIC sends a PD spring equilibrium, not a kinematic position command.
    # The upstream simulator computes PD effort from this raw target and clips
    # effort, while its joint constraints limit actual motion. Rejecting the
    # whole 29-motor packet at the *physical* position limits drops normal policy
    # actions and eventually latches our watchdog. Keep a finite protocol bound
    # for corrupt packets, and enforce actual-position/effort limits separately.
    if position_policy == "sonic-pd":
        bad_position = np.flatnonzero(np.abs(q) > 4.0 * np.pi)
        if bad_position.size:
            raise CommandRejected("LowCmd PD equilibrium exceeds 4*pi rad protocol bound")
    else:
        bad_position = np.flatnonzero((q < position_low) | (q > position_high))
    if position_policy == "strict" and bad_position.size:
        index = int(bad_position[0])
        raise CommandRejected(
            f"LowCmd motor {index} target {q[index]:.4f} rad outside "
            f"[{position_low[index]:.4f}, {position_high[index]:.4f}] rad"
        )
    if np.any(np.abs(dq) > velocity_limits):
        raise CommandRejected("LowCmd velocity target exceeds configured joint limits")
    if np.any(kp < 0) or np.any(kp > 2000) or np.any(kd < 0) or np.any(kd > 200):
        raise CommandRejected("LowCmd gains are outside accepted finite range")
    if np.any(np.abs(tau) > 1000):
        raise CommandRejected("LowCmd feed-forward torque is outside accepted range")
    return MotorCommand(q, dq, kp, kd, tau)


def guard_joint_effort(torque, current_q, current_dq, position_low,
                       position_high, velocity_limits, tolerance=0.002):
    """Suppress effort that pushes an actual joint farther beyond its limits."""
    torque = vector(torque, "guard torque").copy()
    q = vector(current_q, "guard q")
    dq = vector(current_dq, "guard dq")
    low = vector(position_low, "position low")
    high = vector(position_high, "position high")
    velocity = vector(velocity_limits, "velocity limits")
    outward = (((q <= low + tolerance) & (torque < 0))
               | ((q >= high - tolerance) & (torque > 0)))
    accelerating_overspeed = (np.abs(dq) >= velocity) & (torque * dq > 0)
    torque[outward | accelerating_overspeed] = 0.0
    return torque


def first_command_error(command, current_q):
    q = vector(current_q, "current q")
    return float(np.max(np.abs(command.q - q)))


def guard_implicit_command(command, current_q, current_dq, position_low,
                           position_high, velocity_limits, tolerance=.002):
    """Keep PhysX PD from driving outward at a measured limit; allow braking."""
    q = vector(current_q, "current q")
    dq = vector(current_dq, "current dq")
    low = vector(position_low, "position low")
    high = vector(position_high, "position high")
    velocity = vector(velocity_limits, "velocity limits")
    at_low, at_high = q <= low + tolerance, q >= high - tolerance
    outward = ((at_low & (command.q < q)) | (at_high & (command.q > q))
               | ((np.abs(dq) >= velocity) & ((command.q - q) * dq > 0)))
    kp = vector(command.kp, "drive kp").copy()
    kp[outward] = 0.
    target_dq = vector(command.dq, "drive target dq").copy()
    target_dq[(at_low & (target_dq < 0)) | (at_high & (target_dq > 0))] = 0.
    feedforward = guard_joint_effort(command.tau, q, dq, low, high, velocity, tolerance)
    return replace(command, kp=kp, dq=target_dq, tau=feedforward)


def implicit_feedback_budget(feedforward, effort_limits):
    """Reserve force for external feed-forward within the total motor limit."""
    effort = vector(effort_limits, "effort limits")
    ff = vector(feedforward, "feed-forward")
    if np.any(effort <= 0) or np.any(np.abs(ff) > effort):
        raise CommandRejected("invalid implicit effort budget")
    return effort - np.abs(ff)


def slew_pd_command(command, previous_target, velocity_limits, dt, max_step):
    """Bound applied equilibrium motion without dropping normal policy packets."""
    previous = vector(previous_target, "previous PD target")
    limits = vector(velocity_limits, "target velocity limits")
    if dt <= 0 or max_step <= 0:
        raise ValueError("dt and max_step must be positive")
    step = np.minimum(limits * dt, max_step)
    target = previous + np.clip(command.q - previous, -step, step)
    return replace(command, q=target)


def compute_torque(command, current_q, current_dq, kp_scale, kd_scale,
                   torque_scale, effort_limits, gain_kp_limit=450.0, gain_kd_limit=25.0):
    q = vector(current_q, "current q")
    dq = vector(current_dq, "current dq")
    effort = vector(effort_limits, "effort limits")
    if np.any(effort <= 0):
        raise CommandRejected("non-positive effort limit")
    if not (0 < kp_scale <= 1 and 0 < kd_scale <= 1 and 0 < torque_scale <= 1):
        raise CommandRejected("gain/torque scales must be in (0, 1]")
    kp = np.clip(command.kp * kp_scale, 0, gain_kp_limit)
    kd = np.clip(command.kd * kd_scale, 0, gain_kd_limit)
    # Keep the three safety controls independent.  ``kp_scale`` and
    # ``kd_scale`` scale SONIC's feedback gains; ``torque_scale`` only scales
    # its feed-forward torque.  Multiplying the whole expression by all three
    # scales made a launch such as .1/.1/.2 reduce feedback to two percent,
    # which cannot track a pose and makes the result look like a bad mapping.
    torque = (torque_scale * command.tau
              + kp * (command.q - q)
              + kd * (command.dq - dq))
    if not np.isfinite(torque).all():
        raise CommandRejected("computed torque is non-finite")
    return np.clip(torque, -effort, effort)


def hold_torque(target_q, current_q, current_dq, kp, kd, effort_limits):
    target = vector(target_q, "hold target")
    q = vector(current_q, "current q")
    dq = vector(current_dq, "current dq")
    kp = vector(kp, "hold kp")
    kd = vector(kd, "hold kd")
    effort = vector(effort_limits, "effort limits")
    torque = kp * (target - q) - kd * dq
    if not np.isfinite(torque).all():
        raise CommandRejected("computed hold torque is non-finite")
    return np.clip(torque, -effort, effort)


def inverse_quaternion_matrix(quaternion):
    """Single-frame wxyz inverse rotation, matching Isaac's unit-quat formula.

    CPU bridge observations need only four vectors, not a batched Torch/JIT
    graph. The caller retains the measured quaternion; no pose normalization.
    """
    w, x, y, z = quaternion
    a = 2*w*w - 1
    return np.array(((a+2*x*x, 2*x*y+2*w*z, 2*x*z-2*w*y),
                     (2*x*y-2*w*z, a+2*y*y, 2*y*z+2*w*x),
                     (2*x*z+2*w*y, 2*y*z-2*w*x, a+2*z*z)))
