"""One-way startup gate for a real floating-base SONIC simulation.

Only startup physics is withheld. Once released, this gate never supports,
teleports, fixes, or substitutes measurements of the robot.
"""
from pathlib import Path
import math

import numpy as np


class PolicyStartupGate:
    def __init__(self, log_path, nominal_q, *, require_measured_history=False):
        self.path = Path(log_path)
        # A pre-existing marker must never release a new simulation.
        self.offset = self.path.stat().st_size if self.path.exists() else 0
        self.pending = ""
        self.policy_observed = False
        self.released = False
        self.release_error_rad = None
        self.require_measured_history = require_measured_history
        self.history_ready = False
        self.nominal_q = np.asarray(nominal_q, dtype=float)

    def ready(self, command, command_age, watchdog, *, current_q=None, max_initial_error=None):
        if self.released:
            return True
        if self.path.exists():
            size = self.path.stat().st_size
            if size < self.offset:
                self.offset = 0
                self.pending = ""
                self.policy_observed = False
                self.history_ready = False
            with self.path.open(errors="replace") as stream:
                stream.seek(self.offset)
                new = stream.read()
                self.offset = stream.tell()
            content = self.pending + new
            lines = content.split("\n")
            self.pending = lines.pop()
            self.policy_observed |= any("[Mode Filter] Switched to mode" in line for line in lines)
            self.history_ready |= any(line.startswith('[SIM STARTUP] measured history ready: 10 policy cycles; mode g1')
                                      for line in lines)
            if self.require_measured_history:
                self.policy_observed |= self.history_ready
        # INIT/WAIT publish nominal posture packets too. A real encoder marker
        # plus a changed, validated LowCmd proves that policy actuation exists.
        ready = bool(self.policy_observed and command is not None
                     and (self.history_ready or not self.require_measured_history)
                     and command_age <= watchdog
                     and np.max(np.abs(command.q-self.nominal_q)) > 1e-3)
        if ready and current_q is not None:
            error = float(np.max(np.abs(command.q - np.asarray(current_q))))
            if error > max_initial_error:
                raise ValueError(f'SONIC policy takeover target/current error {error:.3f} rad '
                                 f'> {max_initial_error:.3f}; INIT alignment does not validate the policy command')
            self.release_error_rad = error
        self.released = ready
        return self.released


def base_tilt_radians(quaternion_wxyz):
    """Tilt of the pelvis z-axis from world vertical; independent of yaw."""
    q = np.asarray(quaternion_wxyz, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-9:
        raise ValueError("Invalid pelvis quaternion")
    q = q / np.linalg.norm(q)
    return math.acos(float(np.clip(1-2*(q[1]*q[1]+q[2]*q[2]), -1, 1)))
