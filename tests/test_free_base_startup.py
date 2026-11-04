import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from bridge_core import MotorCommand
from free_base_startup import PolicyStartupGate, base_tilt_radians


class FloatingBaseTests(unittest.TestCase):
    def test_export_rejects_startup_reference_rows(self):
        from export_capture_dataset import validate_startup_window
        config = {'simulation_startup_reference': True, 'simulation_startup_reference_cycles': 150}
        with self.assertRaisesRegex(ValueError, 'startup reference transition'):
            validate_startup_window(config, [{'index': 149}, {'index': 150}])
        validate_startup_window(config, [{'index': 150}, {'index': 151}])
        validate_startup_window({}, [{'index': 0}])

    def test_simulation_warmup_keeps_original_threshold(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / 'sonic.log'
            gate = PolicyStartupGate(log, np.zeros(29), require_measured_history=True)
            cmd = MotorCommand(np.ones(29)*1.25, *[np.ones(29) for _ in range(4)])
            log.write_text("[Mode Filter] Switched to mode 'g1'\n")
            self.assertFalse(gate.ready(cmd, .01, .3, current_q=np.zeros(29), max_initial_error=.75))
            with log.open('a') as stream:
                stream.write('[SIM STARTUP] measured history ready: 10 policy cycles; mode g1\n')
            with self.assertRaisesRegex(ValueError, 'policy takeover'):
                gate.ready(cmd, .01, .3, current_q=np.zeros(29), max_initial_error=.75)
            cmd.q[:] = .35
            self.assertTrue(gate.ready(cmd, .01, .3, current_q=np.zeros(29), max_initial_error=.75))

    def test_aligned_init_does_not_authorize_misaligned_policy_takeover(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / 'sonic.log'
            gate = PolicyStartupGate(log, np.zeros(29))
            aligned = MotorCommand(np.zeros(29), *[np.ones(29) for _ in range(4)])
            self.assertFalse(gate.ready(aligned, .01, .3, current_q=np.zeros(29), max_initial_error=.75))
            log.write_text("[Mode Filter] Switched to mode 'g1'\n")
            jumped = MotorCommand(np.ones(29) * 1.245, *[np.ones(29) for _ in range(4)])
            with self.assertRaisesRegex(ValueError, 'policy takeover'):
                gate.ready(jumped, .01, .3, current_q=np.zeros(29), max_initial_error=.75)
            self.assertFalse(gate.released)
            aligned.q[:] = .1
            self.assertTrue(gate.ready(aligned, .01, .3, current_q=np.zeros(29), max_initial_error=.75))

    def test_startup_needs_fresh_policy_marker_and_non_init_command(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder)/"sonic.log"
            log.write_text("[Mode Filter] Switched to mode 'g1'\n")
            gate = PolicyStartupGate(log, np.zeros(29))
            cmd = MotorCommand(np.ones(29)*.1, *[np.ones(29) for _ in range(4)])
            self.assertFalse(gate.ready(cmd, .01, .3))
            with log.open("a") as stream:
                stream.write("[Mode Filter] Switched to mode 'g1'\n")
            nominal = MotorCommand(np.zeros(29), *[np.ones(29) for _ in range(4)])
            self.assertFalse(gate.ready(nominal, .01, .3))
            self.assertFalse(gate.ready(cmd, .4, .3))
            self.assertTrue(gate.ready(cmd, .01, .3))
            # A released plant never re-enters startup support on timeout.
            self.assertTrue(gate.ready(None, math.inf, .3))

    def test_truncated_log_does_not_reuse_prior_marker(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder)/"sonic.log"
            log.write_text("[Mode Filter] Switched to mode 'g1'\n"+" "*200)
            gate = PolicyStartupGate(log, np.zeros(29))
            log.write_text("new initialization\n")
            cmd = MotorCommand(np.ones(29)*.1, *[np.ones(29) for _ in range(4)])
            self.assertFalse(gate.ready(cmd, .01, .3))

    def test_tilt_ignores_yaw_and_detects_pitch_and_roll(self):
        self.assertAlmostEqual(base_tilt_radians([0, 0, 0, 1]), 0.)
        for q in ([math.cos(.3), math.sin(.3), 0, 0],
                  [math.cos(.3), 0, math.sin(.3), 0]):
            self.assertAlmostEqual(base_tilt_radians(q), .6)
        with self.assertRaises(ValueError):
            base_tilt_radians([0, 0, 0, 0])
