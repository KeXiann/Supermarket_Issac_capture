import math
import unittest
from simulation_timing import StartupPostureReadiness

class StartupPostureTests(unittest.TestCase):
    def test_fast_physics_does_not_qualify_tilted_or_moving_robot(self):
        check = StartupPostureReadiness()
        for t in range(10):
            self.assertFalse(check.update(t, math.radians(33.7), [0,0,0], [0,0,0]))
        self.assertFalse(check.update(10, 0., [0,0,0], [0,0,0]))
        self.assertTrue(check.update(12, 0., [0,0,0], [0,0,0]))
        self.assertFalse(check.update(13, 0., [.3,0,0], [0,0,0]))
        self.assertFalse(check.update(14, 0., [0,0,0], [0,0,1.]))
        self.assertFalse(check.update(15, float('nan'), [0,0,0], [0,0,0]))
        self.assertFalse(check.update(16, 0., [0,0,0], [0,0,0]))
        self.assertTrue(check.update(18, 0., [0,0,0], [0,0,0]))
