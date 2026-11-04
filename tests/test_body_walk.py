import importlib.util
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

source = Path.home()/"repo/GR00T-WholeBodyControl/gear_sonic/utils/teleop/body_walk.py"
spec = importlib.util.spec_from_file_location("body_walk", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def sample(t, pelvis=(0., 1., 0.), feet_shift=0., yaw=0.):
    raw = np.zeros((24, 7));raw[:, 6] = 1.
    raw[0, :3] = pelvis
    raw[[7, 8], 2] = feet_shift
    head = np.zeros(7);head[3:] = Rotation.from_euler("y", yaw).as_quat()
    return dict(body_poses_np=raw, headset_pose=head,
                timestamp_ns=int((10.+t)*1e9), body_timestamp_ns=int((10.+t)*1e9),
                timestamp_monotonic=100.+t)


class BodyWalkTests(unittest.TestCase):
    def test_forward_left_and_calibrated_heading(self):
        for yaw, velocity in [(0., [0., 0., -.3]), (np.pi/2, [-.3, 0., 0.]),
                              (0., [-.3, 0., 0.])]:
            walk = module.BodyWalk();self.assertTrue(walk.reset(sample(0., yaw=yaw)))
            for t in np.arange(0., 1.01, .05):
                p = np.array([0., 1., 0.])+np.array(velocity)*t
                result = walk.update(sample(t, p, feet_shift=.15*t, yaw=yaw), now=100.+t)
            np.testing.assert_allclose(result, [.3, 0.] if yaw or velocity[2] else [0., .3], atol=.01)

    def test_noise_in_place_steps_and_lean_do_not_start_walking(self):
        for kind in ["noise", "in_place", "lean"]:
            walk = module.BodyWalk();walk.reset(sample(0.))
            for t in np.arange(0., 2., .05):
                x = .001*np.sin(20*t) if kind == "noise" else (-.2*t if kind == "lean" else 0.)
                s = sample(t, pelvis=(x, 1., 0.), feet_shift=.05*np.sin(8*t) if kind == "in_place" else 0.)
                np.testing.assert_array_equal(walk.update(s, now=100.+t), [0., 0.])

    def test_speed_is_bounded_and_stale_frozen_or_jumped_tracking_stops(self):
        walk = module.BodyWalk();walk.reset(sample(0.))
        for t in np.arange(0., 1., .05):
            s = sample(t, pelvis=(0., 1., -t), feet_shift=.5*t)
            result = walk.update(s, now=100.+t)
        self.assertGreater(result[0], .45);self.assertLessEqual(np.linalg.norm(result), .5)
        # Headset timestamp updates, but actual body timestamp remains frozen.
        s["timestamp_monotonic"] = 101.3
        np.testing.assert_array_equal(walk.update(s, now=101.3), [0., 0.])
        s = sample(1.4, pelvis=(0., 1., -5.), feet_shift=1.)
        np.testing.assert_array_equal(walk.update(s, now=102.), [0., 0.])
        self.assertEqual(walk.reason, "stale tracking")
        walk.reset(sample(0.));walk.update(sample(0.), now=100.)
        np.testing.assert_array_equal(walk.update(sample(.05, pelvis=(0., 1., -1.)), now=100.05), [0., 0.])
        self.assertEqual(walk.reason, "tracking discontinuity")

    def test_missing_or_invalid_headset_cannot_calibrate(self):
        walk = module.BodyWalk();s=sample(0.);s["headset_pose"][:]=0.
        self.assertFalse(walk.reset(s))
        np.testing.assert_array_equal(walk.update(s, now=100.), [0., 0.])
