from project_paths import SONIC_ROOT
"""Verify analog grasp mapping and signed feedback through SONIC's slew limit."""
import sys
import unittest
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET

sys.path.insert(0, str(SONIC_ROOT))
from gear_sonic.utils.teleop.revo2_hand_control import (
    controller_hand_command, dex3_to_revo2, revo2_to_dex3, revo2_full_target,
    DEX3_THUMB_OPPOSITION, REVO2_CLOSED, REVO2_OPEN, THUMB_OPPOSITION,
)


class Revo2HandTests(unittest.TestCase):
    def test_thumb_default_is_opposed_and_within_pinned_limits(self):
        from reference_robot import SNAPSHOT
        for side in ('left','right'):
            tree=ET.parse(SNAPSHOT/f'assets/robot/urdf/revo2_{side}_clean.urdf')
            limit=tree.find(f"joint[@name='{side}_thumb_metacarpal_joint']/limit")
            angle = THUMB_OPPOSITION[side == 'left']
            self.assertGreater(angle, np.deg2rad(86))
            self.assertLessEqual(angle, float(limit.get('upper')))

    def test_main_trigger_closes_all_five_fingers(self):
        for side in (True, False):
            for grip in (0., .5, 1.):
                q = dex3_to_revo2(controller_hand_command(1., grip, side), side)
                np.testing.assert_allclose(q, [REVO2_OPEN[side][0], *REVO2_CLOSED[1:]])

    def test_thumb_opposition_is_separate_from_thumb_flexion(self):
        for trigger in (0., .35, 1.):
            command = controller_hand_command(trigger, 1.)
            q = dex3_to_revo2(command)
            self.assertAlmostEqual(float(command[0]), DEX3_THUMB_OPPOSITION, places=6)
            self.assertAlmostEqual(float(q[0]), REVO2_OPEN[True][0], places=6)
            self.assertEqual(bool(q[1] > 0.), trigger > .04)

    def test_side_grip_alone_leaves_hand_open_and_cannot_override_trigger(self):
        for side in (True, False):
            for trigger in (0., .02, .2, .55, 1.):
                expected = controller_hand_command(trigger, 0., side)
                for grip in np.linspace(0., 1., 11):
                    np.testing.assert_array_equal(controller_hand_command(trigger, grip, side), expected)
            np.testing.assert_allclose(dex3_to_revo2(controller_hand_command(0., 1., side), side), REVO2_OPEN[side])

    def test_analog_power_grasp_and_both_mirrors(self):
        for side in (True, False):
            closure = []
            for trigger in np.linspace(0., 1., 21):
                cmd = controller_hand_command(trigger, 1., side)
                physical = dex3_to_revo2(cmd, side)
                closure.append(physical)
                np.testing.assert_allclose(revo2_to_dex3(physical, side), cmd, atol=1e-7)
            self.assertTrue((np.diff(closure, axis=0) >= -1e-8).all())
            np.testing.assert_allclose(closure[0], REVO2_OPEN[side])
            # Trigger closure keeps the thumb opposed.
            np.testing.assert_allclose(closure[-1], [REVO2_OPEN[side][0], *REVO2_CLOSED[1:]])

    def test_sonic_feedback_limiter_reaches_closure_and_opens(self):
        # Mirror exactly Dex3Hands::writeOnce current+clip(desired-current,.25).
        # Unsigned/Revo2-index feedback would stall or reverse left-hand motion.
        for side in (True, False):
            physical = REVO2_OPEN[side].copy()
            for trigger, grip in ((1., 0.), (.5, 1.), (0., 1.), (0., 0.)):
                desired = controller_hand_command(trigger, grip, side)
                for _ in range(40):
                    measured = revo2_to_dex3(physical, side)
                    wire = measured + np.clip(desired - measured, -.25, .25)
                    target = dex3_to_revo2(wire, side)
                    physical += .25 * (target - physical)
                np.testing.assert_allclose(revo2_to_dex3(physical, side), desired, atol=.002)

    def test_opposed_rest_and_velocity_use_consistent_linear_maps(self):
        for side in (True, False):
            rest = controller_hand_command(0., 0., side)
            np.testing.assert_allclose(dex3_to_revo2(rest, side), REVO2_OPEN[side])
            np.testing.assert_allclose(revo2_to_dex3(REVO2_OPEN[side], side), rest)
            np.testing.assert_allclose(revo2_to_dex3(np.zeros(6), side, velocity=True), 0.)
            q = REVO2_OPEN[side] * .4 + REVO2_CLOSED * .6
            velocity = np.array([-.1, .2, .3, .4, .5, .6])
            epsilon = 1e-5
            finite_difference = (revo2_to_dex3(q+epsilon*velocity, side)-revo2_to_dex3(q, side))/epsilon
            np.testing.assert_allclose(revo2_to_dex3(velocity, side, velocity=True), finite_difference, atol=1e-8)

    def test_blocked_proximal_does_not_force_full_distal_curl(self):
        measured = np.array([.5, .6, .4, .45, .5, .55])
        target = revo2_full_target(REVO2_CLOSED, measured)
        np.testing.assert_allclose(target[:6], REVO2_CLOSED)
        np.testing.assert_allclose(target[6:], [.6, .462, .51975, .5775, .63525])
        np.testing.assert_allclose(revo2_full_target(REVO2_CLOSED, np.zeros(6))[6:], 0.)

    def test_resting_input_deadzone(self):
        np.testing.assert_array_equal(controller_hand_command(.02, .03), [.5, 0., 0., 0., 0., 0., 0.])


if __name__ == "__main__":
    unittest.main()
