import unittest

import numpy as np

from bridge_core import (CommandRejected, MotorCommand, compute_torque,
                         decode_lowcmd, first_command_error, guard_joint_effort,
                         hold_torque, slew_pd_command)
from bridge_core import guard_implicit_command, implicit_feedback_budget


class Motor:
    def __init__(self, q=0.0, dq=0.0, kp=2.0, kd=0.5, tau=0.0, mode=1):
        self.q, self.dq, self.kp, self.kd, self.tau, self.mode = q, dq, kp, kd, tau, mode


class Msg:
    def __init__(self, motors):
        self.motor_cmd = motors


def good_msg(**kwargs):
    return Msg([Motor(**kwargs) for _ in range(29)])


class BridgeCoreTests(unittest.TestCase):
    def setUp(self):
        self.low = np.full(29, -2.0)
        self.high = np.full(29, 2.0)
        self.vel = np.full(29, 5.0)

    def test_decode_and_mode(self):
        cmd = decode_lowcmd(good_msg(), self.low, self.high, self.vel)
        self.assertIsInstance(cmd, MotorCommand)
        self.assertEqual(cmd.q.shape, (29,))
        self.assertAlmostEqual(first_command_error(cmd, np.zeros(29)), 0.0)

    def test_rejects_disabled_motor(self):
        with self.assertRaisesRegex(CommandRejected, "mode"):
            decode_lowcmd(good_msg(mode=0), self.low, self.high, self.vel)

    def test_rejects_limits_and_nonfinite(self):
        with self.assertRaises(CommandRejected):
            decode_lowcmd(good_msg(q=2.1), self.low, self.high, self.vel)
        with self.assertRaises(CommandRejected):
            decode_lowcmd(good_msg(dq=5.1), self.low, self.high, self.vel)
        with self.assertRaises(CommandRejected):
            decode_lowcmd(good_msg(q=float("nan")), self.low, self.high, self.vel)

    def test_feedback_and_feedforward_scales_are_independent(self):
        cmd = decode_lowcmd(good_msg(q=1.0, kp=4.0, kd=2.0, tau=1.0), self.low, self.high, self.vel)
        result = compute_torque(cmd, np.zeros(29), np.zeros(29), 0.5, 0.5, 0.5,
                                np.full(29, 4.0))
        # 0.5*1 feed-forward + (4*.5)*1 feedback = 2.5 Nm.
        np.testing.assert_allclose(result, 2.5)
        clipped = compute_torque(cmd, np.zeros(29), np.zeros(29), 1.0, 1.0, 1.0,
                                 np.full(29, 2.0))
        np.testing.assert_allclose(clipped, 2.0)

    def test_user_safe_scales_do_not_reduce_feedback_to_two_percent(self):
        cmd = decode_lowcmd(good_msg(q=1.0, kp=100.0, kd=2.0, tau=1.0),
                            self.low, self.high, self.vel)
        result = compute_torque(cmd, np.zeros(29), np.zeros(29), 0.1, 0.1, 0.2,
                                np.full(29, 50.0))
        # .1/.1/.2 means 10% feedback and 20% feed-forward, not 2% feedback.
        np.testing.assert_allclose(result, 10.2)

    def test_hold_torque_is_zero_at_pose(self):
        zero = hold_torque(np.zeros(29), np.zeros(29), np.zeros(29),
                           np.ones(29), np.ones(29), np.full(29, 5.0))
        np.testing.assert_allclose(zero, 0.0)

    def test_sonic_pd_target_outside_physical_limit_is_not_dropped(self):
        # Reproduce the user's ankle-pitch packet: equilibrium 1.4458 rad,
        # physical upper limit .5236 rad. A PD target is not measured position.
        msg = good_msg()
        msg.motor_cmd[4].q = 1.4458
        high = self.high.copy()
        high[4] = 0.5236
        with self.assertRaisesRegex(CommandRejected, "motor 4"):
            decode_lowcmd(msg, self.low, high, self.vel)
        cmd = decode_lowcmd(msg, self.low, high, self.vel, "sonic-pd")
        self.assertAlmostEqual(cmd.q[4], 1.4458)
        effort = compute_torque(cmd, np.zeros(29), np.zeros(29), 1, 1, 1,
                                np.full(29, 1.0))
        self.assertEqual(effort[4], 1.0)

    def test_sonic_pd_still_rejects_corrupt_targets(self):
        for q in [float("nan"), float("inf"), 100.0]:
            with self.assertRaises(CommandRejected):
                decode_lowcmd(good_msg(q=q), self.low, self.high, self.vel, "sonic-pd")

    def test_physical_guard_blocks_outward_effort_and_allows_recovery(self):
        q, dq, effort = np.zeros(29), np.zeros(29), np.ones(29)
        q[0], q[1], q[2] = 2.0, -2.0, 2.0
        effort[1], effort[2] = -1.0, -1.0
        dq[3], dq[4], effort[4] = 6.0, 6.0, -1.0
        result = guard_joint_effort(effort, q, dq, self.low, self.high, self.vel)
        np.testing.assert_allclose(result[:5], [0, 0, -1, 0, -1])

    def test_large_pd_target_step_is_rate_limited_without_dropping_frame(self):
        command = decode_lowcmd(good_msg(q=3.), self.low, self.high, self.vel, "sonic-pd")
        limited = slew_pd_command(command, np.zeros(29), self.vel, .005, .75)
        np.testing.assert_allclose(limited.q, .025)
        self.assertIs(limited.kp, command.kp)
        np.testing.assert_allclose(command.q, 3.)

    def test_implicit_guard_removes_outward_drive_without_doubling_feedback(self):
        command = decode_lowcmd(good_msg(q=1., dq=1., kp=4., kd=2., tau=.25),
                                self.low, self.high, self.vel, "sonic-pd")
        guarded = guard_implicit_command(command, np.zeros(29), np.zeros(29),
                                        self.low, self.high, self.vel)
        # Native drive receives gains/targets and only the original feed-forward.
        np.testing.assert_allclose(guarded.tau, .25)
        np.testing.assert_allclose(guarded.kp, 4.)
        q = np.full(29, 2.)
        outward = decode_lowcmd(good_msg(q=3., dq=1., kp=4., kd=2., tau=.25),
                               self.low, self.high, self.vel, "sonic-pd")
        guarded = guard_implicit_command(outward, q, np.ones(29),
                                        self.low, self.high, self.vel)
        np.testing.assert_allclose(guarded.kp, 0.)
        np.testing.assert_allclose(guarded.dq, 0.)
        np.testing.assert_allclose(guarded.kd, 2.)
        np.testing.assert_allclose(guarded.tau, 0.)
        # Braking survives and the incoming command was not mutated.
        np.testing.assert_allclose(compute_torque(guarded, q, np.ones(29), 1, 1, 1,
                                                np.full(29, 10.)), -2.)
        np.testing.assert_allclose(outward.kp, 4.)

    def test_implicit_feedback_plus_feedforward_stays_within_motor_effort(self):
        effort = np.full(29, 5.)
        feedforward = np.full(29, 2.)
        budget = implicit_feedback_budget(feedforward, effort)
        np.testing.assert_allclose(budget, 3.)
        for sign in [-1, 1]:
            self.assertTrue(np.all(np.abs(sign*budget + feedforward) <= effort))
        with self.assertRaises(CommandRejected):
            implicit_feedback_budget(np.full(29, 6.), effort)


if __name__ == "__main__":
    unittest.main()
