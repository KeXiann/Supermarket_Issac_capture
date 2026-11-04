import unittest
import numpy as np
from replay_clock import recorded_control_clock,due_rows


class ReplayClockTests(unittest.TestCase):
    def test_half_speed_clock_is_not_expanded_to_wall_time(self):
        release,obs,duration,report=recorded_control_clock([1000,1010,1020],[1000000000,1020000000,1040000000],[1004000000,1024000000,1044000000],[0,.02,.04])
        np.testing.assert_allclose(release,[0,.01,.02],atol=1e-12)
        np.testing.assert_allclose(obs,[-.002,.008,.018],atol=1e-12)
        self.assertAlmostEqual(report['measured_simulation_to_host_ratio'],.5)
        self.assertAlmostEqual(duration,.03)

    def test_repeated_feedback_rows_preserve_all_actions(self):
        release,_,_,_=recorded_control_clock([1000,1000,1020],[1000000000,1000000000,1040000000],[1004000000,1024000000,1044000000],[0,.02,.04])
        self.assertTrue((np.diff(release)>0).all())
        self.assertEqual(list(due_rows([0,.002,.004,.01],-1,.005)),[0,1,2])
        self.assertEqual(list(due_rows([0,.002,.004,.01],2,.005)),[])

    def test_backwards_tick_fails(self):
        with self.assertRaises(ValueError):
            recorded_control_clock([1000,999],[1,2],[2,3],[0,.02])

    def test_uint32_wrap(self):
        release,_,_,_=recorded_control_clock([4294967290,4,14],[1000000000,1020000000,1040000000],[1004000000,1024000000,1044000000],[0,.02,.04])
        np.testing.assert_allclose(release,[0,.01,.02],atol=1e-8)


if __name__=='__main__':unittest.main()
