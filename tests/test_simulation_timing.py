import unittest
from unittest.mock import patch
from simulation_timing import FrameSchedule, PhysicsPacer, LoopPerformance


class SchedulingTests(unittest.TestCase):
    def test_30fps_at_200hz_does_not_drift_to_28fps(self):
        camera = FrameSchedule(30)
        frames = []
        for step in range(2000):
            now = step * .005
            if camera.due(now):
                frames.append(now)
                camera.captured(now)
        self.assertEqual(len(frames), 300)
        self.assertLess(max(b-a for a, b in zip(frames, frames[1:])), .041)

    def test_late_frame_does_not_cause_duplicate_catchup_frames(self):
        camera = FrameSchedule(30)
        camera.captured(10.)
        self.assertTrue(camera.due(11.))
        camera.captured(11.)
        self.assertFalse(camera.due(11.001))
        self.assertTrue(camera.due(11.034))

    def test_calibration_wait_does_not_become_physics_catchup_debt(self):
        pacer = PhysicsPacer(.005)
        self.assertEqual(pacer.remaining(100.), 0.)
        pacer.begin(125.)
        self.assertAlmostEqual(pacer.remaining(125.002), .003)
        for step in range(1, 200):
            pacer.begin(125. + step*.005)
            self.assertAlmostEqual(pacer.remaining(125. + step*.005 + .002), .003)
        self.assertEqual(pacer.steps, 200)

    def test_metrics_exclude_wait_and_final_disk_flush(self):
        perf = LoopPerformance(.005, profile=True)
        with patch('simulation_timing.time.monotonic', side_effect=[1., 2., 50., 50.001, 50.002, 50.005, 50.007]):
            perf.begin(); perf.mark('render'); perf.finish(False)
            perf.begin(); perf.mark('physics'); perf.finish(True)
            perf.begin(); perf.finish(True)
        result = perf.summary()
        self.assertEqual(result['physical_steps'], 2)
        self.assertAlmostEqual(result['active_wall_seconds'], .007)
        self.assertAlmostEqual(result['physics_steps_per_wall_second'], 2/.007)
        self.assertNotIn('render', result['phases'])

    def test_long_startup_stall_does_not_speed_up_the_whole_next_episode(self):
        pacer = PhysicsPacer(.005)
        pacer.begin(10.)
        self.assertEqual(pacer.remaining(12.), 0.)
        self.assertAlmostEqual(pacer.discarded_debt_s, 1.975)
        now = 12.
        for _ in range(10):
            now += .001
            delay = pacer.remaining(now)
            if delay > 0:
                break
        self.assertGreater(delay, 0.)
        self.assertLess(pacer.steps, 10)


if __name__ == '__main__':
    unittest.main()
