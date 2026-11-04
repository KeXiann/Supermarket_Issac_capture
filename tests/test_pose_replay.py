import tempfile
from pathlib import Path
import unittest

import numpy as np

from pose_replay import PoseReplay


class PoseReplayTests(unittest.TestCase):
    def archive(self, path, timestamps):
        n = len(timestamps)
        np.savez(path, packet_time=np.asarray(timestamps),
                 smpl_pose=np.broadcast_to(np.arange(n)[:,None,None], (n,21,3)),
                 smpl_joints=np.zeros((n,24,3)),
                 body_quat_w=np.tile([1.,0.,0.,0.],(n,1)), joint_pos=np.zeros((n,29)))

    def test_full_duration_includes_warmup_and_tail_without_truncating(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'pose.npz';self.archive(path,[100.,120.,166.138])
            replay = PoseReplay(path)
            self.assertAlmostEqual(replay.run_seconds(0.,5.,1.),72.138)
            self.assertEqual(replay.run_seconds(30.,5.,1.),30.)
            replay = PoseReplay(path,start=60.)
            self.assertAlmostEqual(replay.run_seconds(0.,2.,1.),9.138)
            with self.assertRaises(ValueError):PoseReplay(path,start=67.)

    def test_final_reference_advances_window_after_source_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'pose.npz';self.archive(path,[100.,100.03,100.06])
            replay = PoseReplay(path)
            for i,t in enumerate(np.arange(0.,.40,.02)):
                packet = replay.payload(t,200.+t)
                self.assertEqual(packet['frame_index'][-1],i)
                self.assertTrue(np.all(np.diff(packet['frame_index'])==1))
            self.assertEqual(replay.last_source_frame,2)
            np.testing.assert_array_equal(packet['smpl_pose'],np.full((5,21,3),2.))

    def test_rejects_reversed_source_clock(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'pose.npz';self.archive(path,[100.,100.03,100.02])
            with self.assertRaisesRegex(ValueError,'strictly increasing'):PoseReplay(path)

    def test_preserves_recorded_hands_and_controller_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'pose.npz'
            self.archive(path, [100., 100.02, 100.04])
            with np.load(path) as archive:
                data = dict(archive)
            data.update(left_hand_joints=np.full((3, 7), .25),
                        right_hand_joints=np.full((3, 7), -.5),
                        left_trigger=np.full((3, 1), .3), right_grip=np.full((3, 1), .6))
            np.savez(path, **data)
            packet = PoseReplay(path).payload(.02, 200.)
            np.testing.assert_allclose(packet['left_hand_joints'], .25)
            np.testing.assert_allclose(packet['right_hand_joints'], -.5)
            np.testing.assert_allclose(packet['left_trigger'], [.3])
            self.assertEqual(packet['timestamp_monotonic'][0], 200.)
