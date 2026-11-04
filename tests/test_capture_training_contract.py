"""Regression checks for source recovery, reference rotations and durable logs."""
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

import msgpack
import numpy as np
from scipy.spatial.transform import Rotation

from export_capture_dataset import mentor_rot6d, mentor_target_orientation
from prepare_trajectory import load_human
from sonic_debug_recorder import MAGIC, validate_recording


class CaptureTrainingContract(unittest.TestCase):
    def test_rotation_matches_pinned_reference(self):
        path = Path(__file__).resolve().parents[1] / "assets/collection_reference_2609/gear_sonic/utils/data_collection/transforms.py"
        spec = importlib.util.spec_from_file_location("reference_transforms", path)
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        rng = np.random.default_rng(42)
        q = Rotation.random(60, random_state=rng).as_quat()[:, [3, 0, 1, 2]]
        np.testing.assert_allclose(mentor_rot6d(q), reference.quat_to_rot6d(q))
        np.testing.assert_allclose(mentor_rot6d(q.reshape(20, 3, 4)).reshape(20, 18), reference.quat_to_rot6d(q).reshape(20, 18))
        heading, valid = rng.uniform(-1, 1, 60), np.arange(60) % 7 != 0
        actual = mentor_target_orientation(q, heading, valid)
        initial_yaw = None
        for i in np.flatnonzero(valid):
            target = Rotation.from_euler("z", heading[i]) * Rotation.from_quat(q[i, [1, 2, 3, 0]])
            euler = target.as_euler("ZYX")
            if initial_yaw is None:
                initial_yaw = euler[0]
            euler[0] -= initial_yaw
            normalized = Rotation.from_euler("ZYX", euler).as_quat()[[3, 0, 1, 2]]
            np.testing.assert_allclose(actual[i], reference.quat_to_rot6d(normalized), atol=1e-12)

    def test_raw_fields_recovered_from_older_archive(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "pico_input").mkdir()
            np.savez(root / "pico_replay_archive.npz", packet_time=[10., 10.02], source_frame=[4, 5],
                     smpl_pose=np.zeros((2, 21, 3)), smpl_joints=np.zeros((2, 24, 3)),
                     body_quat_w=np.tile([1., 0., 0., 0.], (2, 1)), joint_pos=np.zeros((2, 29)))
            for i in range(2):
                np.savez(root / f"pico_input/pose_{i:06d}.npz", timestamp_monotonic=[10. + i * .02],
                         frame_index=[4 + i], vr_position=np.arange(9) + i,
                         vr_orientation=np.tile([1., 0., 0., 0.], 3), heading_increment=[.01],
                         timestamp_realtime=[1800000000. + i * .02])
            human, _ = load_human(root)
            self.assertEqual(human["vr_position"].shape, (2, 9))
            self.assertEqual(human["vr_orientation"].shape, (2, 12))
            self.assertEqual(human["timestamp_realtime"].shape, (2, 1))
            with np.load(root / "pico_replay_archive.npz", allow_pickle=False) as original:
                self.assertNotIn("vr_position", original.files)

    def test_debug_stream_rejects_truncation_drop_and_unclosed_file(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "debug.msgpack"
            marker = path.with_suffix(".msgpack.json")
            def write(indices, complete=True):
                rows = [{"topic": "robot_config", "payload": {"body_action_alignment_version": 2}}]
                rows += [{"topic": "g1_debug", "payload": {"body_action_alignment_version": 2,
                          "index": i, "state_monotonic_ns": 1000000 + i * 20000000}} for i in indices]
                with path.open("wb") as stream:
                    stream.write(MAGIC)
                    for row in rows:
                        blob = msgpack.packb(row, use_bin_type=True)
                        stream.write(struct.pack("<Q", len(blob)))
                        stream.write(blob)
                marker.write_text(json.dumps({"format": "sonic_debug_msgpack_v2", "complete": complete, "records": len(rows)}))
            write([0, 1, 2])
            self.assertEqual(validate_recording(path)[0]["state_action_cycles"], 3)
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, "truncated"):
                validate_recording(path)
            write([0, 2])
            with self.assertRaisesRegex(ValueError, "missing, duplicated or reordered"):
                validate_recording(path)
            write([0, 1], complete=False)
            with self.assertRaisesRegex(ValueError, "close marker"):
                validate_recording(path)


if __name__ == "__main__":
    unittest.main()
