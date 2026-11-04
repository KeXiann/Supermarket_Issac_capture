"""Reference schema and real Arrow/H264 roundtrip; synthetic non-training fixtures."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import pyarrow.parquet as pq

import export_capture_dataset as exporter
from capture_session import initialize, inspect, preflight


class DatasetTests(unittest.TestCase):
    def fixture(self, run):
        info = exporter.template()
        n = 12
        arrays = {k: np.zeros((n, *v["shape"]), dtype=v["dtype"]) for k, v in info["features"].items() if v["dtype"] != "video"}
        for k in ("frame_index", "index"):
            arrays[k][:, 0] = np.arange(n)
        arrays["timestamp"][:, 0] = np.arange(n) / 50.
        arrays["observation.state"][:] = .1
        arrays["action.wbc"][:] = .2
        arrays["observation.hand_joint_position"][:] = .15
        arrays["action.hand_joint_position"][:] = .7
        for key in ("observation.body_valid", "observation.hand_valid", "action.hand_valid", "teleop.stream_mode"):
            arrays[key][:] = 1
        arrays["observation.control_cycle_index"][:, 0] = np.arange(n)
        arrays["observation.body_monotonic_ns"][:, 0] = 1000000000 + np.arange(n) * 20000000
        arrays["teleop.smpl_frame_index"][:, 0] = np.arange(n)
        source = run / "ego_view.mp4"
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 25, (640, 360))
        self.assertTrue(writer.isOpened())
        for i in range(3):
            writer.write(np.full((360, 640, 3), i * 60, dtype=np.uint8))
        writer.release()
        alignment = {"camera_source_frame": np.repeat(np.arange(3), 4),
                     "human_valid": np.ones(n, dtype=bool), "camera_age_s": np.zeros(n),
                     "sim_time_s": np.arange(n) / 50.}
        availability = {k: {"available": k != "observation.previous_body_action_valid",
                            "source": "synthetic roundtrip fixture; not task-training data"} for k in arrays}
        semantics = {"stage": "synthetic_test_fixture", "field_availability": availability,
                     "body_protocol_v2": False}
        return info, arrays, alignment, semantics

    def test_reference_schema_real_parquet_and_last_video_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            fixture = self.fixture(run)
            original = (run / "ego_view.mp4").read_bytes()
            with patch.object(exporter, "build_arrays", return_value=fixture):
                result = exporter.export(run)
            self.assertEqual(result["numeric_columns"], 75)
            self.assertEqual(result["decoded_video_frames"], 12)
            self.assertTrue((run / "video").is_symlink())
            self.assertEqual((run / "video").resolve(), run / "videos")
            table = pq.read_table(run / "data/chunk-000/episode_000000.parquet")
            import pyarrow as pa
            self.assertEqual(table['action.wbc'].type, pa.list_(pa.float64(), 29))
            self.assertEqual(json.loads(table.schema.metadata[b'huggingface'])['info']['features']['action.wbc']['length'], 29)
            self.assertEqual(table["action.wbc"].to_pylist()[0][0], .2)
            self.assertAlmostEqual(table["observation.hand_joint_position"].to_pylist()[0][0], .15)
            self.assertAlmostEqual(table["action.hand_joint_position"].to_pylist()[0][0], .7)
            self.assertEqual(original, (run / "ego_view.mp4").read_bytes())
            with self.assertRaises(FileExistsError):
                exporter.export(run)

    def test_corrupt_video_and_broken_index_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            with patch.object(exporter, "build_arrays", return_value=self.fixture(run)):
                exporter.export(run)
            file = run / "data/chunk-000/episode_000000.parquet"
            table = pq.read_table(file)
            import pyarrow as pa
            bad = table.set_column(table.schema.get_field_index("index"), "index", pa.array([*range(11), 99], type=pa.int64()))
            pq.write_table(bad, file)
            with self.assertRaisesRegex(ValueError, "invalid index"):
                exporter.validate(run)
            pq.write_table(table, file)
            video = next((run / "videos").rglob("*.mp4"))
            video.write_bytes(b"truncated")
            with self.assertRaisesRegex(ValueError, "video/data count"):
                exporter.validate(run)

    def test_nonempty_sessions_and_missing_data_never_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            initialize(run)
            with self.assertRaises(FileExistsError):
                initialize(run)
            report, _ = inspect(run)
            self.assertEqual(report["status"], "incomplete")
            self.assertFalse((run / "completion.json").exists())

    def test_explicit_bridge_schema_retains_analog_hand_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            with patch.object(exporter, "build_arrays", return_value=self.fixture(run)):
                result = exporter.export(run, schema_mode="bridge")
            self.assertEqual(result["numeric_columns"], 75)
            table = pq.read_table(run / "data/chunk-000/episode_000000.parquet")
            self.assertIn("observation.previous_hand_action_joint_position", table.column_names)
            self.assertIn("action.aligned_hand_joint_position", table.column_names)

    def test_pinned_builders_match_template_and_metadata_corruption_is_rejected(self):
        from check_reference_schema import check
        self.assertEqual(check()["numeric_columns"], 75)
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            with patch.object(exporter, "build_arrays", return_value=self.fixture(run)):
                exporter.export(run)
            path = run / 'meta/info.json'
            info = json.loads(path.read_text())
            info['features']['action.wbc']['names'][0] = 'wrong_joint'
            path.write_text(json.dumps(info))
            with self.assertRaisesRegex(ValueError, 'feature metadata differs'):
                exporter.validate(run)

    def test_launcher_rejects_old_session_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            initialize(run)
            old = run / 'capture_sonic.json'
            old.write_text('old capture')
            with self.assertRaises(FileExistsError):
                preflight(run, 'sonic')
            self.assertEqual(old.read_text(), 'old capture')

    def test_launcher_rejects_a_pid_from_another_session(self):
        import os
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            session = initialize(run)
            (run / 'components').mkdir()
            (run / 'components/bridge.json').write_text(json.dumps({
                'status': 'running', 'session_id': session['session_id'], 'pid': os.getpid()}))
            with self.assertRaisesRegex(RuntimeError, 'PID no longer belongs'):
                preflight(run, 'sonic')


if __name__ == "__main__":
    unittest.main()
