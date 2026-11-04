"""Ensure automatic submission cannot erase or mislabel a completed capture."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from capture_session import finalize, initialize, publish_mentor_submission
from prepare_mentor_sample import (ensure_submission, prepare, validate_submission, MENTOR_FILES,
                                  META_FILES, _rename_directory_without_replacing, summarize_manager_timing)


class MentorSubmissionTests(unittest.TestCase):
    def test_manager_timing_preserves_real_stage_costs_and_marks_unavailable_packets(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            packets = root / 'pico_input'
            packets.mkdir()
            np.savez(packets / 'pose_000000.npz', timestamp_monotonic=[10.])
            self.assertIsNone(summarize_manager_timing(root))
            keys = ['timing_' + stage + '_monotonic_ns' for stage in
                    ('read', 'body_done', 'hand_done', 'retarget_done', 'pack_done', 'send_done')]
            times = np.array([1000, 1001, 1003, 1013, 1014, 1015], dtype=np.int64) * 1000000
            np.savez(packets / 'pose_000001.npz', **{key: [value] for key, value in zip(keys, times)})
            np.savez(packets / 'pose_000002.npz', **{key: [value] for key, value in zip(keys, times[::-1])})
            summary = summarize_manager_timing(root)
            self.assertEqual(summary['packets_with_timing'], 1)
            self.assertEqual(summary['packets_without_timing'], 1)
            self.assertEqual(summary['invalid_timing_packets'], 1)
            self.assertEqual(summary['stages_ms']['resample_retarget_and_preview']['p95'], 10.)
            self.assertEqual(summary['stages_ms']['read_to_send']['max'], 15.)

    def test_cleanup_refuses_to_delete_world_trace_without_retained_replay(self):
        from capture_session import _publish_mentor_only_output
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            run = root / '.capture_work' / 'recording'
            output = root / 'pick_up_the_basket_0'
            session = initialize(run, mentor_only_output=output, publication_root=root)
            (run / 'world_trace.npz').write_bytes(b'original world evidence')
            (output / 'meta').mkdir(parents=True)
            (output / 'meta/info.json').write_text(json.dumps({'script_config': {'submission': {
                'source_session_id': session['session_id']}}}))
            result = {'status': 'complete', 'mentor_submission': {'status': 'complete', 'directory': str(output)}}
            with patch('prepare_mentor_sample.validate_submission', return_value={
                    'status': 'complete', 'mentor_file_layout_match': True}):
                with self.assertRaisesRegex(RuntimeError, 'original world trajectory'):
                    _publish_mentor_only_output(run, result)
            self.assertEqual((run / 'world_trace.npz').read_bytes(), b'original world evidence')

    def test_new_livingroom_session_enables_auto_submission(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.assertTrue(initialize(root / "livingroom")["publish_mentor_sample"])
            self.assertFalse(initialize(root / "basic", profile="bridge")["publish_mentor_sample"])

    def test_completed_finalize_publishes_separate_sample(self):
        with tempfile.TemporaryDirectory() as name:
            run = Path(name) / "recording"
            initialize(run)
            marker = run / "completion.json"
            marker.write_text('{"status":"complete"}\n')
            before = marker.read_bytes()
            sample = {"status": "complete", "directory": str(run.with_name(run.name + "_mentor_sample"))}
            with patch("capture_session.inspect", return_value=({"status": "complete"}, {})), \
                 patch("prepare_mentor_sample.ensure_submission", return_value=sample) as publish:
                result = finalize(run)
            self.assertEqual(result["mentor_submission"], sample)
            publish.assert_called_once_with(run.resolve())
            self.assertEqual(marker.read_bytes(), before)

    def test_incomplete_capture_never_publishes(self):
        with patch("prepare_mentor_sample.ensure_submission") as publish:
            result = publish_mentor_submission(Path("missing"), {"status": "incomplete"})
        publish.assert_not_called()
        self.assertEqual(result, {"status": "incomplete"})

    def test_submission_failure_preserves_completed_raw_capture(self):
        with tempfile.TemporaryDirectory() as name:
            run = Path(name)
            initialize(run)
            marker = run / "completion.json"
            marker.write_text('{"status":"complete"}\n')
            before = marker.read_bytes()
            with patch("prepare_mentor_sample.ensure_submission", side_effect=OSError("disk full")):
                result = publish_mentor_submission(run, {"status": "complete"})
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["mentor_submission"]["status"], "incomplete")
            self.assertIn("disk full", result["mentor_submission"]["error"])
            self.assertEqual(marker.read_bytes(), before)

    def test_repeated_publish_keeps_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            run = root / "recording"
            session = initialize(run)
            sample = root / "recording_mentor_sample"
            (sample / "meta").mkdir(parents=True)
            (sample / "meta/info.json").write_text(json.dumps({"script_config": {"submission": {"source_session_id": session["session_id"]}}}))
            sentinel = sample / "original.bin"
            sentinel.write_bytes(b"must remain unchanged")
            with patch("prepare_mentor_sample.validate_submission", return_value={"status": "complete"}), \
                 patch("prepare_mentor_sample.prepare", side_effect=AssertionError("must not rebuild")):
                ensure_submission(run)
                ensure_submission(run)
            self.assertEqual(sentinel.read_bytes(), b"must remain unchanged")
            self.assertEqual((root / "latest_mentor_sample.txt").read_text(), str(sample.resolve()) + "\n")

    def test_foreign_submission_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            run = root / "recording"
            initialize(run)
            sample = root / "recording_mentor_sample"
            (sample / "meta").mkdir(parents=True)
            origin = sample / "meta/info.json"
            origin.write_text('{"script_config":{"submission":{"source_session_id":"different-session"}}}')
            before = origin.read_bytes()
            with patch("prepare_mentor_sample.validate_submission", return_value={"status": "complete"}):
                with self.assertRaisesRegex(ValueError, "different session"):
                    ensure_submission(run)
            self.assertEqual(origin.read_bytes(), before)
            self.assertFalse((root / "latest_mentor_sample.txt").exists())

    def test_old_extended_submission_is_preserved(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            run = root / "recording"
            initialize(run)
            old = root / "recording_mentor_sample/meta/submission.json"
            old.parent.mkdir(parents=True)
            old.write_text('{"source_session_id":"legacy"}')
            before = old.read_bytes()
            clean = root / "recording_mentor_sample_minimal"
            with patch("prepare_mentor_sample.prepare", return_value={"status": "complete"}) as build:
                result = ensure_submission(run)
            build.assert_called_once_with(run.resolve(), clean.resolve())
            self.assertEqual(old.read_bytes(), before)
            self.assertEqual(result["directory"], str(clean.resolve()))

    def test_prepare_copies_current_metadata_files_and_preserves_source(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            run = root / "recording"
            initialize(run)
            for relative in MENTOR_FILES:
                p = run / relative
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"fixture-data" if p.suffix not in (".json", ".jsonl") else b"{}")
            info_path = run / "meta/info.json"
            info_path.write_text(json.dumps({"script_config": {"lerobot_schema_mode": "kinetra_46443c1", "training_artifacts": {"manifest": "../training/manifest.json"},
                                                                 "source_run": str(run)}}))
            (run / "meta/stats.json").write_text("{}")
            (run / "training").mkdir()
            (run / "training/extra.npz").write_bytes(b"auxiliary")
            (run / 'pico_input').mkdir(exist_ok=True)
            np.savez(run / 'pico_input/pose_000000.npz', **{
                'timing_' + stage + '_monotonic_ns': np.array([1000000000 + i * 1000000], dtype=np.int64)
                for i, stage in enumerate(('read', 'body_done', 'hand_done', 'retarget_done', 'pack_done', 'send_done'))})
            original_info = info_path.read_bytes()
            clean = root / "recording_mentor_sample"
            with patch("capture_session.inspect", return_value=({"status": "complete"}, {})), \
                 patch("export_capture_dataset.validate", return_value={"rows": 2}), \
                 patch("prepare_mentor_sample.validate_submission", return_value={"status": "complete", "rows": 2}):
                prepare(run, clean)
            self.assertEqual({str(p.relative_to(clean)) for p in clean.rglob("*") if p.is_file()}, MENTOR_FILES)
            self.assertEqual({p.name for p in (clean / "meta").iterdir()}, set(META_FILES))
            self.assertEqual(info_path.read_bytes(), original_info)
            self.assertEqual((run / "training/extra.npz").read_bytes(), b"auxiliary")
            self.assertTrue(clean.with_name(clean.name + ".validation.json").is_file())
            metadata = json.loads((clean / "meta/info.json").read_text())["script_config"]
            self.assertNotIn("training_artifacts", metadata)
            self.assertNotIn("source_run", metadata)
            self.assertEqual(metadata["submission_layout"], "mentor_minimal_v1")
            timing = metadata['manager_processing_timing']
            self.assertEqual(timing['packets_with_timing'], 1)
            self.assertEqual(timing['stages_ms']['read_to_send']['max'], 5.)

    def test_minimal_validation_rejects_extra_auxiliary_file(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for relative in MENTOR_FILES:
                p = root / relative
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("{}")
            (root / "meta/info.json").write_text(json.dumps({"script_config": {"lerobot_schema_mode": "kinetra_46443c1"}}))
            (root / "meta/unexpected.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "exactly the versioned dataset files"):
                validate_submission(root)

    def test_atomic_promotion_never_overwrites_even_an_empty_old_directory(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            stage, old = root / 'staging', root / 'old'
            stage.mkdir()
            (stage / 'evidence').write_bytes(b'staged')
            old.mkdir()
            with self.assertRaises(FileExistsError):
                _rename_directory_without_replacing(stage, old)
            self.assertEqual(list(old.iterdir()), [])
            self.assertEqual((stage / 'evidence').read_bytes(), b'staged')

    def test_atomic_failed_copy_retains_raw_and_never_publishes_partial_output(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            run = root / '.capture_work' / 'capture'
            output = root / 'pick_up_the_basket_test'
            initialize(run)
            before = (run / 'session.json').read_bytes()
            def fail_copy(source, destination, **kwargs):
                destination.mkdir()
                (destination / 'partial').write_bytes(b'partial')
                raise OSError('disk full')
            with patch('prepare_mentor_sample.prepare', side_effect=fail_copy):
                with self.assertRaisesRegex(OSError, 'disk full'):
                    ensure_submission(run, destination=output, write_validation=False,
                                      atomic_publish=True, update_latest=False)
            self.assertFalse(output.exists())
            self.assertEqual((run / 'session.json').read_bytes(), before)
            self.assertEqual(list(run.parent.iterdir()), [run])

    def test_real_minimal_roundtrip_retains_task_and_detects_corruption_without_sidecar(self):
        import export_capture_dataset as exporter
        from tests.test_capture_dataset import DatasetTests
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            run = root / '.capture_work' / 'pick_up_the_basket_fixture'
            initialize(run, task='pick_up_the_basket')
            with patch.object(exporter, 'build_arrays', return_value=DatasetTests().fixture(run)):
                exporter.export(run, task='pick_up_the_basket')
            output = root / 'pick_up_the_basket_20261031_fixture'
            with patch('capture_session.inspect', return_value=({'status': 'complete'}, {})):
                result = ensure_submission(run, destination=output, write_validation=False,
                                           atomic_publish=True, update_latest=False)
            self.assertEqual(result['rows'], 12)
            self.assertEqual(result['decoded_video_frames'], 12)
            self.assertEqual(result['file_count'], 9)
            self.assertEqual({p.name for p in root.iterdir()}, {'.capture_work', output.name})
            task = json.loads((output / 'meta/tasks.jsonl').read_text())
            episode = json.loads((output / 'meta/episodes.jsonl').read_text())
            self.assertEqual(task['task'], 'pick_up_the_basket')
            self.assertEqual(episode['tasks'], ['pick_up_the_basket'])
            self.assertEqual(ensure_submission(run, destination=output, update_latest=False)['rows'], 12)
            with (output / 'meta/tasks.jsonl').open('a') as stream:
                stream.write('\n')
            with self.assertRaisesRegex(ValueError, 'checksum differs: meta/tasks.jsonl'):
                validate_submission(output)


if __name__ == "__main__":
    unittest.main()
