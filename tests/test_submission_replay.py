import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from submission_replay import embed, extract, validate, REQUIRED, KEY


class SubmissionReplayTests(unittest.TestCase):
    def fixture(self, root):
        raw, sample = root / 'raw', root / 'sample'
        (raw / 'meta').mkdir(parents=True)
        (sample / 'meta').mkdir(parents=True)
        parquet = sample / 'data/chunk-000/episode_000000.parquet'
        parquet.parent.mkdir(parents=True)
        table = pa.table({'observation.state': [[1., 2.], [3., 4.]], 'timestamp': [0., .02]})
        pq.write_table(table, parquet)
        for name in ('bridge.json', 'bridge_saved.json', 'world_trace.json'):
            (raw / name).write_text(json.dumps({'recording_finalized': True, 'recording_error': '', 'samples': 3}))
        (raw / 'world_paths.json').write_text(json.dumps({'body_paths': ['basket'], 'robot_body_names': ['pelvis']}))
        np.savez_compressed(raw / 'bridge.npz', time=np.arange(3)*.02,
            root_pos=np.ones((3, 3)), root_quat_wxyz=np.ones((3, 4)), root_lin_vel_w=np.ones((3, 3)),
            joint_pos=np.ones((3, 29)), left_hand_joint_pos=np.ones((3, 11)), right_hand_joint_pos=np.ones((3, 11)))
        np.savez_compressed(raw / 'world_trace.npz', time=np.arange(3)*.02,
            host_monotonic_ns=np.arange(3)*20000000+1000000000,
            object_transform_xyzw=np.ones((3, 1, 7)), body_link_pos_w=np.ones((3, 1, 3)),
            body_link_quat_wxyz=np.ones((3, 1, 4)), foot_net_contact_force_w=np.ones((3, 2, 3)))
        np.savez_compressed(raw / 'meta/bridge_alignment.npz', sim_time_s=np.arange(3)*.02)
        return raw, sample, parquet, table

    def test_round_trip_preserves_training_data_and_exact_original_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, sample, parquet, table = self.fixture(root)
            descriptor = embed(raw, parquet)
            (sample / 'meta/info.json').write_text(json.dumps({'data_path': 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
                'script_config': {'world_recording': {'retained_in_submission': True, 'embedded_replay': descriptor}}}))
            self.assertTrue(table.equals(pq.read_table(parquet), check_metadata=False))
            self.assertEqual(table.column_names, pq.read_table(parquet).column_names)
            files, summary = validate(sample)
            self.assertTrue(summary['original_world_positions_retained'])
            destination = root / 'extracted'
            extract(sample, destination)
            for name in REQUIRED:
                self.assertEqual((raw / name).read_bytes(), (destination / name).read_bytes())
            with self.assertRaises(FileExistsError):
                extract(sample, destination)
            # A retained-world claim can never survive a removed or corrupt payload.
            corrupt = pq.read_table(parquet)
            md = dict(corrupt.schema.metadata)
            md[KEY] = b'corrupt'
            pq.write_table(corrupt.replace_schema_metadata(md), parquet)
            with self.assertRaisesRegex(ValueError, 'checksum'):
                validate(sample)

    def test_missing_world_evidence_refuses_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            raw, sample, parquet, table = self.fixture(Path(directory))
            (raw / 'world_trace.npz').unlink()
            before = parquet.read_bytes()
            with self.assertRaises(FileNotFoundError):
                embed(raw, parquet)
            self.assertEqual(before, parquet.read_bytes())

    def test_upstream_pre_action_phase_is_retained_in_archive_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            raw, sample, parquet, table = self.fixture(Path(directory))
            status = json.loads((raw/'world_trace.json').read_text())
            status['sampling_phase'] = 'pre_action'
            (raw/'world_trace.json').write_text(json.dumps(status))
            embed(raw, parquet)
            payload = pq.read_schema(parquet).metadata[KEY]
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                manifest = json.loads(archive.read('manifest.json'))
            self.assertEqual(manifest['world_sampling_phase'], 'pre_action')
            self.assertIn('pre-action world', manifest['clock'])
            self.assertNotIn('post-step', manifest['clock'])


if __name__ == '__main__':
    unittest.main()
