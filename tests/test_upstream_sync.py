"""Capture/replay physics agreement and real video/data-standard regression checks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import data_standard
import export_capture_dataset as exporter
from tests import test_capture_dataset
from upstream_contract import (COMMIT, check_physics_metadata,
                               document, model_contract, replay_selection, task_initialization)


class UpstreamSyncTests(unittest.TestCase):
    def test_global_statistics_match_example_and_reject_corruption(self):
        example = Path(__file__).resolve().parents[1] / 'assets/upstream/7e5fd9d0ae2f/configs/data_standard/examples/data_standard_new/meta/stats.json'
        expected = json.loads(example.read_text())
        with tempfile.TemporaryDirectory() as name:
            run = Path(name)
            with patch.object(exporter, 'build_arrays', return_value=test_capture_dataset.DatasetTests().fixture(run)):
                exporter.export(run)
            path = run / 'meta/stats.json'
            actual = json.loads(path.read_text())
            self.assertEqual(set(actual), set(expected))
            self.assertEqual(set(actual['__fingerprints__']), set(expected['__fingerprints__']))
            for field in actual.keys() - {'__fingerprints__'}:
                self.assertEqual(set(actual[field]), set(expected[field]))
                for statistic in actual[field]:
                    self.assertEqual(np.shape(actual[field][statistic]), np.shape(expected[field][statistic]))
            np.testing.assert_allclose(actual['timestamp']['q01'], [.0022])
            np.testing.assert_allclose(actual['timestamp']['q99'], [.2178])
            for kind in ('quantile', 'fingerprint'):
                with self.subTest(kind=kind):
                    corrupt = copy.deepcopy(actual)
                    if kind == 'quantile':
                        corrupt['timestamp']['q01'] = [123.]
                    else:
                        corrupt['__fingerprints__']['timestamp'] = 'sha256:' + '0' * 64
                    path.write_text(json.dumps(corrupt))
                    with self.assertRaisesRegex(ValueError, 'Dataset statistics'):
                        exporter.validate(run)

    def test_current_export_preserves_published_contract_and_all_hand_columns(self):
        with tempfile.TemporaryDirectory() as name:
            run = Path(name)
            with patch.object(exporter, 'build_arrays', return_value=test_capture_dataset.DatasetTests().fixture(run)):
                exporter.export(run)
            actual = json.loads((run / 'meta/info.json').read_text())
            expected = document('configs/data_standard/meta/info.json')
            self.assertEqual(actual['features'], expected['features'])
            for key in ('body_alignment', 'hand_recording'):
                for field, value in expected['script_config'][key].items():
                    self.assertEqual(actual['script_config'][key][field], value)
            stats = json.loads((run / 'meta/episodes_stats.jsonl').read_text())['stats']
            self.assertEqual(set(stats), set(expected['features']))
            self.assertEqual(stats[data_standard.VIDEO_KEY]['count'], [12])
            self.assertEqual(len(expected['features']), 76)

    def test_video_statistics_use_rgb_population_std_and_frame_count(self):
        first = np.zeros((360, 640, 3), dtype=np.uint8)
        first[:, :, 2] = 255
        second = np.zeros_like(first)
        second[:, :, 0] = 255
        stats = data_standard.RGBStatistics()
        stats.add_bgr(first)
        stats.add_bgr(second)
        result = stats.result()
        np.testing.assert_allclose(np.ravel(result['mean']), [.5, 0, .5])
        np.testing.assert_allclose(np.ravel(result['std']), [.5, 0, .5])
        self.assertEqual(result['count'], [2])

    def test_corrupted_video_statistics_fail(self):
        with tempfile.TemporaryDirectory() as name:
            run = Path(name)
            with patch.object(exporter, 'build_arrays', return_value=test_capture_dataset.DatasetTests().fixture(run)):
                exporter.export(run)
            path = run / 'meta/episodes_stats.jsonl'
            row = json.loads(path.read_text())
            row['stats'][data_standard.VIDEO_KEY]['mean'][0][0][0] += .1
            path.write_text(json.dumps(row) + '\n')
            with self.assertRaisesRegex(ValueError, 'decoded RGB'):
                exporter.validate(run)

    def test_task1_named_state_and_root_are_not_replaced_with_policy_nominal(self):
        scene, state = task_initialization()
        np.testing.assert_allclose(np.asarray(scene['root_T'])[:3, 3], [6.338, 4.949999809265137, .792], atol=1e-9)
        model = model_contract()
        self.assertEqual(state['joint_names'].tolist(), model['joint_order'])
        self.assertGreater(np.max(np.abs(state['observation.state'][0] - model['g1_29dof']['default_angles_rad'])), 1.)

    def test_quality_gate_checks_readback_limits_and_backend(self):
        model = model_contract()
        metadata = {'pd_backend': 'external', 'effort_limits_motor_order': model['g1_29dof']['effort_limit_nm'],
                    'physics_contract': {'profile': 'upstream', 'commit': COMMIT, 'pd_backend': 'external',
                                         'joint_order': model['joint_order'], 'armature_kg_m2': [.01] * 29}}
        self.assertTrue(check_physics_metadata(metadata))
        wrong = copy.deepcopy(metadata)
        wrong['effort_limits_motor_order'][0] = 139.
        self.assertFalse(check_physics_metadata(wrong))
        wrong = copy.deepcopy(metadata)
        wrong['pd_backend'] = 'implicit'
        self.assertFalse(check_physics_metadata(wrong))

    def test_replay_uses_recorded_backend_and_rejects_override(self):
        with tempfile.TemporaryDirectory() as name:
            bundle = Path(name) / 'input.npz'
            bundle.with_suffix('.json').write_text(json.dumps({'physics_contract': {
                'profile': 'legacy', 'pd_backend': 'implicit'}}))
            with self.assertRaisesRegex(ValueError, 'Historical physics profile removed'):
                replay_selection(bundle)
            with self.assertRaisesRegex(ValueError, 'Historical physics profile removed'):
                replay_selection(bundle, 'upstream', 'external')
            bundle.with_suffix('.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'lacks physics'):
                replay_selection(bundle)

    def test_historical_schema_remains_distinct(self):
        old = {'script_config': {'lerobot_schema_mode': 'mentor_2609_exact'}}
        expected = data_standard.expected_template(old)
        self.assertEqual(len(expected['features']), 74)
        self.assertNotIn('action.aligned_hand_joint_position', expected['features'])


if __name__ == '__main__':
    unittest.main()
