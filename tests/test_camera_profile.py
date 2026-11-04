import json
from pathlib import Path
import tempfile
import unittest

from camera_profile import resolve_camera_profile


class CameraProfileTests(unittest.TestCase):
    def test_legacy_scene_keeps_household_values(self):
        result = resolve_camera_profile('/example/livingroom/assets/scenes/robot_livingroom_scene.usda',extrinsics_config='virtual')
        self.assertEqual(result['mount_position_m'], [.1, 0., .35])
        self.assertEqual(result['pitch_down_deg'], 45.)
        self.assertEqual(result['focal_length_mm'], 14.)
        self.assertIsNone(result['config_path'])

    def test_supermarket_loads_current_camera_contract_and_hash(self):
        from upstream_camera import SCENE, COMMIT
        result = resolve_camera_profile(SCENE)
        self.assertEqual(result['capture_contract']['commit'], COMMIT)
        self.assertAlmostEqual(result['pitch_down_deg'], 41.10635710810979)
        self.assertEqual(result['focal_length_mm'], 14.)
        self.assertEqual(len(result['extrinsics']['config_sha256']), 64)

    def test_explicit_operator_values_override_only_the_selected_fields(self):
        from upstream_camera import SCENE
        result = resolve_camera_profile(SCENE, pitch=32.)
        self.assertEqual(result['pitch_down_deg'], 32.)
        self.assertEqual(result['focal_length_mm'], 14.)
        self.assertEqual(result['override_fields'], ['pitch_down_deg'])

    def test_invalid_camera_configuration_fails_before_simulation(self):
        for bad_value in (float('nan'), -1., 85.):
            with self.subTest(pitch=bad_value), self.assertRaises(ValueError):
                resolve_camera_profile(pitch=bad_value)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'test.camera.json'
            path.write_text(json.dumps({'schema_version': 2, 'profile': 'unknown'}))
            with self.assertRaises(ValueError):
                resolve_camera_profile(config=path)
