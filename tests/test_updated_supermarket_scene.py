import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / 'assets/scenes/pico_supermarket_task1_7e5fd9d.usda'
MANIFEST = SCENE.with_suffix('.json')
CAMERA = SCENE.with_suffix('.camera.json')


class UpdatedSupermarketSceneTests(unittest.TestCase):
    def test_updated_scene_is_complete_and_pinned_to_upstream(self):
        self.assertTrue(SCENE.is_file())
        data = json.loads(MANIFEST.read_text())
        self.assertEqual(data['source_commit'], '7e5fd9d0ae2f414249a3f2b83367dbb6d773f946')
        self.assertEqual(data['missing_dependencies'], [])
        self.assertEqual(data['environment_counts']['modules'], 93)
        self.assertEqual(data['environment_counts']['visual_meshes'], 1346)
        self.assertEqual(data['environment_counts']['dynamic_rigid_bodies'], 110)
        self.assertEqual(data['environment_counts']['fixed_joints'], 47)
        dependency_paths = {entry['path'] for entry in data['dependency_files']}
        self.assertIn(str(SCENE), dependency_paths)
        self.assertTrue(any(path.endswith('/assets/supermarket/scene.usda') for path in dependency_paths))

    def test_production_supermarket_entries_use_updated_scene(self):
        for source in ('start_pico_test.sh', 'batch_capture.py', 'smoke_capture.py'):
            text = (ROOT / ('scripts' if source.endswith('.sh') else 'bridge') / source).read_text()
            self.assertIn('pico_supermarket_task1_7e5fd9d.usda', text)
            self.assertNotIn('pico_supermarket_basket_31f4a73.usda', text)



if __name__ == '__main__':
    unittest.main()
