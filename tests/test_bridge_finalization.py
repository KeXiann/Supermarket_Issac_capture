"""Exercise Bridge shutdown logic without initializing Isaac or DDS."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


def finalizer(camera_error=False, previous_error=''):
    tree = ast.parse(Path(__file__).resolve().parents[1].joinpath('bridge/sonic_isaacsim_bridge.py').read_text())
    function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'finish_episode_recording')
    class Recorder:
        samples = [1, 2]
        def __init__(self): self.writes = []
        def save(self, path, metadata): self.writes.append(dict(metadata))
    class Camera:
        frame_count = 2
        def finish_recording(self):
            if camera_error: raise OSError('disk full')
    world, body = Recorder(), Recorder()
    acknowledgements = []
    namespace = dict(ego_capture=Camera(), world_recorder=world, recorder=body,
        world_path='world.npz', world_metadata={}, args=SimpleNamespace(record='bridge.npz'),
        Path=Path, time=SimpleNamespace(monotonic=lambda: 1.), sim_time=1., stop_reason='normal',
        fault_reason='', atomic_json=lambda path, value: acknowledgements.append(value), initial_error=previous_error)
    wrapped = ('def create():\n'
               '    recording_finalized = False\n'
               '    recording_error = initial_error\n'
               '    def record_metadata():\n'
               '        return dict(recording_finalized=recording_finalized, recording_error=recording_error)\n'
               + '\n'.join('    '+line for line in ast.unparse(function).splitlines()) + '\n'
               '    return finish_episode_recording\n')
    exec(compile(wrapped, '<bridge-finalization-regression>', 'exec'), namespace)
    return namespace['create'](), world, body, acknowledgements


class BridgeFinalizationTests(unittest.TestCase):
    def test_normal_close_writes_finalized_metadata_and_ack(self):
        finish, world, body, acknowledgements = finalizer()
        finish()
        finish()
        self.assertTrue(world.writes[0]['recording_finalized'])
        self.assertTrue(body.writes[0]['recording_finalized'])
        self.assertEqual(len(body.writes), 1)
        self.assertTrue(acknowledgements[0]['recording_finalized'])

    def test_video_failure_still_saves_body_and_world_with_incomplete_markers(self):
        finish, world, body, acknowledgements = finalizer(camera_error=True)
        with self.assertRaisesRegex(RuntimeError, 'disk full'):
            finish()
        self.assertFalse(world.writes[0]['recording_finalized'])
        self.assertFalse(body.writes[0]['recording_finalized'])
        self.assertIn('disk full', body.writes[0]['recording_error'])
        self.assertEqual(acknowledgements, [])

    def test_prior_control_error_is_never_relabelled_complete(self):
        finish, world, body, acknowledgements = finalizer(previous_error='physics failed')
        with self.assertRaisesRegex(RuntimeError, 'physics failed'):
            finish()
        self.assertFalse(body.writes[0]['recording_finalized'])
        self.assertEqual(acknowledgements, [])


if __name__ == '__main__':
    unittest.main()
