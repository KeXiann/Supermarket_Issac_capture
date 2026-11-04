import json
from pathlib import Path
import tempfile
import threading
import unittest

import numpy as np

from bridge_recording import Recorder


class RecordingTests(unittest.TestCase):
    def test_checkpoint_does_not_block_and_final_save_includes_later_samples(self):
        entered, release = threading.Event(), threading.Event()
        class SlowRecorder(Recorder):
            def _write(self, out, samples, metadata):
                if metadata.get("checkpoint"):
                    entered.set()
                    if not release.wait(3):
                        raise TimeoutError("test writer was not released")
                super()._write(out, samples, metadata)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.npz"
            recorder = SlowRecorder()
            recorder.add(time=0., joint_pos=np.zeros(29))
            self.assertTrue(recorder.checkpoint(path, {"checkpoint": True}))
            self.assertTrue(entered.wait(1))
            # Control can append while the previous file's writer is blocked.
            recorder.add(time=.02, joint_pos=np.ones(29))
            self.assertFalse(recorder.checkpoint(path, {"checkpoint": True}))
            release.set()
            recorder.save(path, {"final": True})
            with np.load(path) as data:
                np.testing.assert_allclose(data["time"], [0., .02])
                self.assertEqual(data["joint_pos"].shape, (2, 29))
            self.assertEqual(json.loads(path.with_suffix(".json").read_text()), {"final": True})
            self.assertFalse(path.with_name(path.name + ".tmp").exists())

    def test_samples_own_their_input_arrays(self):
        recorder = Recorder()
        source = np.zeros(29)
        recorder.add(joint_pos=source)
        source[:] = 1
        np.testing.assert_array_equal(recorder.samples[0]["joint_pos"], np.zeros(29))
