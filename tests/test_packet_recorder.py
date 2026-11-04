import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

source = Path.home()/"repo/GR00T-WholeBodyControl/gear_sonic/utils/teleop/packet_recorder.py"
spec = importlib.util.spec_from_file_location("pico_packet_recorder", source)
recorder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recorder)


class PacketRecorderTests(unittest.TestCase):
    def test_collision_preserves_old_packet_and_close_reports_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pose_000000.npz'
            path.write_bytes(b'old capture must survive')
            r = recorder.PacketRecorder(directory)
            r.submit(0, {'sample': np.ones(1)})
            with self.assertRaisesRegex(RuntimeError, 'recording failed'):
                r.close()
            self.assertEqual(path.read_bytes(), b'old capture must survive')
            self.assertFalse(r.thread.is_alive())
            status = json.loads((Path(directory) / 'recording_status.json').read_text())
            self.assertEqual(status['written'], 0)
            self.assertTrue(status['error'])
            with self.assertRaisesRegex(RuntimeError, 'closed'):
                r.submit(1, {'sample': np.ones(1)})

    def test_slow_disk_does_not_delay_stream_and_packets_own_their_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            writing = threading.Event();release = threading.Event()
            original = np.savez_compressed
            def slow(*args, **kwargs):
                writing.set();release.wait(3)
                return original(*args, **kwargs)
            with patch.object(recorder.np, 'savez_compressed', slow):
                r = recorder.PacketRecorder(directory, capacity=1)
                a = np.array([1.])
                self.assertTrue(r.submit(0, {'sample':a}))
                self.assertTrue(writing.wait(2))
                start=time.monotonic()
                self.assertTrue(r.submit(1, {'sample':a}))
                self.assertFalse(r.submit(2, {'sample':a}))
                self.assertLess(time.monotonic()-start, .2)
                a[0]=99
                release.set();r.close()
            with np.load(Path(directory)/'pose_000001.npz') as d:
                np.testing.assert_array_equal(d['sample'],[1.])
            self.assertEqual(r.dropped,1)
            self.assertEqual(r.written,2)
            self.assertFalse(r.thread.is_alive())
