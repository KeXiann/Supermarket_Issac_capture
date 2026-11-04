import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from owned_process_cleanup import descendants, identity, terminate


class OwnedCleanupTests(unittest.TestCase):
    def test_separate_session_descendant_survives_parent_but_is_reaped(self):
        with tempfile.TemporaryDirectory() as tmp:
            ready = Path(tmp) / 'ready'
            worker = "import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)"
            parent = subprocess.Popen([sys.executable, '-c',
                'import subprocess,sys,time;from pathlib import Path;'
                'p=subprocess.Popen([sys.executable,"-c",sys.argv[2]],start_new_session=True);'
                'Path(sys.argv[1]).write_text(str(p.pid));time.sleep(60)', str(ready), worker],
                start_new_session=True)
            captured = {}
            try:
                deadline = time.monotonic() + 5
                while not ready.exists() and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertTrue(ready.exists())
                child = int(ready.read_text())
                captured = descendants([parent.pid])
                self.assertIn(child, captured)
                parent.terminate()
                parent.wait(timeout=3)
                self.assertEqual(terminate(captured, grace=.1), [])
                state = identity(child)
                self.assertTrue(state is None or state[2] == 'Z')
            finally:
                terminate(captured, grace=.1)
                if parent.poll() is None:
                    parent.kill()
                parent.wait(timeout=3)

    def test_different_start_time_is_never_signalled(self):
        pid = os.getpid()
        self.assertEqual(terminate({pid: 'wrong-start-time'}, grace=0), [])
