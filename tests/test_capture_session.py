"""Exercise real supervisor signals and its normal child-finally drain path."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest

from unittest.mock import patch

import numpy as np

from capture_session import (FINALIZATION_JOURNAL, FinalizationProgress, finalize,
                             initialize, _publish_mentor_only_output)

ROOT = Path(__file__).resolve().parents[1]
CHILD = """
import os,sys,time
from pathlib import Path
p=Path(sys.argv[1])
try:
    (p/'child_ready').touch()
    while True: time.sleep(.05)
except KeyboardInterrupt:
    pass
finally:
    with (p/'drained').open('x') as f:
        f.write('last accepted packet saved');f.flush();os.fsync(f.fileno())
"""


class SceneProfileTests(unittest.TestCase):
    def test_authored_scenes_require_complete_capture(self):
        for profile in ('bridge-livingroom', 'bridge-supermarket'):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as tmp:
                session = initialize(Path(tmp), profile=profile)
                self.assertEqual(session['control_mode'], 'SONIC_POSE')
                self.assertEqual(session['manager_profile'], 'manager')
                for field in ('require_hands', 'require_camera', 'require_sonic_debug', 'publish_mentor_sample'):
                    self.assertTrue(session[field])

    def test_successful_formal_publish_removes_work_and_keeps_only_mentor_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'outputs'
            work = root / '.capture_work' / 'pick_up_the_basket_ab12cd'
            destination = root / 'pick_up_the_basket_20261031_ab12cd'
            work.mkdir(parents=True)
            session = initialize(work, profile='bridge-supermarket', task='pick_up_the_basket',
                                 mentor_only_output=destination, publication_root=root)
            self.assertEqual(session['task'], 'pick_up_the_basket')
            for name in ('data', 'meta', 'videos'):
                (destination / name).mkdir(parents=True, exist_ok=True)
            (destination / 'meta/info.json').write_text(json.dumps({
                'script_config': {'submission': {'source_session_id': session['session_id']}}}))
            result = {'status': 'complete', 'mentor_submission':
                      {'status': 'complete', 'directory': str(destination)}}
            with patch('prepare_mentor_sample.validate_submission', return_value={
                    'status': 'complete', 'mentor_file_layout_match': True}):
                published = _publish_mentor_only_output(work, result)
            self.assertEqual(published, result)
            self.assertFalse(work.exists())
            self.assertEqual({path.name for path in destination.iterdir()}, {'data', 'meta', 'videos'})
            self.assertEqual((root / 'latest_pico_run.txt').read_text().strip(), str(destination))
            self.assertEqual((root / 'latest_mentor_sample.txt').read_text().strip(), str(destination))

    def test_incomplete_formal_capture_retains_recoverable_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'outputs'
            work = root / '.capture_work' / 'pick_up_the_basket_ab12cd'
            destination = root / 'pick_up_the_basket_20261031_ab12cd'
            work.mkdir(parents=True)
            initialize(work, profile='bridge-supermarket', task='pick_up_the_basket',
                       mentor_only_output=destination, publication_root=root)
            result = {'status': 'incomplete'}
            self.assertEqual(_publish_mentor_only_output(work, result), result)
            self.assertTrue(work.exists())
            self.assertFalse(destination.exists())

    def test_failed_or_foreign_submission_never_removes_capture_work(self):
        for foreign in (False, True):
            with self.subTest(foreign=foreign), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / 'outputs'
                work = root / '.capture_work' / 'pick_up_the_basket_ab12cd'
                destination = root / 'pick_up_the_basket_20261031_ab12cd'
                work.mkdir(parents=True)
                session = initialize(work, mentor_only_output=destination, publication_root=root)
                (destination / 'meta').mkdir(parents=True)
                (destination / 'meta/info.json').write_text(json.dumps({'script_config': {'submission': {
                    'source_session_id': 'foreign' if foreign else session['session_id']}}}))
                result = {'status': 'complete', 'mentor_submission':
                          {'status': 'complete', 'directory': str(destination)}}
                with patch('prepare_mentor_sample.validate_submission', return_value={
                        'status': 'complete', 'mentor_file_layout_match': foreign}):
                    with self.assertRaises(RuntimeError):
                        _publish_mentor_only_output(work, result)
                self.assertTrue((work / 'session.json').is_file())
                self.assertFalse((root / 'latest_mentor_sample.txt').exists())

    def test_concurrent_finalizers_do_not_recreate_deleted_raw_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'outputs'
            work = root / '.capture_work' / 'pick_up_the_basket_ab12cd'
            destination = root / 'pick_up_the_basket_20261031_ab12cd'
            work.mkdir(parents=True)
            session = initialize(work, mentor_only_output=destination, publication_root=root)
            (destination / 'meta').mkdir(parents=True)
            (destination / 'meta/info.json').write_text(json.dumps({'script_config': {'submission': {
                'source_session_id': session['session_id']}}}))
            result = {'status': 'complete', 'mentor_submission':
                      {'status': 'complete', 'directory': str(destination)}}
            entered, release, second_snapshot = threading.Event(), threading.Event(), threading.Event()
            second_ident = []
            from capture_session import read_json
            def read_with_snapshot(path):
                value = read_json(path)
                if (Path(path).name == 'session.json' and second_ident
                        and threading.get_ident() == second_ident[0]):
                    second_snapshot.set()
                return value
            def finish_artifacts(run, progress):
                # A late deferred signal must not restore the deleted journal.
                progress.defer(signal.SIGTERM, None)
                entered.set()
                self.assertTrue(release.wait(5))
                return result
            with patch('capture_session._finalize_artifacts', side_effect=finish_artifacts), \
                 patch('capture_session.read_json', side_effect=read_with_snapshot), \
                 patch('prepare_mentor_sample.validate_submission', return_value={
                     'status': 'complete', 'mentor_file_layout_match': True}):
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=2) as pool:
                    first = pool.submit(finalize, work)
                    self.assertTrue(entered.wait(5))
                    def second_finalizer():
                        second_ident.append(threading.get_ident())
                        return finalize(work)
                    second = pool.submit(second_finalizer)
                    self.assertTrue(second_snapshot.wait(5))
                    release.set()
                    self.assertEqual(first.result(timeout=5)['status'], 'complete')
                    self.assertEqual(second.result(timeout=5)['status'], 'complete')
            self.assertFalse(work.exists())

    def test_mentor_only_rejects_raw_path_outside_isolated_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, 'isolated'):
                initialize(root / 'old_capture', mentor_only_output=root / 'sample', publication_root=root)

    def test_diagnostic_submission_preserves_latest_pointer(self):
        from prepare_mentor_sample import ensure_submission
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            run = parent / 'diagnostic'
            initialize(run, profile='bridge-supermarket', input_kind='recorded_pose_smoke')
            pointer = parent / 'latest_mentor_sample.txt'
            original = b'/previous/real/capture_mentor_sample\n'
            pointer.write_bytes(original)
            with patch('prepare_mentor_sample.prepare', return_value={'status': 'complete'}):
                result = ensure_submission(run, update_latest=False)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(pointer.read_bytes(), original)

    def test_diagnostic_finalize_disables_latest_pointer_update(self):
        from capture_session import publish_mentor_submission
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            initialize(run, profile='bridge-supermarket', input_kind='recorded_pose_smoke')
            sample = {'status': 'complete', 'directory': str(run)+'_mentor_sample'}
            with patch('prepare_mentor_sample.ensure_submission', return_value=sample) as publish:
                result = publish_mentor_submission(run, {'status': 'complete'})
            self.assertEqual(result['mentor_submission'], sample)
            publish.assert_called_once_with(run, update_latest=False)


class SupervisorSignalTests(unittest.TestCase):
    def exercise(self, use_stop_command):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            initialize(run)
            process = subprocess.Popen([sys.executable, str(ROOT / "bridge/capture_session.py"),
                'run-component', '--run', str(run), '--component', 'manager', '--',
                sys.executable, '-c', CHILD, str(run)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            try:
                deadline = time.monotonic() + 5
                while not (run/'child_ready').exists():
                    if process.poll() is not None or time.monotonic() > deadline:
                        self.fail('supervised child did not start')
                    time.sleep(.01)
                if use_stop_command:
                    subprocess.run([sys.executable, str(ROOT / "bridge/capture_session.py"), 'stop', str(run)],
                                   check=True, stdout=subprocess.DEVNULL, timeout=5)
                else:
                    os.kill(process.pid, signal.SIGTERM)
                _, error = process.communicate(timeout=5)
                self.assertEqual(process.returncode, 0, error.decode())
                self.assertEqual((run/'drained').read_text(), 'last accepted packet saved')
                state = json.loads((run/'components/manager.json').read_text())
                self.assertEqual(state['status'], 'exited')
                self.assertEqual(state['exit_code'], 0)
                self.assertFalse((run/'completion.json').exists())
                self.assertFalse((run/FINALIZATION_JOURNAL).exists())
            finally:
                if process.poll() is None:
                    process.kill()
                process.communicate(timeout=5)

    def test_stop_cli_delivers_interrupt_and_waits_for_child_drain(self):
        self.exercise(True)

    def test_sigterm_is_forwarded_as_interrupt_and_waits_for_child_drain(self):
        self.exercise(False)


EXPORT_CHILD = """
import json, os, sys, time
from pathlib import Path
import cv2
import numpy as np
import export_capture_dataset as exporter
from capture_session import FinalizationProgress
run = Path(sys.argv[1])
source = run / 'source.mp4'
writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), 25, (640, 360))
assert writer.isOpened()
for i in range(6):
    writer.write(np.full((360, 640, 3), i * 30, dtype=np.uint8))
writer.release()
real_popen = exporter.subprocess.Popen
def pause_after_encoder_start(*args, **kwargs):
    child = real_popen(*args, **kwargs)
    (run / 'encoder_ready').write_text(json.dumps({'pid': child.pid, 'pgid': os.getpgid(child.pid)}))
    deadline = time.monotonic() + 10
    while not (run / 'release').exists():
        if time.monotonic() > deadline:
            child.kill()
            child.wait()
            raise TimeoutError('test did not release encoder')
        time.sleep(.01)
    return child
exporter.subprocess.Popen = pause_after_encoder_start
with FinalizationProgress(run) as progress:
    progress.stage('exporting_standard_data_metadata_video')
    exporter.encode_video(source, np.repeat(np.arange(6), 2), run / 'result.mp4')
    progress.finish({'status': 'complete'})
"""


class FinalizationTests(unittest.TestCase):
    def test_closed_output_pipe_keeps_completion_and_success_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            code = ("from capture_session import FinalizationProgress; import sys\n"
                    "with FinalizationProgress(sys.argv[1]) as progress:\n"
                    " progress.stage('exporting_without_terminal')\n"
                    " progress.finish({'status':'complete'})\n")
            process = subprocess.Popen([sys.executable, '-c', code, tmp], cwd=ROOT,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            process.stdout.close()
            try:
                process.wait(timeout=5)
                error = process.stderr.read().decode()
                self.assertEqual(process.returncode, 0, error)
                self.assertEqual(json.loads((Path(tmp) / FINALIZATION_JOURNAL).read_text())['status'],
                                 'complete')
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
                process.stderr.close()

    def test_terminal_group_signals_allow_real_video_to_finish(self):
        """Terminal signals hit the exporter while a real FFmpeg child is live."""
        import cv2
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            process = subprocess.Popen([sys.executable, '-c', EXPORT_CHILD, str(run)],
                                       cwd=ROOT, start_new_session=True,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            try:
                deadline = time.monotonic() + 10
                while not (run / 'encoder_ready').exists():
                    if process.poll() is not None or time.monotonic() > deadline:
                        self.fail('exporter did not start its encoder')
                    time.sleep(.01)
                encoder = json.loads((run / 'encoder_ready').read_text())
                self.assertNotEqual(encoder['pgid'], process.pid)
                signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
                for sig in signals:
                    os.killpg(process.pid, sig)
                    time.sleep(.03)
                    self.assertIsNone(process.poll(), f'export interrupted by {sig.name}')
                (run / 'release').touch()
                _, error = process.communicate(timeout=10)
                self.assertEqual(process.returncode, 0, error.decode())
                journal = json.loads((run / FINALIZATION_JOURNAL).read_text())
                self.assertEqual(journal['status'], 'complete')
                self.assertEqual([x['signal'] for x in journal['deferred_signals']],
                                 [sig.name for sig in signals])
                video = cv2.VideoCapture(str(run / 'result.mp4'))
                count = 0
                try:
                    while video.read()[0]:
                        count += 1
                finally:
                    video.release()
                self.assertEqual(count, 12)
            finally:
                (run / 'release').touch()
                if process.poll() is None:
                    process.kill()
                process.communicate(timeout=10)

    def test_failed_export_preserves_phase_error_and_restores_signal_handlers(self):
        previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, 'encoder failure'):
                with FinalizationProgress(run) as progress:
                    progress.stage('exporting_standard_data_metadata_video')
                    raise RuntimeError('encoder failure')
            journal = json.loads((run / FINALIZATION_JOURNAL).read_text())
            self.assertEqual(journal['status'], 'failed')
            self.assertEqual(journal['stage'], 'exporting_standard_data_metadata_video')
            self.assertIn('encoder failure', journal['traceback'])
        for sig, handler in previous.items():
            self.assertEqual(signal.getsignal(sig), handler)

    def test_revalidation_keeps_artifacts_and_mutable_journal_outside_checksums(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            initialize(run, profile='bridge')
            (run / 'pico_replay_archive.npz').write_bytes(b'preserved replay fixture')
            (run / 'prepared').mkdir()
            (run / 'prepared/quality_report.json').write_text('{"status":"development_checks_pass"}')
            report = {'status': 'complete', 'checks': []}
            robot = {'state_monotonic_ns': np.array([1000000000, 1020000000]),
                     'time': np.array([0., .02])}
            detail = {'human': {'packet_time': np.array([1., 1.02])},
                      'robot': ({}, robot, None, None), 'modes': np.array([2, 2])}
            with patch('capture_session.inspect', return_value=(report, detail)):
                self.assertEqual(finalize(run)['status'], 'complete')
            before = (run / 'completion.json').read_bytes()
            checksum_bytes = (run / 'checksums.json').read_bytes()
            entries = json.loads(checksum_bytes)['files']
            self.assertNotIn(FINALIZATION_JOURNAL, [x['relative_path'] for x in entries])
            from capture_manifest import fingerprint
            for entry in entries:
                self.assertEqual(fingerprint(run / entry['relative_path'])['sha256'], entry['sha256'])
            attempt = json.loads((run / FINALIZATION_JOURNAL).read_text())['attempt_id']
            with patch('capture_session.inspect', return_value=(report, {})):
                self.assertEqual(finalize(run)['status'], 'complete')
            self.assertNotEqual(json.loads((run / FINALIZATION_JOURNAL).read_text())['attempt_id'], attempt)
            self.assertEqual((run / 'checksums.json').read_bytes(), checksum_bytes)
            self.assertEqual((run / 'completion.json').read_bytes(), before)
