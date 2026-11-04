"""Batch boundaries, camera file rotation, and isolation without an Isaac runtime."""
import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from batch_runtime import (acknowledge_reset, command, new_episode, read_prepare,
                           process_start_ticks, request_batch_stop, reserve_episode_output)
from ego_camera import EgoCameraCapture
from batch_capture import BatchCollector
from capture_session import initialize, inspect


class BatchProtocolTests(unittest.TestCase):
    def test_task_named_episodes_are_independent_and_never_replace_an_old_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs = Path(tmp) / 'outputs'
            runtime = outputs / '.capture_work/.batch_fixture'
            runtime.mkdir(parents=True)
            runs = [new_episode(outputs, 'pick_up_the_basket', 'bridge-supermarket', runtime) for _ in range(2)]
            self.assertNotEqual(runs[0][0], runs[1][0])
            self.assertNotEqual(runs[0][1], runs[1][1])
            self.assertEqual([destination.name for _, destination in runs],
                             ['pick_up_the_basket_0', 'pick_up_the_basket_1'])
            for run, destination in runs:
                session = json.loads((run / 'session.json').read_text())
                self.assertEqual(session['task'], 'pick_up_the_basket')
                self.assertTrue(session['batch_bridge_segment'])
                self.assertEqual(session['mentor_only_output_directory'], str(destination))
                self.assertTrue(destination.name.startswith('pick_up_the_basket_'))
                self.assertFalse(destination.exists())

    def test_numbering_resumes_and_preserves_existing_timestamped_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs = Path(tmp)
            legacy = outputs / 'pick_up_the_basket_20261031_174946_9110fhy7'
            legacy.mkdir()
            (legacy / 'evidence').write_text('preserve old capture')
            (outputs / 'pick_up_the_basket_4').mkdir()
            self.assertEqual(reserve_episode_output(outputs, 'pick_up_the_basket').name,
                             'pick_up_the_basket_5')
            self.assertEqual(reserve_episode_output(outputs, 'pick_up_the_basket').name,
                             'pick_up_the_basket_6')
            self.assertEqual(reserve_episode_output(outputs, 'another_task').name, 'another_task_0')
            self.assertEqual((legacy / 'evidence').read_text(), 'preserve old capture')

    def test_concurrent_launchers_reserve_distinct_indices(self):
        from concurrent.futures import ThreadPoolExecutor
        with tempfile.TemporaryDirectory() as tmp, ThreadPoolExecutor(max_workers=4) as pool:
            destinations = list(pool.map(lambda _: reserve_episode_output(tmp, 'pick_up_the_basket'),
                                         range(16)))
            self.assertEqual({p.name for p in destinations},
                             {f'pick_up_the_basket_{i}' for i in range(16)})
            self.assertFalse(any(p.exists() for p in destinations))

    def test_corrupt_index_registry_fails_instead_of_recycling_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reserve_episode_output(root, 'pick_up_the_basket')
            counter = root / '.capture_work/.episode_indices.json'
            for value in ('{broken json', '[]', '{"pick_up_the_basket": -1}',
                          '{"pick_up_the_basket": true}'):
                counter.write_text(value)
                with self.assertRaises(ValueError):
                    reserve_episode_output(root, 'pick_up_the_basket')

    def test_reset_command_binds_new_run_to_the_same_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs = Path(tmp)
            runtime = outputs / '.capture_work/.batch_fixture'
            runtime.mkdir(parents=True)
            first, _ = new_episode(outputs, 'pick_up_the_basket', 'bridge-supermarket', runtime)
            second, _ = new_episode(outputs, 'pick_up_the_basket', 'bridge-supermarket', runtime)
            control = runtime / 'control.json'
            token = command(control, second)
            data = read_prepare(control, first)
            self.assertEqual(data['token'], token)
            acknowledge_reset(control, data, {'scene_body_count': 110})
            ack = json.loads((runtime / 'reset_ack.json').read_text())
            self.assertEqual(ack['token'], token)
            self.assertEqual(ack['bridge_pid'], os.getpid())
            self.assertEqual(ack['physics_steps_since_reset'], 0)
            self.assertIsNone(read_prepare(control, second))
            (second / 'bridge_saved.json').write_text('{}')
            with self.assertRaises(FileExistsError):
                read_prepare(control, first)

    def test_reset_rejects_another_runtime_and_paths_outside_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_runtime = root / '.capture_work/.batch_first'
            other_runtime = root / '.capture_work/.batch_other'
            first_runtime.mkdir(parents=True)
            other_runtime.mkdir()
            run, _ = new_episode(root, 'pick_up_the_basket', 'bridge-supermarket', other_runtime)
            control = first_runtime / 'control.json'
            command(control, run)
            with self.assertRaisesRegex(ValueError, 'another runtime'):
                read_prepare(control, root / 'previous')
            command(control, root / 'old_capture')
            with self.assertRaisesRegex(ValueError, 'outside'):
                read_prepare(control, root / 'previous')

    def test_stop_binds_pid_start_time_and_batch_executable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pid = os.getpid()
            data = {'owner_pid': pid, 'owner_start_ticks': process_start_ticks(pid),
                    'host_boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
            (root / 'batch.json').write_text(json.dumps(data))
            with patch('pathlib.Path.read_bytes', return_value=b'python\0/repo/batch_capture.py\0'), \
                 patch('batch_runtime.os.kill') as stop:
                self.assertEqual(request_batch_stop(root), pid)
                self.assertEqual(stop.call_count, 1)
            data['owner_start_ticks'] += 1
            (root / 'batch.json').write_text(json.dumps(data))
            with patch('batch_runtime.os.kill') as stop:
                with self.assertRaisesRegex(RuntimeError, 'no longer belongs'):
                    request_batch_stop(root)
                stop.assert_not_called()

    def test_persistent_bridge_requires_closed_segment_evidence_not_a_fake_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = initialize(root, profile='bridge-supermarket')
            session['batch_bridge_segment'] = True
            (root / 'session.json').write_text(json.dumps(session))
            (root / 'components').mkdir()
            for component in ('sonic', 'manager'):
                (root / f'components/{component}.json').write_text(json.dumps({
                    'status': 'exited', 'exit_code': 0, 'session_id': session['session_id']}))
            bridge = {'status': 'segment_closed', 'segment_close_code': 0,
                      'session_id': session['session_id'], 'pid': 12345}
            marker = {'bridge_pid': 12345, 'recording_finalized': True, 'physics_paused': True}
            (root / 'components/bridge.json').write_text(json.dumps(bridge))
            (root / 'bridge_segment_closed.json').write_text(json.dumps(marker))
            def closed_check():
                report, _ = inspect(root, verify_hashes=False)
                return next(c for c in report['checks'] if c['name'] == 'components_closed')
            self.assertTrue(closed_check()['passed'])
            marker['physics_paused'] = False
            (root / 'bridge_segment_closed.json').write_text(json.dumps(marker))
            self.assertFalse(closed_check()['passed'])
            marker['physics_paused'] = True
            (root / 'bridge_segment_closed.json').write_text(json.dumps(marker))
            bridge.pop('segment_close_code')
            bridge['exit_code'] = 0
            (root / 'components/bridge.json').write_text(json.dumps(bridge))
            self.assertFalse(closed_check()['passed'])


class CameraRotationTests(unittest.TestCase):
    def test_empty_warmup_frame_keeps_render_rate_bounded_without_saving_a_frame(self):
        from simulation_timing import FrameSchedule
        camera = EgoCameraCapture.__new__(EgoCameraCapture)
        camera.fps, camera._last_capture, camera._frame_count = 30, 0., 0
        camera._capture_schedule = FrameSchedule(30)
        camera.read_rgb = lambda: None
        from frame_output import FrameOutput
        camera._output = FrameOutput(camera._write_frame)
        with patch('ego_camera.time.monotonic', return_value=10.):
            self.assertFalse(camera.maybe_capture(0.))
        self.assertEqual(camera.frame_count, 0)
        self.assertFalse(camera.capture_due(10.005))
        self.assertTrue(camera.capture_due(10.034))
        camera._output.close()

    def test_recording_rotation_keeps_stream_and_has_independent_zero_based_frames(self):
        camera = EgoCameraCapture.__new__(EgoCameraCapture)
        camera.width, camera.height, camera.fps = 640, 360, 25
        camera._cv2 = cv2
        camera._writer = camera._timestamps = camera._rtp = None
        camera._zmq_context = object()
        stream = camera._zmq_context
        camera._zmq_socket = None
        camera.read_rgb = lambda: np.zeros((360, 640, 3), dtype=np.uint8)
        from frame_output import FrameOutput
        camera._output = FrameOutput(camera._write_frame)
        with tempfile.TemporaryDirectory() as tmp:
            for episode in range(2):
                path = Path(tmp) / f'episode_{episode}/ego_view.mp4'
                camera.start_recording(path)
                for index in range(3):
                    camera._last_capture = 0.
                    self.assertTrue(camera.maybe_capture(index * .04))
                camera.finish_recording()
                camera.finish_recording()  # durable close remains idempotent
                self.assertIs(camera._zmq_context, stream)
                status = json.loads(path.with_suffix('.recording.json').read_text())
                self.assertEqual(status['frames'], 3)
                with path.with_suffix('.timestamps.csv').open() as source:
                    self.assertEqual([int(row['frame']) for row in csv.DictReader(source)], [0, 1, 2])
                cap = cv2.VideoCapture(str(path))
                count = 0
                while cap.read()[0]:
                    count += 1
                cap.release()
                self.assertEqual(count, 3)
                with self.assertRaises(FileExistsError):
                    camera.start_recording(path)
        camera._output.close()


class BatchShutdownTests(unittest.TestCase):
    def test_manager_startup_notice_does_not_claim_recording_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            collector = BatchCollector.__new__(BatchCollector)
            collector.runtime = collector.run = Path(tmp)
            collector.bridge_notice_offset = collector.manager_log_offset = 0
            (Path(tmp) / 'manager.log').write_text('[REC] READY: buttons installed\n')
            with patch('builtins.print') as output:
                collector.pump_manager_notices()
                collector.pump_manager_notices()
            self.assertEqual(output.call_count, 1)
            message = output.call_args.args[0]
            self.assertIn('waiting for ABXY', message)
            self.assertNotIn('READY:', message)

    def test_perf_notices_reach_batch_terminal_once_even_with_partial_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            collector = BatchCollector.__new__(BatchCollector)
            collector.runtime, collector.run = Path(tmp), None
            collector.bridge_notice_offset = 0
            collector.runtime_log_offset = 123
            path = collector.runtime / 'bridge.log'
            path.write_text('[BRIDGE] routine heartbeat\n[PERF] RE')
            with patch('builtins.print') as output:
                collector.pump_manager_notices()
                output.assert_not_called()
                with path.open('a') as stream:
                    stream.write('ADY: physics=200\n')
                collector.pump_manager_notices()
                collector.pump_manager_notices()
                output.assert_called_once_with('[PERF] READY: physics=200', flush=True)
            self.assertEqual(collector.runtime_log_offset, 123)

    def test_native_close_without_episode_marker_terminates_owned_bridge(self):
        import subprocess
        import signal
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as tmp:
            collector = BatchCollector.__new__(BatchCollector)
            collector.shutdown_done = False
            bridge = Mock(pid=1003)
            bridge.poll.return_value = None
            def wait(timeout):
                if timeout == 30:
                    raise subprocess.TimeoutExpired('bridge', 30)
                bridge.poll.return_value = 0
                return 0
            bridge.wait.side_effect = wait
            collector.children = {'bridge': bridge}
            collector.run = collector.runtime = Path(tmp)
            collector.closed_runs = set()
            collector.results = []
            collector.log = lambda _: None
            (Path(tmp) / 'bridge.log').write_text('[STOP] Closing Isaac Sim\n')
            with patch('batch_capture.os.killpg') as stop:
                collector.shutdown()
            self.assertEqual([call.args for call in stop.call_args_list],
                             [(1003, signal.SIGINT), (1003, signal.SIGTERM)])
            self.assertTrue(collector.run.exists())

    def test_normal_stop_waits_for_sonic_cleanup_without_a_second_signal(self):
        class Child:
            def __init__(self, pid, code=None):
                self.pid, self.returncode = pid, code
                self.waits = []
            def poll(self):
                return self.returncode
            def wait(self, timeout):
                self.waits.append(timeout)
                self.returncode = 0
                return 0
        with tempfile.TemporaryDirectory() as tmp:
            collector = BatchCollector.__new__(BatchCollector)
            collector.shutdown_done = False
            collector.children = {'manager': Child(1001), 'sonic': Child(1002), 'bridge': Child(1003, 0)}
            collector.run = Path(tmp)
            collector.runtime = Path(tmp)
            collector.closed_runs = set()
            collector.results = []
            collector.log = lambda _: None
            with patch('batch_capture.os.killpg') as stop:
                collector.shutdown()
                collector.shutdown()
            self.assertEqual(stop.call_count, 1)
            self.assertEqual(stop.call_args.args[0], 1001)
            self.assertEqual(collector.children['sonic'].waits, [15])


if __name__ == '__main__':
    unittest.main()
