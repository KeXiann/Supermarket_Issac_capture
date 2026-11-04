from project_paths import SONIC_ROOT
#!/usr/bin/env python3
"""Batch PICO collection: one Isaac instance, independently finalized episodes.

The free-base policy is disconnected only after physics stop ACK. Between
episodes the simulator restores the initial robot and all scene rigid bodies;
SONIC and Manager are automatically relaunched with fresh histories/recorders.
"""
import argparse
import fcntl
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import time

from batch_runtime import command, new_episode, task_slug, process_start_ticks
from capture_manifest import capture
from capture_session import _write_text_pointer, finalize, read_json
from capture_utils import atomic_json

ROOT = Path(__file__).resolve().parents[1]
PROJECT = SONIC_ROOT
DEPLOY = PROJECT / 'gear_sonic_deploy'


def existing_control_processes():
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            arguments = (proc / 'cmdline').read_bytes().split(b'\0')
            executable = Path(os.fsdecode(arguments[0])).name
            if not (executable.startswith('python') or executable == 'g1_deploy_onnx_ref'):
                continue
            names = {Path(os.fsdecode(arg)).name for arg in arguments if arg}
            if names & {'sonic_isaacsim_bridge.py', 'g1_deploy_onnx_ref', 'pico_manager_thread_server.py'}:
                yield int(proc.name)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue


class BatchCollector:
    def __init__(self, args):
        camera_overrides = [key for key in ('PICO_EGO_PITCH_DOWN_DEG', 'PICO_EGO_FOCAL_LENGTH_MM',
                            'PICO_EGO_MOUNT_X_M', 'PICO_EGO_MOUNT_Y_M', 'PICO_EGO_MOUNT_Z_M') if key in os.environ]
        if camera_overrides:
            raise ValueError('Formal collection uses pinned upstream cameras; unset ' + ', '.join(camera_overrides))
        self.args = args
        self.outputs = ROOT / 'outputs' / 'collected_data' if not args.smoke_source else ROOT / 'outputs'
        work = self.outputs / '.capture_work'
        work.mkdir(parents=True, exist_ok=True)
        self.runtime = Path(tempfile.mkdtemp(prefix='.batch_' + task_slug(args.task) + '_', dir=work))
        self.control = self.runtime / 'control.json'
        self.scene = ROOT / 'assets/scenes/pico_supermarket_task1_7e5fd9d.usda'
        self.children = {}
        self.stop_requested = False
        self.results = []
        self.runtime_log_offset = 0
        self.bridge_notice_offset = 0
        self.manager_log_offset = 0
        self.shutdown_done = False
        self.shutdown_error = None
        self.closed_runs = set()
        self.run = None
        self.input_kind = 'recorded_pose_smoke' if args.smoke_source else 'live_pico'
        atomic_json(self.runtime / 'batch.json', {'owner_pid': os.getpid(), 'task': args.task,
                    'owner_start_ticks': process_start_ticks(os.getpid()),
                    'host_boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                    'scene': str(self.scene), 'started_utc': datetime.now().astimezone().isoformat(),
                    'input_kind': self.input_kind, 'results': []})

    def launch(self, name, argv, cwd, log):
        env = os.environ.copy()
        env.update(OMNI_KIT_ACCEPT_EULA='YES', PYTHONUNBUFFERED='1', PICO_RUN_DIR=str(self.run))
        env['LD_LIBRARY_PATH'] = ':'.join([str(Path.home() / 'TensorRT/lib'), '/usr/local/cuda-13.0/lib64',
                                         '/opt/onnxruntime/lib', env.get('LD_LIBRARY_PATH', '')])
        with Path(log).open('xb') as stream:
            child = subprocess.Popen(argv, cwd=cwd, env=env, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        self.children[name] = child
        return child

    def wait(self, predicate, timeout, label, watch=()):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.pump_manager_notices()
            if predicate():
                return
            if self.stop_requested:
                raise KeyboardInterrupt
            for name in watch:
                if self.children[name].poll() is not None:
                    raise RuntimeError(f'{name} ended before {label}; inspect {self.runtime}')
            time.sleep(.1)
        raise TimeoutError(f'timed out waiting for {label}; recoverable work: {self.run}')

    def pump_manager_notices(self):
        # The Bridge process is persistent; forward its timing notices from the
        # private runtime log without disturbing per-episode hash/copy offsets.
        bridge_log = self.runtime / 'bridge.log'
        if bridge_log.exists():
            with bridge_log.open(errors='replace') as stream:
                stream.seek(self.bridge_notice_offset)
                while True:
                    position = stream.tell()
                    line = stream.readline()
                    if not line.endswith('\n'):
                        stream.seek(position)
                        break
                    if line.startswith('[PERF]'):
                        print(line.rstrip(), flush=True)
                self.bridge_notice_offset = stream.tell()
        if self.run is None or not (self.run / 'manager.log').exists():
            return
        with (self.run / 'manager.log').open(errors='replace') as stream:
            stream.seek(self.manager_log_offset)
            for line in stream:
                if '[REC] READY:' in line:
                    print('[BATCH] INPUT_CONNECTED: waiting for ABXY calibration, then A+X POSE; recording is not ready yet.', flush=True)
                elif '[REC]' in line or '[Manager] StreamMode switch' in line:
                    print(line.rstrip(), flush=True)
            self.manager_log_offset = stream.tell()

    def new_run(self):
        self.manager_log_offset = 0
        self.run, destination = new_episode(self.outputs, self.args.task, f'bridge-{self.args.scene}',
                                            self.runtime, input_kind=self.input_kind)
        if self.args.smoke_source:
            session = read_json(self.run / 'session.json')
            # Exercise Mentor-only lifecycle with explicit diagnostic provenance;
            # this test never updates the repository's real latest pointers.
            session.update(mentor_only_output_directory=str(destination), publication_root=str(self.outputs),
                           diagnostic_preserve_latest=True)
            atomic_json(self.run / 'session.json', session)
        else:
            _write_text_pointer(self.outputs / 'latest_pico_run.txt', self.run)
        (self.run / 'scene_path.txt').write_text(str(self.scene) + '\n')
        (self.run / 'hand_profile.txt').write_text('revo2\n')
        capture(self.run, 'bridge', scene_file=self.scene,
                parameters={'batch_runtime': str(self.runtime), 'automatic_reset': True})
        self.log(f'EPISODE {destination.name.rsplit("_", 1)[-1]}: {destination.name}')
        return destination

    def log(self, message):
        print('[BATCH] ' + message, flush=True)

    def start_bridge(self):
        self.log('STARTING_SCENE: loading Isaac Sim and task assets ...')
        argv = [sys.executable, str(ROOT / "bridge/sonic_isaacsim_bridge.py"), '--device', 'cpu',
                '--mode', 'drive', '--scene', str(self.scene), '--dexterous-hands', '--ego-camera',
                '--ego-camera-height', '360', '--ego-camera-record', str(self.run / 'ego_view.mp4'),
                '--ego-camera-zmq-port', '0' if self.args.smoke_source else '5555',
                '--target-policy', 'sonic-pd', '--pd-backend', 'external', '--pd-target-slew', 'none',
                '--kp-scale', '1', '--kd-scale', '1', '--torque-scale', '1', '--watchdog', '.3',
                '--wait-for-policy-log', str(self.run / 'sonic.log'), '--startup-timeout', '3600',
                '--render-every', '8', '--duration', '0', '--record', str(self.run / 'bridge.npz'),
                '--checkpoint-seconds', '5', '--stop-request-file', str(self.run / 'simulation_stop.json'),
                '--batch-control-file', str(self.control)]
        if self.args.headless:
            argv.append('--headless')
        from bridge_runtime import NATIVE_LAB
        lab = NATIVE_LAB
        self.launch('bridge', argv, lab, self.runtime / 'bridge.log')
        self.wait(lambda: (self.runtime / 'bridge.log').exists() and '[INFO] started:' in
                  (self.runtime / 'bridge.log').read_text(errors='replace'), 300, 'Isaac scene ready', ('bridge',))

    def bind_bridge(self):
        child = self.children['bridge']
        (self.run / 'components').mkdir(exist_ok=True)
        atomic_json(self.run / 'components/bridge.json', {
            'session_id': read_json(self.run / 'session.json')['session_id'], 'component': 'bridge',
            'status': 'running', 'pid': child.pid, 'supervisor_pid': os.getpid(),
            'batch_runtime_directory': str(self.runtime), 'command': child.args})

    def start_control(self):
        self.log('STARTING_CONTROL: waiting for SONIC initialization ...')
        if not self.args.smoke_source:
            self.launch('sonic', ['bash', str(ROOT / "scripts/start_pico_test.sh"), 'sonic'], ROOT,
                        self.run / 'supervisor_sonic.log')
        else:
            capture(self.run, 'sonic', pose_file=self.args.smoke_source)
            sonic = [str(DEPLOY / 'target/release/g1_deploy_onnx_ref'), 'lo', 'policy/release/model_decoder.onnx',
                     'reference/example/', '--obs-config', 'policy/release/observation_config.yaml',
                     '--encoder-file', 'policy/release/model_encoder.onnx', '--planner-file',
                     'planner/target_vel/V2/planner_sonic.onnx', '--disable-crc-check', '--input-type', 'zmq_manager',
                     '--zmq-port', str(self.args.port), '--output-type', 'zmq', '--zmq-out-port', '5557',
                     '--target-motion-logfile', str(self.run / 'sonic_target.csv'), '--planner-motion-logfile',
                     str(self.run / 'sonic_planner.csv'), '--enable-csv-logs', '--logs-dir', str(self.run / 'sonic_state')]
            self.launch('sonic', [sys.executable, str(ROOT / "bridge/capture_session.py"), 'run-component',
                        '--run', str(self.run), '--component', 'sonic', '--', sys.executable,
                        str(ROOT / "bridge/run_sonic_with_debug.py"), '--run', str(self.run), '--', *sonic],
                        DEPLOY, self.run / 'supervisor_sonic.log')
        self.wait(lambda: (self.run / 'sonic.log').exists() and 'Init Done' in
                  (self.run / 'sonic.log').read_text(errors='replace'), 180, 'SONIC ready', ('bridge', 'sonic'))
        if not self.args.smoke_source:
            self.launch('manager', ['bash', str(ROOT / "scripts/start_pico_test.sh"), 'manager'], ROOT,
                        self.run / 'supervisor_manager.log')
        else:
            capture(self.run, 'manager', pose_file=self.args.smoke_source)
            self.launch('manager', [sys.executable, str(ROOT / "bridge/capture_session.py"), 'run-component',
                        '--run', str(self.run), '--component', 'manager', '--', sys.executable,
                        str(ROOT / "bridge/smoke_capture.py"), '--driver', str(self.run), '--source', str(self.args.smoke_source),
                        '--pose-seconds', str(self.args.pose_seconds), '--port', str(self.args.port), '--manual-recording'],
                        ROOT, self.run / 'supervisor_manager.log')
        if not self.args.smoke_source:
            self.log('WAITING_FOR_PICO: connect XRoboToolkit and enable body tracking; waiting for body data ...')
            self.wait(lambda: (self.run / 'manager.log').exists() and '[REC] READY:' in
                      (self.run / 'manager.log').read_text(errors='replace'), 3600,
                      'PICO body data (connect XRoboToolkit and enable body tracking)',
                      ('bridge', 'sonic', 'manager'))
        self.log('WAITING_FOR_CALIBRATION: calibrate ABXY, enter POSE with A+X; left grip+A starts/stops. Left grip+B discards.')
        self.log('Robot and basket reset automatically after saving; keep a neutral ready pose before the next calibration.')
        self.log('After entering POSE, wait for [PERF] READY before starting a formal recording.')

    def close_episode(self):
        if str(self.run) in self.closed_runs:
            return
        self.wait(lambda: self.children['manager'].poll() is not None, 24 * 3600,
                  'episode stop', ('bridge',))
        self.wait(lambda: self.children['sonic'].poll() is not None, 30, 'SONIC drain', ('bridge',))
        self.wait(lambda: (self.run / 'bridge_segment_closed.json').is_file(), 60, 'Bridge segment flush', ('bridge',))
        # The persistent process log stays private to the batch. Copy a closed
        # byte range for this episode before finalization hashes raw evidence.
        with (self.runtime / 'bridge.log').open('rb') as stream:
            stream.seek(self.runtime_log_offset)
            data = stream.read()
            self.runtime_log_offset = stream.tell()
        (self.run / 'bridge.log').write_bytes(data)
        state = read_json(self.run / 'components/bridge.json')
        state.update(status='segment_closed', segment_close_code=0,
                     segment=read_json(self.run / 'bridge_segment_closed.json'))
        atomic_json(self.run / 'components/bridge.json', state)
        if any(self.children[name].returncode != 0 for name in ('manager', 'sonic')):
            raise RuntimeError(f'control process failed; original data retained: {self.run}')
        self.log('SAVING: checking closed files and publishing Mentor dataset ...')
        result = finalize(self.run)
        if result.get('deferred_terminal_signals'):
            self.stop_requested = True
        self.closed_runs.add(str(self.run))
        self.results.append({'run': str(self.run), 'result': result})
        summary = read_json(self.runtime / 'batch.json')
        summary['results'] = self.results
        atomic_json(self.runtime / 'batch.json', summary)
        if result['status'] not in ('complete', 'discarded') or result.get('mentor_submission', {}).get('status') == 'incomplete':
            raise RuntimeError(f'export incomplete; stopped before next episode: {self.run}')
        if result['status'] == 'discarded':
            self.log(f'DISCARDED: raw diagnostics preserved at {self.run}')
        else:
            self.log('SAVED: ' + result['mentor_submission']['directory'])

    def collect(self):
        self.new_run()
        self.start_bridge()
        self.bind_bridge()
        try:
            while not self.stop_requested:
                self.start_control()
                self.close_episode()
                if self.stop_requested or (self.args.episodes and len(self.results) >= self.args.episodes):
                    break
                self.new_run()
                token = command(self.control, self.run)
                self.wait(lambda: (self.runtime / 'reset_ack.json').exists() and
                          read_json(self.runtime / 'reset_ack.json').get('token') == token,
                          60, 'verified robot/basket reset', ('bridge',))
                reset = read_json(self.runtime / 'reset_ack.json')
                if not reset.get('reset_verified'):
                    raise RuntimeError('reset was not verified')
                atomic_json(self.run / 'batch_reset.json', reset)
                self.bind_bridge()
                self.log(f'RESET VERIFIED: same Isaac PID {reset["bridge_pid"]}; initial robot/hands/scene restored.')
        finally:
            self.shutdown()

    def shutdown(self):
        if self.shutdown_done:
            return
        self.shutdown_done = True
        from owned_process_cleanup import descendants, terminate
        owned = descendants([child.pid for child in self.children.values() if child.poll() is None])
        bridge = self.children.get('bridge')
        bridge_owned = descendants([bridge.pid]) if bridge is not None and bridge.poll() is None else {}
        try:
            self._shutdown()
        finally:
            # Supervisors launch children in separate sessions. Capture their
            # identities before teardown so reparenting cannot hide them.
            owned.update(descendants([child.pid for child in self.children.values() if child.poll() is None]))
            remaining = terminate(bridge_owned)
            remaining += terminate({pid: ticks for pid, ticks in owned.items() if pid not in bridge_owned})
            for child in self.children.values():
                if child.poll() is None:
                    try:
                        child.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        remaining.append(child.pid)
            if remaining:
                self.shutdown_error = f'Cleanup failed for owned processes: {sorted(set(remaining))}'
                self.log(self.shutdown_error)
            else:
                self.log('EXITED: all owned control processes stopped.')

    def _shutdown(self):
        # Manager must request physics ACK before SONIC disconnects. Never
        # broad-kill matching names; only terminate this collector's children.
        for name in ('manager', 'sonic'):
            child = self.children.get(name)
            if child is not None and child.poll() is None:
                try:
                    if name == 'sonic':
                        # Manager already sent the normal policy stop after
                        # physics ACK. An immediate second SIGINT races native
                        # cleanup and turns its clean exit into -SIGINT/254.
                        try:
                            child.wait(timeout=15)
                            continue
                        except subprocess.TimeoutExpired:
                            bridge = self.children.get('bridge')
                            if (bridge is not None and bridge.poll() is None and self.run.exists()
                                    and not (self.run / 'bridge_segment_closed.json').exists()):
                                self.log('No physics pause ACK; retaining SONIC until Isaac stops.')
                                os.killpg(bridge.pid, signal.SIGINT)
                                bridge.wait(timeout=30)
                    os.killpg(child.pid, signal.SIGINT)
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    self.log(f'{name} cleanup pending, PID {child.pid}; preserving {self.runtime}')
        if (self.run and self.run.exists() and str(self.run) not in self.closed_runs
                and (self.run / 'bridge_segment_closed.json').is_file()
                and all(self.children.get(name) is not None and self.children[name].poll() is not None
                        for name in ('manager', 'sonic'))):
            try:
                self.close_episode()
            except Exception as error:
                self.shutdown_error = str(error)
                self.log(f'Final episode retained for recovery: {error}')
        bridge = self.children.get('bridge')
        if bridge is not None and bridge.poll() is None:
            os.killpg(bridge.pid, signal.SIGINT)
            try:
                bridge.wait(timeout=30)
            except subprocess.TimeoutExpired:
                # The native-close marker is emitted after recording cleanup
                # and camera shutdown, even when no episode was recorded or
                # finalization failed. Preserve raw work, but do not strand Kit.
                bridge_log = self.runtime / 'bridge.log'
                native_close = (bridge_log.is_file() and '[STOP] Closing Isaac Sim' in
                                bridge_log.read_text(errors='replace'))
                if native_close or (self.run and (not self.run.exists() or (self.run / 'bridge_segment_closed.json').exists())):
                    os.killpg(bridge.pid, signal.SIGTERM)
                    try:
                        bridge.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.log(f'Owned Isaac native teardown still blocked; terminating PID {bridge.pid}, preserving raw work.')
                        os.killpg(bridge.pid, signal.SIGKILL)
                        bridge.wait(timeout=5)
                else:
                    self.log(f'Isaac cleanup pending PID {bridge.pid}; raw work retained.')
        complete = (bool(self.results) and all(row['result']['status'] == 'complete' and
                    row['result'].get('mentor_submission', {}).get('status') == 'complete' for row in self.results))
        if (complete and self.run and not self.run.exists()
                and all(child.poll() is not None for child in self.children.values()) and not self.args.smoke_source):
            shutil.rmtree(self.runtime)
            self.log(f'Batch stopped. Saved {len(self.results)} Mentor datasets; temporary runtime removed.')
        else:
            self.log(f'Batch stopped. Closed episodes: {len(self.results)}. Runtime log: {self.runtime}')


def main():
    from runtime_cpu import configure_cpu_affinity
    configure_cpu_affinity()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', choices=['supermarket'], default='supermarket')
    parser.add_argument('--task', default=os.environ.get('PICO_TASK'))
    parser.add_argument('--headless', action='store_true', default=os.environ.get('PICO_HEADLESS', '0') == '1')
    parser.add_argument('--episodes', type=int, default=0, help='0: until Ctrl+C')
    parser.add_argument('--smoke-source', type=Path, help='diagnostic recorded POSE instead of a live headset')
    parser.add_argument('--pose-seconds', type=float, default=4.)
    parser.add_argument('--port', type=int, default=15556)
    args = parser.parse_args()
    if args.task is None:
        from upstream_contract import document
        args.task = document("configs/task1/init/task.json")["instruction"] if args.scene == "supermarket" else "pick_up_the_basket"
    busy = list(existing_control_processes())
    if busy:
        parser.error(f'existing control processes must finish first: {busy}')
    if args.episodes < 0 or not 3 <= args.pose_seconds <= 12:
        parser.error('episodes must be nonnegative and smoke pose seconds must be 3..12')
    if args.smoke_source and (not args.smoke_source.is_file() or args.episodes < 1):
        parser.error('smoke requires an existing source and an explicit episode limit')
    # One coordinator owns this checkout, including before the simulator PID exists.
    import hashlib
    lock_path = Path(tempfile.gettempdir()) / ('g1_batch_' + hashlib.sha256(str(ROOT).encode()).hexdigest()[:16] + '.lock')
    lock = lock_path.open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error('another batch collector already owns this checkout')
    collector = BatchCollector(args)
    previous = {sig: signal.signal(sig, lambda *_: setattr(collector, 'stop_requested', True))
                for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        collector.collect()
    except KeyboardInterrupt:
        collector.log('Stop requested; normal shutdown.')
    finally:
        collector.shutdown()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        lock.close()
    if collector.shutdown_error:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
