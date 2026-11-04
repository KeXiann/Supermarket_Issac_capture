"""File protocol for rotating captures in one local Isaac process.

No robot/physics work runs here. Commands are local, scoped to one batch owner,
and acknowledged only after the simulator verifies the requested reset.
"""
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import tempfile
import time
import uuid

from capture_utils import atomic_json


def task_slug(task):
    slug = re.sub(r'[^a-z0-9]+', '_', task.lower()).strip('_')
    if not slug:
        raise ValueError('task must contain at least one ASCII letter or digit')
    return slug


def reserve_episode_output(outputs, task):
    """Reserve a durable per-task index without creating a partial dataset.

    Both single-episode and batch launchers share this lock/counter. Failed or
    discarded captures retain their reservation; indices are never recycled.
    """
    outputs = Path(outputs).resolve()
    work = outputs / '.capture_work'
    work.mkdir(parents=True, exist_ok=True)
    slug = task_slug(task)
    counter = work / '.episode_indices.json'
    with (work / '.episode_indices.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(counter.read_text()) if counter.exists() else {}
        if not isinstance(state, dict):
            raise ValueError('invalid episode index registry; refusing to reuse capture numbers')
        index = state.get(slug, 0)
        if type(index) is not int or index < 0:
            raise ValueError('invalid episode index; refusing to reuse capture numbers')
        pattern = re.compile(re.escape(slug) + r'_([0-9]+)')
        for child in outputs.iterdir():
            match = pattern.fullmatch(child.name)
            if match:
                index = max(index, int(match[1]) + 1)
        destination = outputs / f'{slug}_{index}'
        state[slug] = index + 1
        atomic_json(counter, state)
        return destination


def new_episode(outputs, task, profile, runtime, *, input_kind='live_pico'):
    from capture_session import initialize
    outputs, runtime = Path(outputs).resolve(), Path(runtime).resolve()
    work = outputs / '.capture_work'
    work.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=task_slug(task) + '_', dir=work))
    destination = reserve_episode_output(outputs, task)
    session = initialize(run, profile=profile, input_kind=input_kind, manual_recording=True,
                         task=task, mentor_only_output=destination if input_kind == 'live_pico' else None,
                         publication_root=outputs if input_kind == 'live_pico' else None)
    session.update(batch_runtime_directory=str(runtime), finalization_owner='batch_coordinator',
                   batch_bridge_segment=True, batch_reset_policy='restore initial robot, hands, scene bodies and zero velocities')
    atomic_json(run / 'session.json', session)
    return run, destination


def command(control, run, action='prepare'):
    token = uuid.uuid4().hex
    atomic_json(control, {'schema_version': 1, 'action': action, 'run': str(Path(run).resolve()),
                          'token': token, 'requested_monotonic_ns': time.monotonic_ns(),
                          'owner_pid': os.getpid()})
    return token


def read_prepare(control, current_run):
    """Reject foreign paths and replayed commands before any reset or write."""
    control = Path(control).resolve()
    if not control.exists():
        return None
    data = json.loads(control.read_text())
    if data.get('schema_version') != 1 or data.get('action') != 'prepare':
        return None
    run = Path(data['run']).resolve()
    if run == Path(current_run).resolve():
        return None
    runtime = control.parent
    if run.parent != runtime.parent or run == runtime:
        raise ValueError('batch episode is outside the isolated work root')
    session = json.loads((run / 'session.json').read_text())
    if Path(session['batch_runtime_directory']).resolve() != runtime:
        raise ValueError('batch episode belongs to another runtime')
    if any((run / name).exists() for name in ('bridge.npz', 'bridge_saved.json', 'ego_view.mp4')):
        raise FileExistsError('batch reset refuses to overwrite a previous recording')
    return data


def acknowledge_reset(control, data, details):
    atomic_json(Path(control).with_name('reset_ack.json'), {
        'token': data['token'], 'run': data['run'], 'bridge_pid': os.getpid(),
        'reset_verified': True, 'physics_steps_since_reset': 0,
        'host_monotonic_ns': time.monotonic_ns(), **details})


def process_start_ticks(pid):
    # comm may contain spaces; fields after its closing parenthesis start at 3.
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return int(fields[19])


def request_batch_stop(runtime):
    """Stop exactly this coordinator, defending against stale/reused PIDs."""
    data = json.loads((Path(runtime) / 'batch.json').read_text())
    pid = data['owner_pid']
    if (data.get('host_boot_id') != Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            or data.get('owner_start_ticks') != process_start_ticks(pid)):
        raise RuntimeError('batch coordinator no longer belongs to this runtime')
    argv = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    if b'batch_capture.py' not in {Path(os.fsdecode(arg)).name.encode() for arg in argv if arg}:
        raise RuntimeError('batch PID is not the capture coordinator')
    os.kill(pid, signal.SIGINT)
    return pid


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Reserve a sequential capture output directory')
    parser.add_argument('--outputs', type=Path, required=True)
    parser.add_argument('--task', required=True)
    args = parser.parse_args()
    print(reserve_episode_output(args.outputs, args.task))
