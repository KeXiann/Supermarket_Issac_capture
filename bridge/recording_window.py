"""Read explicit operator recording bounds without changing raw diagnostics."""
import json
from pathlib import Path
import numpy as np


def recording_window(run):
    run = Path(run)
    session_path = run / 'session.json'
    session = json.loads(session_path.read_text()) if session_path.exists() else {}
    if not session.get('manual_recording'):
        return None  # Historical and ordinary diagnostic captures keep their semantics.
    data = json.loads((run / 'recording_control.json').read_text())
    if data.get('schema_version') != 1 or data.get('status') != 'stopped':
        raise ValueError(f"recording window is {data.get('status')!r}; left grip + A starts/stops, left grip + B discards")
    start, end = data['start_monotonic_ns'], data['end_monotonic_ns']
    if type(start) is not int or type(end) is not int or not 0 < start < end:
        raise ValueError('invalid explicit recording interval')
    if data['host_boot_id'] != session['host_boot_id']:
        raise ValueError('recording window and session belong to different host boots')
    return data


def recording_mask(run, host_ns):
    window = recording_window(run)
    times = np.asarray(host_ns, dtype=np.int64)
    if window is None:
        return np.ones(times.shape, dtype=bool)
    return (times >= window['start_monotonic_ns']) & (times < window['end_monotonic_ns'])
