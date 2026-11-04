"""Bounded cleanup of captured descendants, including separate POSIX sessions."""
import os
from pathlib import Path
import signal
import time


def identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return int(fields[1]), fields[19], fields[0]
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def descendants(pids):
    table = {}
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            value = identity(int(path.name))
            if value:
                table[int(path.name)] = value
    owned = set(pids) & table.keys()
    while True:
        added = {pid for pid, value in table.items() if value[0] in owned} - owned
        if not added:
            break
        owned.update(added)
    return {pid: table[pid][1] for pid in owned if pid != os.getpid()}


def terminate(owned, grace=5.):
    """Only signal the captured PID/start-time identities; zombies are exited."""
    def alive():
        return [pid for pid, ticks in owned.items()
                if (value := identity(pid)) and value[1] == ticks and value[2] != 'Z']
    for sig, timeout in ((signal.SIGTERM, grace), (signal.SIGKILL, 2.)):
        for pid in alive():
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + timeout
        while alive() and time.monotonic() < deadline:
            time.sleep(.05)
        if not alive():
            return []
    return alive()
