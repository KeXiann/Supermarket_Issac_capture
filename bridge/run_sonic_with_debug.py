#!/usr/bin/env python3
"""Run SONIC together with its durable causal debug recorder."""
from __future__ import annotations

import argparse
from pathlib import Path
import signal
import subprocess
import sys
import time


def main() -> int:
    from runtime_cpu import configure_cpu_affinity
    configure_cpu_affinity()
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--port", type=int, default=5557)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a SONIC command is required after --")
    root = Path(__file__).resolve().parents[1]
    output = args.run / "sonic_debug.msgpack"
    ready_file = args.run / ".sonic_debug_recorder.ready"
    ready_file.unlink(missing_ok=True)
    recorder = subprocess.Popen([sys.executable, str(root / "bridge/sonic_debug_recorder.py"),
                                 "--output", str(output), "--port", str(args.port),
                                 "--ready-file", str(ready_file)],
                                cwd=root, start_new_session=True)
    child = None

    def forward(signum, frame):
        # Keep recording while SONIC handles its shutdown; stop the recorder
        # only after the publisher has flushed and exited.
        if child is not None and child.poll() is None:
            try:
                child.send_signal(signum)
            except ProcessLookupError:
                pass

    old = {sig: signal.signal(sig, forward) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        deadline = time.monotonic() + 10
        while not ready_file.exists():
            if recorder.poll() is not None:
                raise RuntimeError(f"SONIC debug recorder exited with {recorder.returncode} before ready")
            if time.monotonic() >= deadline:
                raise TimeoutError("SONIC debug recorder did not become ready")
            time.sleep(.02)
        child = subprocess.Popen(command, start_new_session=False)
        while child.poll() is None:
            if recorder.poll() is not None:
                raise RuntimeError(f"SONIC debug recorder exited unexpectedly with {recorder.returncode}")
            time.sleep(.05)
        return_code = child.returncode
        time.sleep(.15)  # drain final PUB/SUB frames before recorder shutdown.
    finally:
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        if recorder.poll() is None:
            recorder.send_signal(signal.SIGINT)
        try:
            recorder.wait(timeout=20)
        except subprocess.TimeoutExpired:
            recorder.kill()
            recorder.wait()
        ready_file.unlink(missing_ok=True)
        for sig, handler in old.items():
            signal.signal(sig, handler)
    if recorder.returncode != 0:
        raise RuntimeError(f"SONIC debug recorder failed with {recorder.returncode}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
