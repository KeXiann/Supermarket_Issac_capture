#!/usr/bin/env python3
"""Durably record SONIC's causal state/action debug stream.

The patched SONIC binary publishes a msgpack map on ``g1_debug`` for every
50 Hz logger cycle and a ``robot_config`` map at startup.  This recorder keeps
the maps losslessly in a length-prefixed msgpack stream so the training export
can join the exact C++ state/action pair by ``state_monotonic_ns``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import struct
import time

import msgpack
import zmq


MAGIC = b"SONICDBG2\0"


def record(output: Path, port: int, topics: tuple[str, ...] = (), ready_file: Path | None = None) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    started = time.monotonic_ns()
    stop = False

    def request_stop(signum, frame):
        nonlocal stop
        stop = True

    old = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    socket.setsockopt(zmq.RCVHWM, 10000)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(f"tcp://127.0.0.1:{port}")
    try:
        # Give the PUB/SUB connection time to establish before SONIC starts.
        time.sleep(.15)
        # A second launch must never truncate an earlier recording.
        with output.open("xb") as stream:
            stream.write(MAGIC)
            stream.flush()
            os.fsync(stream.fileno())
            if ready_file is not None:
                ready_file.parent.mkdir(parents=True, exist_ok=True)
                ready_file.write_text("ready\n")
            while not stop:
                if not socket.poll(100):
                    continue
                packet = socket.recv()
                now = time.monotonic_ns()
                topic = "robot_config" if packet.startswith(b"robot_config") else "g1_debug"
                if topic == "robot_config":
                    payload = packet[len(b"robot_config"):]
                else:
                    # The topic is a configurable prefix; default is g1_debug.
                    payload = packet
                    for prefix in (b"g1_debug", b"state"):
                        if packet.startswith(prefix):
                            payload = packet[len(prefix):]
                            break
                try:
                    decoded = msgpack.unpackb(payload, raw=False, strict_map_key=False)
                except Exception:
                    # Preserve malformed frames as a diagnostic record rather
                    # than silently losing evidence.
                    decoded = {"_decode_error": True, "payload_hex": payload.hex()}
                if topics and topic not in topics:
                    continue
                row = {"topic": topic, "received_monotonic_ns": now, "payload": decoded}
                blob = msgpack.packb(row, use_bin_type=True)
                stream.write(struct.pack("<Q", len(blob)))
                stream.write(blob)
                count += 1
                if count % 50 == 0:
                    stream.flush()
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        socket.close(0)
        context.term()
        for sig, handler in old.items():
            signal.signal(sig, handler)
    metadata = {"format": "sonic_debug_msgpack_v2", "port": port,
                "records": count, "started_monotonic_ns": started,
                "closed_monotonic_ns": time.monotonic_ns(), "complete": True}
    tmp = output.with_suffix(output.suffix + ".json.tmp")
    tmp.write_text(json.dumps(metadata, indent=2) + "\n")
    tmp.replace(output.with_suffix(output.suffix + ".json"))
    return count


def read_records(path: Path):
    """Yield records and fail closed on truncation or wrong magic."""
    with Path(path).open("rb") as stream:
        if stream.read(len(MAGIC)) != MAGIC:
            raise ValueError(f"invalid SONIC debug stream: {path}")
        while True:
            raw = stream.read(8)
            if not raw:
                return
            if len(raw) != 8:
                raise ValueError(f"truncated SONIC debug length: {path}")
            size = struct.unpack("<Q", raw)[0]
            if not 1 <= size <= 128 * 1024 * 1024:
                raise ValueError(f"invalid SONIC debug record size: {size}")
            blob = stream.read(size)
            if len(blob) != size:
                raise ValueError(f"truncated SONIC debug record: {path}")
            yield msgpack.unpackb(blob, raw=False, strict_map_key=False)


def validate_recording(path: Path):
    """Require a durable close marker and uninterrupted publisher cycles."""
    path = Path(path)
    metadata = json.loads(path.with_suffix(path.suffix + ".json").read_text())
    if metadata.get("format") != "sonic_debug_msgpack_v2" or metadata.get("complete") is not True:
        raise ValueError("SONIC debug recorder has no successful close marker")
    records = list(read_records(path))
    if len(records) != metadata.get("records"):
        raise ValueError("SONIC debug record count differs from close marker")
    configs, states = [], []
    for record in records:
        payload = record.get("payload")
        if not isinstance(payload, dict) or payload.get("_decode_error"):
            raise ValueError("SONIC debug stream contains an invalid payload")
        if record.get("topic") == "robot_config":
            configs.append(payload)
        elif record.get("topic") == "g1_debug":
            if payload.get("body_action_alignment_version") != 2:
                raise ValueError("SONIC debug state is not body protocol v2")
            states.append(payload)
        else:
            raise ValueError("unexpected SONIC debug topic")
    if not configs or not states:
        raise ValueError("SONIC debug stream lacks robot config or state/action cycles")
    for previous, current in zip(states, states[1:]):
        if int(current["index"]) != int(previous["index"]) + 1:
            raise ValueError("SONIC debug publisher cycles are missing, duplicated or reordered")
        if int(current["state_monotonic_ns"]) <= int(previous["state_monotonic_ns"]):
            raise ValueError("SONIC debug state clocks must strictly increase")
    return {"records": len(records), "state_action_cycles": len(states),
            "first_state_monotonic_ns": int(states[0]["state_monotonic_ns"]),
            "last_state_monotonic_ns": int(states[-1]["state_monotonic_ns"]),
            "complete": True}, records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=5557)
    parser.add_argument("--read", type=Path)
    parser.add_argument("--ready-file", type=Path)
    args = parser.parse_args()
    if args.read:
        rows = list(read_records(args.read))
        print(json.dumps({"records": len(rows), "topics": sorted({r.get("topic") for r in rows})}))
    else:
        print(f"SONIC_DEBUG_RECORDING {args.output} tcp://127.0.0.1:{args.port}", flush=True)
        print(f"SONIC_DEBUG_RECORDS {record(args.output, args.port, ready_file=args.ready_file)}", flush=True)


if __name__ == "__main__":
    main()
