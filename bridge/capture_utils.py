"""Small durable, atomic file helpers shared by the capture pipeline."""
import json
import math
import os
from pathlib import Path
import uuid


# Permit bounded PICO/XRT or host scheduling stalls without discarding all
# successfully saved data. This is a storage/continuity gate, NOT permission
# to reuse a human label for two seconds. The per-row limit stays at 100 ms.
MAX_HUMAN_PACKET_GAP_S = 2.0
HUMAN_REFERENCE_MAX_AGE_S = 0.1


def human_packet_gap_summary(packet_times):
    """Keep capture tolerance and strictly masked training gaps auditable."""
    times = [float(t) for t in packet_times]
    if len(times) < 2 or not all(math.isfinite(t) for t in times):
        raise ValueError("need at least two finite human packet timestamps")
    gaps = [b - a for a, b in zip(times, times[1:])]
    if min(gaps) <= 0:
        raise ValueError("human packet timestamps must strictly increase")
    issues = [{"before_packet_index": i, "after_packet_index": i + 1,
               "before_host_monotonic_s": times[i], "after_host_monotonic_s": times[i + 1],
               "offset_from_first_packet_s": times[i] - times[0], "gap_s": gap,
               "uncovered_fresh_reference_s": max(0., gap - HUMAN_REFERENCE_MAX_AGE_S)}
              for i, gap in enumerate(gaps) if gap > HUMAN_REFERENCE_MAX_AGE_S]
    return {"max_packet_gap_s": max(gaps), "capture_gap_gate_s": MAX_HUMAN_PACKET_GAP_S,
            "capture_gap_gate_passed": max(gaps) <= MAX_HUMAN_PACKET_GAP_S,
            "training_human_max_age_s": HUMAN_REFERENCE_MAX_AGE_S,
            "packet_gaps_over_100ms": len(issues),
            "packet_gaps_over_gate": sum(g > MAX_HUMAN_PACKET_GAP_S for g in gaps),
            "gap_intervals": issues,
            "gap_handling": "original timestamps preserved; human reference older than 100 ms stays invalid; no gap filling"}


def human_reference_valid(rows, age_s, modes):
    """Causal, fresh POSE association, independent of the capture gap gate."""
    return ((rows >= 0) & (age_s >= 0) & (age_s <= HUMAN_REFERENCE_MAX_AGE_S)
            & (modes == 2))


def sync_directory(path):
    fd = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def sync_tree(root):
    """Persist closed artifacts and directory entries before completion marker."""
    root = Path(root)
    directories = [root]
    for path in root.rglob("*"):
        if path.is_symlink():
            continue
        if path.is_dir():
            directories.append(path)
        elif path.is_file() and path.suffix != ".lock" and not path.name.startswith("supervisor_"):
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
    for directory in sorted(directories, key=lambda p: len(p.parts), reverse=True):
        sync_directory(directory)
