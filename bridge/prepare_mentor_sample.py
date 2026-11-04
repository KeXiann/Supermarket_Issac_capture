#!/usr/bin/env python3
"""Publish the versioned upstream data/meta/videos layout.

Formal live captures publish atomically and retain only data/meta/videos.
Legacy and diagnostic exports can keep raw evidence and a validation sidecar.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid

from capture_utils import atomic_json, sync_directory, sync_tree
import data_standard

LEGACY_META_FILES = ("info.json", "modality.json", "tasks.jsonl", "episodes.jsonl", "episodes_stats.jsonl")
META_FILES = (*LEGACY_META_FILES, "stats.json", "relative_stats.json")
MENTOR_FILES = {"data/chunk-000/episode_000000.parquet",
                "videos/chunk-000/observation.images.ego_view/episode_000000.mp4",
                *("meta/" + name for name in META_FILES)}


LEGACY_MENTOR_FILES = MENTOR_FILES - {"meta/stats.json", "meta/relative_stats.json"}


def submission_files(info):
    return MENTOR_FILES if data_standard.current(info) else LEGACY_MENTOR_FILES


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_manager_timing(run):
    """Retain host-clock diagnostics before formal publication removes raw packets."""
    import numpy as np
    stages = ('read', 'body_done', 'hand_done', 'retarget_done', 'pack_done', 'send_done')
    keys = ['timing_' + name + '_monotonic_ns' for name in stages]
    rows = []
    missing = invalid = 0
    for path in sorted((Path(run) / 'pico_input').glob('pose_*.npz')):
        with np.load(path, allow_pickle=False) as packet:
            if not all(key in packet for key in keys):
                missing += 1
                continue
            values = [np.asarray(packet[key]) for key in keys]
            if any(value.size != 1 or value.dtype.kind not in 'iu' for value in values):
                invalid += 1
                continue
            row = np.asarray([value.item() for value in values], dtype=np.int64)
            if row[0] <= 0 or np.any(np.diff(row) < 0):
                invalid += 1
                continue
            rows.append(row)
    if not rows and not invalid:
        return None  # Historical packets do not provide these measurements.
    result = {'scope': 'all recorded POSE packets in Manager session; includes pre-recording warmup',
              'clock': 'host monotonic; excludes headset/network latency before reader.get_latest',
              'packets_with_timing': len(rows), 'packets_without_timing': missing,
              'invalid_timing_packets': invalid, 'stages_ms': {}}
    if rows:
        boundaries = np.asarray(rows, dtype=np.int64)
        intervals = np.diff(boundaries, axis=1) / 1e6
        names = ('body_conversion', 'controller_and_hand', 'resample_retarget_and_preview',
                 'buffer_and_pack', 'socket_send')
        values = {name: intervals[:, i] for i, name in enumerate(names)}
        values['read_to_send'] = (boundaries[:, -1] - boundaries[:, 0]) / 1e6
        for name, value in values.items():
            result['stages_ms'][name] = {'mean': float(np.mean(value)),
                'p95': float(np.percentile(value, 95)), 'p99': float(np.percentile(value, 99)),
                'max': float(np.max(value))}
    return result


def validate_submission(root):
    import cv2
    import numpy as np
    import pyarrow.parquet as pq
    from export_capture_dataset import template, TRAINING_ONLY_FIELDS, VIDEO_KEY
    root = Path(root).resolve()
    if {p.name for p in root.iterdir()} != {"data", "meta", "videos"}:
        raise ValueError("submission must contain exactly data/, meta/, videos/")
    info = json.loads((root / "meta/info.json").read_text())
    expected_files = submission_files(info)
    legacy = (root / "meta/submission.json").exists()
    files = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    if not legacy:
        if files != expected_files:
            raise ValueError(f"submission must have exactly the versioned dataset files; extra={sorted(files - expected_files)}, missing={sorted(expected_files - files)}")
        if info.get("script_config", {}).get("submission_layout") != "mentor_minimal_v1":
            raise ValueError("missing minimal-submission metadata")
    expected = data_standard.expected_template(info)
    for key, feature in expected["features"].items():
        actual = dict(info["features"][key])
        if feature["dtype"] == "video":
            actual.pop("info", None)
        expected_feature = {k: v for k, v in feature.items() if k != "info"}
        if actual != expected_feature:
            raise ValueError(f"submission feature differs: {key}")
    if set(expected["features"]) != set(info["features"]):
        raise ValueError("submission feature keys differ")
    reference = Path(__file__).resolve().parents[1] / "assets/collection_reference_2609"
    if json.loads((root / "meta/modality.json").read_text()) != data_standard.modality():
        raise ValueError("submission modality differs")
    data_path = root / info["data_path"].format(episode_chunk=0, episode_index=0)
    video_path = root / info["video_path"].format(episode_chunk=0, episode_index=0, video_key=VIDEO_KEY)
    table = pq.read_table(data_path)
    numeric = {k for k, v in expected["features"].items() if v["dtype"] != "video"}
    if len(table) != info["total_frames"] or set(table.column_names) != numeric:
        raise ValueError("submission row count or Parquet columns differ")
    from submission_replay import validate as validate_original_replay
    validate_original_replay(root)
    if not np.array_equal(np.array(table["timestamp"]), (np.arange(len(table)) / 50).astype(np.float32)):
        raise ValueError("submission timestamp grid differs")
    cap = cv2.VideoCapture(str(video_path))
    decoded = 0
    rgb_stats = data_standard.RGBStatistics()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[:2] != (360, 640):
                raise ValueError("submission video dimensions differ")
            rgb_stats.add_bgr(frame)
            decoded += 1
    finally:
        cap.release()
    if decoded != len(table):
        raise ValueError("submission video/Parquet length mismatch")
    if data_standard.current(info):
        records = [json.loads(line) for line in (root / "meta/episodes_stats.jsonl").read_text().splitlines() if line]
        if len(records) != 1 or records[0]["episode_index"] != 0 or set(records[0]["stats"]) != set(info["features"]):
            raise ValueError("incomplete per-episode statistics")
        stats = records[0]["stats"]
        if any(value.get("count") != [len(table)] for value in stats.values()):
            raise ValueError("per-episode statistics count differs")
        aggregate = json.loads((root / 'meta/stats.json').read_text())
        if info['script_config'].get('dataset_statistics_format') == data_standard.DATASET_STATS_FORMAT:
            data_standard.validate_dataset_statistics(aggregate, table)
        elif aggregate != stats:
            raise ValueError('Historical single-episode aggregate statistics differ')
        data_standard.validate_image_statistics(stats[VIDEO_KEY], rgb_stats.result(), len(table))
    if legacy:
        for manifest_path in (root / "meta/training/manifest.json",
                              root / "meta/training/semanticdpc_0826/manifest.json"):
            manifest = json.loads(manifest_path.read_text())
            if "artifact" in manifest:
                if checksum(manifest_path.parent / manifest["artifact"]) != manifest["sha256"]:
                    raise ValueError(f"submission training checksum differs: {manifest_path}")
        metadata = info["script_config"]
        if not (root / "meta" / metadata["training_artifacts"]["manifest"]).is_file():
            raise ValueError("broken submission training manifest reference")
    embedded = info.get("script_config", {}).get("submission", {}).get("artifact_checksums")
    if embedded is not None:
        if set(embedded) != expected_files - {"meta/info.json"}:
            raise ValueError("embedded submission checksum file list differs")
        for name, digest in embedded.items():
            if checksum(root / name) != digest:
                raise ValueError(f"submission checksum differs: {name}")
    entries_path = root / "meta/submission_checksums.json" if legacy else root.with_name(root.name + ".validation.json")
    if entries_path.exists():
        entries = json.loads(entries_path.read_text())["files"]
        actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and p != entries_path}
        if actual != {entry["path"] for entry in entries}:
            raise ValueError("submission checksum file list differs")
        for entry in entries:
            if checksum(root / entry["path"]) != entry["sha256"]:
                raise ValueError(f"submission checksum differs: {entry['path']}")
    return {"status": "complete", "top_level_directories": ["data", "meta", "videos"],
            "rows": len(table), "numeric_columns": len(numeric),
            "decoded_video_frames": decoded, "fps": 50,
            "mentor_feature_definitions_match": True, "file_count": len(files),
            "mentor_file_layout_match": not legacy}


def prepare(run, destination, *, write_validation=True):
    from capture_session import inspect
    from export_capture_dataset import validate
    run, destination = Path(run).resolve(), Path(destination).resolve()
    if destination == run or run in destination.parents:
        raise ValueError("submission must be separate from the original capture")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite: {destination}")
    report_path = destination.with_name(destination.name + ".validation.json")
    if write_validation and report_path.exists():
        raise FileExistsError(f"refusing to overwrite: {report_path}")
    report, _ = inspect(run)
    if report["status"] != "complete":
        failures = [c for c in report["checks"] if not c["passed"]]
        raise ValueError(f"source capture is incomplete: {failures}")
    source_validation = validate(run)
    session = json.loads((run / "session.json").read_text())
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("data", "videos"):
        shutil.copytree(run / name, destination / name)
    (destination / "meta").mkdir()
    source_info = json.loads((run / "meta/info.json").read_text())
    for name in (META_FILES if data_standard.current(source_info) else LEGACY_META_FILES):
        shutil.copy2(run / "meta" / name, destination / "meta" / name)
    info_path = destination / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["script_config"]["submission_layout"] = "mentor_minimal_v1"
    info["script_config"].pop("training_artifacts", None)
    info["script_config"].pop("source_run", None)
    manager_timing = summarize_manager_timing(run)
    if manager_timing is not None:
        info['script_config']['manager_processing_timing'] = manager_timing
    info["script_config"]["submission"] = {
        "source_session_id": session["session_id"], "source_session": run.name,
        "input_kind": session["input_kind"], "control_mode": session["control_mode"],
        "training_approved": False,
        "hand_domain": "Revo2 simulated hands; not BrainCo hardware/write acknowledgements",
        "batch_capture": bool(session.get('batch_bridge_segment')),
        "batch_reset_policy": session.get('batch_reset_policy'),
        "auxiliary_artifacts_included": False,
        "training_row_masks": "not included; training must reconstruct validity and continuous segments from Parquet before making history/action chunks",
    }
    if (run / 'batch_reset.json').is_file():
        info['script_config']['submission']['batch_reset_verification'] = json.loads((run / 'batch_reset.json').read_text())
    if "hand_recording" in info["script_config"]:
        info["script_config"]["hand_recording"]["full_11_joint_rad"] = (
            "not included in minimal Mentor submission; six normalized active joints per hand retained in Parquet")
    if "world_recording" in info["script_config"]:
        world = info["script_config"]["world_recording"]
        if world.get('available_in_raw_capture'):
            from submission_replay import embed
            world['embedded_replay'] = embed(run, destination / 'data/chunk-000/episode_000000.parquet')
            world['retained_in_submission'] = True
            world['path'] = 'data/chunk-000/episode_000000.parquet#' + world['embedded_replay']['key']
            info['script_config'].setdefault('hand_recording', {})['full_11_joint_rad'] = (
                'original measured states retained in embedded_replay bridge.npz; no passive-joint estimation required')
        else:
            world['retained_in_submission'] = False
            world['path'] = None
    atomic_json(info_path, info)
    for name in ("data", "videos"):
        for original in (run / name).rglob("*"):
            if (original.is_file() and not (original.relative_to(run).as_posix() == 'data/chunk-000/episode_000000.parquet' and
                    info['script_config'].get('world_recording', {}).get('embedded_replay'))
                    and checksum(original) != checksum(destination / original.relative_to(run))):
                raise ValueError(f"copied artifact differs: {original}")
    if not write_validation:
        # Keep integrity checks usable after temporary raw evidence is removed,
        # without adding an eighth file or a validation sidecar.
        info["script_config"]["submission"]["artifact_checksums"] = {
            name: checksum(destination / name) for name in sorted(submission_files(info) - {"meta/info.json"})}
        atomic_json(info_path, info)
    result = validate_submission(destination)
    entries = [{"path": str(path.relative_to(destination)), "sha256": checksum(path)}
               for path in sorted(destination.rglob("*")) if path.is_file()]
    if write_validation:
        atomic_json(report_path, {"algorithm": "sha256", "files": entries,
                    "source_session_id": session["session_id"], "source_capture_integrity": report["status"],
                    "source_validation": source_validation, "submission_validation": result})
    sync_tree(destination)
    result = validate_submission(destination)
    result["directory"] = str(destination)
    return result


def _rename_directory_without_replacing(source, destination):
    """Linux atomic promotion, including protection against an empty old directory."""
    libc = ctypes.CDLL(None, use_errno=True)
    rename = libc.renameat2
    rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))
    sync_directory(Path(destination).parent)
    sync_directory(Path(source).parent)


def _prepare_atomic(run, destination):
    # Stage next to the raw capture, on the same filesystem as the final output.
    # A failed copy never appears as a finished, task-named dataset.
    with tempfile.TemporaryDirectory(prefix=".publishing_", dir=run.parent) as temporary:
        staging = Path(temporary) / "dataset"
        result = prepare(run, staging, write_validation=False)
        _rename_directory_without_replacing(staging, destination)
    result["directory"] = str(destination)
    return result


def ensure_submission(run, *, update_latest=True, destination=None, write_validation=True,
                      atomic_publish=False):
    """Publish or revalidate this session's submission without replacing files."""
    run = Path(run).resolve()
    explicit_destination = destination is not None
    destination = Path(destination).resolve() if explicit_destination else run.with_name(run.name + "_mentor_sample")
    # Preserve older 17-file submissions, but stop returning them as the clean
    # Mentor deliverable. New captures use the usual _mentor_sample name.
    if (destination / "meta/submission.json").exists():
        if explicit_destination:
            raise FileExistsError(f"refusing to replace an existing extended submission: {destination}")
        destination = run.with_name(run.name + "_mentor_sample_minimal")
    session = json.loads((run / "session.json").read_text())
    if destination.exists():
        result = validate_submission(destination)
        origin = json.loads((destination / "meta/info.json").read_text())["script_config"]["submission"]
        if origin["source_session_id"] != session["session_id"]:
            raise ValueError("existing submission belongs to a different session; refusing to replace it")
    elif atomic_publish:
        if write_validation:
            raise ValueError("atomic Mentor-only publication does not create a validation sidecar")
        result = _prepare_atomic(run, destination)
    elif write_validation:
        result = prepare(run, destination)
    else:
        result = prepare(run, destination, write_validation=False)
    if not update_latest:
        result["directory"] = str(destination)
        return result
    # This pointer is outside the immutable capture and submission directories.
    session = json.loads((run / "session.json").read_text())
    pointer_root = Path(session.get("publication_root", run.parent))
    pointer = pointer_root / "latest_mentor_sample.txt"
    temporary = pointer.with_name(f".{pointer.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(str(destination) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(pointer)
        sync_directory(pointer.parent)
    finally:
        temporary.unlink(missing_ok=True)
    result["directory"] = str(destination)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    destination = args.output or args.run.with_name(args.run.name + "_mentor_sample")
    if args.check:
        result = validate_submission(args.run)
    elif args.output:
        result = prepare(args.run, destination)
    else:
        result = ensure_submission(args.run)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
