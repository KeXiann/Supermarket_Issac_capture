"""Atomic trajectory checkpoints without blocking the physics loop on compression."""
import json
import os
from pathlib import Path
import threading

import numpy as np


class Recorder:
    def __init__(self):
        self.samples = []
        self._writer = None
        self._error = None

    def add(self, **data):
        self.samples.append({k: np.asarray(v).copy() for k, v in data.items()})

    @staticmethod
    def _write(out, samples, metadata):
        path = Path(out).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {k: np.stack([s[k] for s in samples], axis=0) for k in samples[0]}
        tmp_npz = path.with_name(path.name + ".tmp")
        json_path = path.with_suffix(".json")
        tmp_json = json_path.with_name(json_path.name + ".tmp")
        with tmp_npz.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        tmp_npz.replace(path)
        with tmp_json.open("w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        tmp_json.replace(json_path)
        print(f"[REC] {len(samples)} samples -> {path}", flush=True)

    def checkpoint(self, out, metadata):
        if not out or not self.samples:
            return False
        if self._error is not None:
            raise RuntimeError("background recording failed") from self._error
        if self._writer is not None and self._writer.is_alive():
            return False
        # Samples already own their arrays. Only snapshot the list here; stacking,
        # compression and fsync all run on the writer, never the control thread.
        snapshot = tuple(self.samples)
        def write():
            try:
                self._write(out, snapshot, metadata)
            except Exception as exc:
                self._error = exc
                print(f"[REC] checkpoint failed: {exc}", flush=True)
        self._writer = threading.Thread(target=write, name="bridge-recorder", daemon=True)
        self._writer.start()
        return True

    def save(self, out, metadata):
        if self._writer is not None:
            self._writer.join()
        if self._error is not None:
            raise RuntimeError("background recording failed") from self._error
        if out and self.samples:
            self._write(out, tuple(self.samples), metadata)
