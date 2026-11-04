"""Current data contract and explicit, read-only support for historical datasets."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from upstream_contract import COMMIT, document

# Schema revision is unchanged by later asset commits; keep reader compatibility.
MODE = "kinetra_46443c1"
VIDEO_KEY = "observation.images.ego_view"
LEGACY = Path(__file__).resolve().parents[1] / "assets/collection_reference_2609"
ANALOG_HAND_FIELDS = ("observation.previous_hand_action_joint_position", "action.aligned_hand_joint_position")
DATASET_STATS_FORMAT = 'example_float_quantiles_v1'


def dataset_statistics(columns):
    """Example-compatible global stats: float fields, quantiles and fingerprints.

    Fingerprints describe this dataset's actual column content, not the source
    example. Quantiles use NumPy's linear interpolation over all dataset rows.
    """
    result, fingerprints = {}, {}
    for name, feature in template()['features'].items():
        if feature['dtype'] not in ('float32', 'float64'):
            continue
        column = columns[name]
        if hasattr(column, 'to_pylist'):
            column = column.to_pylist()
        raw = np.asarray(column, dtype=np.dtype(feature['dtype']).newbyteorder('<')).reshape(-1, *feature['shape'])
        if not len(raw) or not np.isfinite(raw).all():
            raise ValueError('Invalid dataset statistics input: ' + name)
        values = raw.astype(np.float64)
        result[name] = {'mean': values.mean(axis=0).tolist(), 'std': values.std(axis=0).tolist(),
                        'min': values.min(axis=0).tolist(), 'max': values.max(axis=0).tolist(),
                        'q01': np.quantile(values, .01, axis=0, method='linear').tolist(),
                        'q99': np.quantile(values, .99, axis=0, method='linear').tolist()}
        header = json.dumps([name, feature['dtype'], list(raw.shape)], separators=(',', ':')).encode()
        fingerprints[name] = 'sha256:' + hashlib.sha256(header + b'\0' + raw.tobytes(order='C')).hexdigest()
    return {**result, '__fingerprints__': fingerprints}


def validate_dataset_statistics(recorded, columns):
    actual = dataset_statistics(columns)
    if set(recorded) != set(actual) or recorded.get('__fingerprints__') != actual['__fingerprints__']:
        raise ValueError('Dataset statistics fields/fingerprints differ from actual data')
    for name in actual.keys() - {'__fingerprints__'}:
        if set(recorded[name]) != set(actual[name]):
            raise ValueError('Dataset statistics require mean/std/min/max/q01/q99: ' + name)
        for key, values in actual[name].items():
            supplied = np.asarray(recorded[name][key])
            if supplied.shape != np.shape(values) or not np.isfinite(supplied).all() or not np.allclose(supplied, values, rtol=1e-7, atol=1e-8):
                raise ValueError('Dataset statistics differ from actual data: ' + name + '/' + key)


def template():
    return document("configs/data_standard/meta/info.json")


def modality():
    return document("configs/data_standard/meta/modality.json")


def current(info):
    return info.get("script_config", {}).get("lerobot_schema_mode") == MODE


def expected_template(info):
    mode = info.get("script_config", {}).get("lerobot_schema_mode", "g1_bridge_v1")
    if mode == MODE:
        return template()
    if mode not in ("mentor_2609_exact", "g1_bridge_v1"):
        raise ValueError("Unknown data standard: " + mode)
    expected = json.loads((LEGACY / "info.template.json").read_text())
    if mode == "mentor_2609_exact":
        for key in ANALOG_HAND_FIELDS:
            expected["features"].pop(key)
    return expected


def capture_semantics(values):
    """Retain the published contract verbatim; declare simulation adaptations."""
    source = template()["script_config"]
    result = copy.deepcopy(source)
    result.update({key: value for key, value in values.items() if key not in ("body_alignment", "hand_recording")})
    result["simulation_adapter"] = {key: values[key] for key in ("body_alignment", "hand_recording") if key in values}
    # Keep existing readers' simulation-only keys, without changing contract text.
    result["hand_recording"].update({key: value for key, value in values.get("hand_recording", {}).items()
                                     if key not in source["hand_recording"]})
    result.update(lerobot_schema_mode=MODE, lerobot_training_only_fields=[],
                  data_standard={"commit": COMMIT, "info": "configs/data_standard/meta/info.json",
                                 "modality": "configs/data_standard/meta/modality.json"})
    return result


class RGBStatistics:
    """Exact byte histograms of decoded RGB; bounded memory, population std."""
    def __init__(self):
        self.hist = np.zeros((3, 256), dtype=np.int64)
        self.frames = 0

    def add_bgr(self, frame):
        if frame.shape != (360, 640, 3) or frame.dtype != np.uint8:
            raise ValueError("encoded frame dimensions/dtype differ")
        for channel in range(3):
            self.hist[channel] += np.bincount(frame[:, :, 2 - channel].ravel(), minlength=256)
        self.frames += 1

    def result(self):
        if not self.frames:
            raise ValueError("video/data count mismatch: empty video")
        levels = np.arange(256, dtype=np.float64) / 255
        count = self.hist.sum(axis=1)
        mean = (self.hist * levels).sum(axis=1) / count
        variance = (self.hist * (levels[None, :] - mean[:, None]) ** 2).sum(axis=1) / count
        values = {"min": np.array([levels[np.flatnonzero(row)[0]] for row in self.hist]),
                  "max": np.array([levels[np.flatnonzero(row)[-1]] for row in self.hist]),
                  "mean": mean, "std": np.sqrt(variance)}
        return {**{key: value[:, None, None].tolist() for key, value in values.items()}, "count": [self.frames]}


def video_statistics(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    stats = RGBStatistics()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            stats.add_bgr(frame)
    finally:
        cap.release()
    return stats.result()


def validate_image_statistics(recorded, actual, frames):
    if recorded.get("count") != [frames] or actual["count"] != [frames]:
        raise ValueError("video statistics count differs")
    for key in ("min", "max", "mean", "std"):
        values = np.asarray(recorded.get(key))
        if values.shape != (3, 1, 1) or not np.isfinite(values).all() or not np.allclose(values, actual[key], rtol=1e-7, atol=1e-8):
            raise ValueError("video statistics differ from decoded RGB: " + key)
