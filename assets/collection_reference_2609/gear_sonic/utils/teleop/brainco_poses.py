"""Validated normalized BrainCo poses, shared by control and recording."""

import json
import math
from pathlib import Path

SIDES = ("left", "right")
JOINT_ORDER = ("thumb", "thumb_aux", "index", "middle", "ring", "pinky")
DEFAULT_POSE_CONFIG = Path(__file__).resolve().parents[3] / "config/brainco_hand_poses.json"


def validate_poses(poses):
    if not isinstance(poses, dict) or set(poses) != set(SIDES):
        raise ValueError("BrainCo poses must contain left and right")
    result = {}
    for side in SIDES:
        if not isinstance(poses[side], dict) or set(poses[side]) != {"open", "closed"}:
            raise ValueError(f"BrainCo {side} must contain open and closed poses")
        result[side] = {}
        for name in ("open", "closed"):
            values = poses[side][name]
            if not isinstance(values, (list, tuple)) or len(values) != 6 or any(
                isinstance(v, bool) or not isinstance(v, (int, float))
                or not math.isfinite(v) or not 0 <= v <= 1 for v in values
            ):
                raise ValueError(f"BrainCo {side}.{name} requires six finite values in [0,1]")
            result[side][name] = [float(v) for v in values]
    return result


def load_poses(path=""):
    with open(path or DEFAULT_POSE_CONFIG) as stream:
        return validate_poses(json.load(stream))


def pose_targets(poses, binary):
    return {side: list(poses[side]["closed" if goal else "open"])
            for side, goal in zip(SIDES, binary)}
