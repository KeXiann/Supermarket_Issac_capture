"""Pinned upstream robot assets shared by the live bridge and verification tools.

The complete robot uses unmodified upstream geometry, joint frames, materials
and wrist adapters. Only the current pinned snapshot is supported.
All runtime opinions belong to the stage's session layer.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from upstream_contract import COMMIT
SNAPSHOT = ROOT / "assets/robot_reference" / COMMIT[:12]
ROBOT = SNAPSHOT / "assets/robot/g1_revo2/g1_revo2.usda"
PROFILE = "kinetra_g1_revo2_" + COMMIT[:12]


def reference_paths(commit=None):
    commit = commit or COMMIT
    if commit != COMMIT:
        raise ValueError("Historical robot assets have been removed")
    snapshot = ROOT / 'assets/robot_reference' / commit[:12]
    manifest = json.loads((snapshot / 'manifest.json').read_text())
    if manifest['commit'] != commit:
        raise ValueError('Robot reference commit mismatch')
    return snapshot, snapshot / 'assets/robot/g1_revo2/g1_revo2.usda'


def verify_snapshot(commit=None):
    snapshot, robot = reference_paths(commit)
    expected_commit = commit or COMMIT
    manifest = json.loads((snapshot / "manifest.json").read_text())
    if manifest["commit"] != expected_commit:
        raise ValueError("Robot reference commit mismatch")
    for entry in manifest["files"]:
        path = snapshot / entry["path"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Robot reference missing/corrupt: {path}")
    return manifest
