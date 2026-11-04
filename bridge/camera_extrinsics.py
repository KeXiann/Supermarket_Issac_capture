"""Photo/CAD mounting prior. It is not a measured robot-camera calibration."""
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRINSICS = ROOT/'assets/camera_calibration/kinetra_camera_extrinsics_7e5fd9d.json'


def validate_extrinsics(extrinsics):
    matrix = np.asarray(extrinsics['T_parent_camera_cv'],dtype=float)
    if (matrix.shape != (4,4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3],[0,0,0,1],rtol=0,atol=1e-9)
            or not np.allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3),rtol=0,atol=1e-7)
            or not np.isclose(np.linalg.det(matrix[:3,:3]),1.,rtol=0,atol=1e-7)):
        raise ValueError('Extrinsics must be a finite proper rigid 4x4 column-vector transform')
    if extrinsics.get('parent_frame')!='torso_link' or extrinsics.get('rotation_follow')!='rigid_parent':
        raise ValueError('Estimated camera requires rigid torso_link mounting; do not treat torso as pelvis')
    if extrinsics.get('calibrated') or extrinsics.get('status') not in ('estimated_not_calibrated','operator_override_not_calibrated'):
        raise ValueError('Photo/CAD mount cannot be promoted to a measured calibration')
    return matrix


def load_extrinsics(path=DEFAULT_EXTRINSICS):
    path=Path(path).expanduser().resolve();raw=path.read_bytes();result=json.loads(raw)
    if result.get('schema_version')!=1:raise ValueError('Extrinsics schema_version must be 1')
    validate_extrinsics(result)
    source=result['source']
    snapshot=ROOT/source['snapshot_path']
    if hashlib.sha256(snapshot.read_bytes()).hexdigest()!=source['sha256']:
        raise ValueError('Pinned mount source snapshot changed')
    result.update(config_path=str(path),config_sha256=hashlib.sha256(raw).hexdigest())
    return result


def with_operator_overrides(extrinsics, *, mount=None, pitch=None):
    result=copy.deepcopy(extrinsics);matrix=validate_extrinsics(result).copy()
    if mount is not None:
        matrix[:3,3]=mount
    if pitch is not None:
        if not math.isfinite(pitch) or not 0<=pitch<85:raise ValueError('Pitch must be finite in [0,85)')
        s,c=math.sin(math.radians(pitch)),math.cos(math.radians(pitch))
        matrix[:3,:3]=[[0,-s,c],[-1,0,0],[0,-c,-s]]
    result.update(T_parent_camera_cv=matrix.tolist(),translation_m=matrix[:3,3].tolist(),
                  pitch_down_deg=math.degrees(math.atan2(-matrix[2,2],math.hypot(matrix[0,2],matrix[1,2]))),
                  status='operator_override_not_calibrated',operator_override_fields=[k for k,v in [('mount',mount),('pitch',pitch)] if v is not None])
    validate_extrinsics(result)
    return result
