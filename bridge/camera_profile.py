"""Resolve scene-local camera defaults and explicit operator overrides without Kit."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from ego_camera import (EGO_CAMERA_PROFILE, EGO_CAMERA_MOUNT_M,
                        EGO_CAMERA_PITCH_DOWN_DEG, EGO_CAMERA_FOCAL_LENGTH_MM)
from camera_intrinsics import load_rectified_intrinsics
from camera_extrinsics import load_extrinsics, with_operator_overrides


def resolve_camera_profile(scene=None, *, config=None, mount=None, pitch=None, focal=None, intrinsics_config=None,
                           extrinsics_config=None):
    result = {'profile': EGO_CAMERA_PROFILE, 'mount_position_m': list(EGO_CAMERA_MOUNT_M),
              'pitch_down_deg': EGO_CAMERA_PITCH_DOWN_DEG, 'focal_length_mm': EGO_CAMERA_FOCAL_LENGTH_MM,
              'config_path': None, 'config_sha256': None, 'override_fields': []}
    path = Path(config).resolve() if config else (Path(scene).resolve().with_suffix('.camera.json') if scene else None)
    if config or (path and path.is_file()):
        raw = path.read_bytes(); data = json.loads(raw)
        if data.get('schema_version') != 1 or not isinstance(data.get('profile'), str) or not data['profile']:
            raise ValueError('Camera configuration requires schema_version=1 and a non-empty profile')
        for key in ('profile', 'mount_position_m', 'pitch_down_deg', 'focal_length_mm'):
            result[key] = data[key]
        result.update(config_path=str(path), config_sha256=hashlib.sha256(raw).hexdigest())
    extrinsics=(None if extrinsics_config=='virtual' else
                load_extrinsics(extrinsics_config) if extrinsics_config else load_extrinsics())
    if extrinsics:
        result['mount_position_m']=extrinsics['translation_m']
        result['pitch_down_deg']=extrinsics['pitch_down_deg']
        result['mount_profile']=extrinsics['profile']
    for key, value in [('mount_position_m', mount), ('pitch_down_deg', pitch), ('focal_length_mm', focal)]:
        if value is not None:
            result[key] = value; result['override_fields'].append(key)
    result['mount_position_m'] = [float(v) for v in result['mount_position_m']]
    result['pitch_down_deg'] = float(result['pitch_down_deg'])
    result['focal_length_mm'] = float(result['focal_length_mm'])
    if (len(result['mount_position_m']) != 3 or not all(math.isfinite(v) for v in result['mount_position_m'])
            or not math.isfinite(result['pitch_down_deg']) or not 0 <= result['pitch_down_deg'] < 85
            or not math.isfinite(result['focal_length_mm']) or result['focal_length_mm'] <= 0):
        raise ValueError('Camera profile requires finite xyz mount, pitch in [0,85), positive focal length')
    if extrinsics and (mount is not None or pitch is not None):
        extrinsics=with_operator_overrides(extrinsics,mount=mount,pitch=pitch)
    result['extrinsics']=extrinsics
    from upstream_camera import load_contract
    result['capture_contract'] = load_contract()
    result['intrinsics'] = (None if intrinsics_config == 'virtual' else
                            load_rectified_intrinsics(intrinsics_config) if intrinsics_config else load_rectified_intrinsics())
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    args = parser.parse_args()
    values = resolve_camera_profile(args.scene)
    print(*values['mount_position_m'], values['pitch_down_deg'], values['focal_length_mm'])
