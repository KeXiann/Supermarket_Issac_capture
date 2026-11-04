"""Pinned, byte-verified Kinetra camera/render contract for new captures.

The complete stereo source is retained, including right-eye and raw calibration.
Mentor's video stream remains the rectified left eye. Estimated torso mounting
is never labelled as a measured hand-eye calibration.
"""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from upstream_contract import COMMIT
SOURCE = ROOT / 'assets/upstream' / COMMIT[:12]
SCENE = ROOT / 'assets/scenes/pico_supermarket_task1_7e5fd9d.usda'
MANIFEST = ROOT / 'assets/camera_calibration/upstream_7e5fd9d.json'


def load_contract():
    manifest = json.loads(MANIFEST.read_text())
    if manifest['commit'] != COMMIT:
        raise ValueError('Camera source commit mismatch')
    documents = []
    for entry in manifest['files']:
        raw = (ROOT / entry['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry['sha256']:
            raise ValueError('Pinned upstream camera/render configuration changed: ' + entry['path'])
        documents.append(json.loads(raw))
    stereo, render, initial = documents
    for eye, view in (('left', 'zed_left'), ('right', 'zed_right')):
        camera = stereo[eye]
        k = camera['K']
        if ([camera['width'], camera['height']] != render['resolution']
                or any(camera['distortion'])
                or [k[0][0], k[1][1], k[0][2], k[1][2]] !=
                [render['intrinsics'][key] for key in ('fx', 'fy', 'cx', 'cy')]
                or camera['mount'] != render['mounts'][view]):
            raise ValueError('Upstream stereo and rendering contracts disagree')
    if stereo['baseline_m'] != render['baseline_m']:
        raise ValueError('Upstream stereo baseline mismatch')
    return {'commit': COMMIT, 'source_manifest': str(MANIFEST),
            'source_manifest_sha256': hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
            'stereo': stereo, 'render': render, 'initial_scene': initial}


def launcher_options(contract=None):
    """Choose RT2 before Kit initializes; ignore unrelated saved UI settings."""
    contract = contract or load_contract()
    values = contract['initial_scene']['render_settings']
    tokens = ['--/app/settings/persistent=false', '--/app/settings/loadUserConfig=false']
    tokens.extend('--' + key + '=' + (value if isinstance(value, str) else json.dumps(value, separators=(',', ':')))
                  for key, value in values.items())
    return {'renderer': contract['render']['renderer'], 'kit_args': ' '.join(tokens)}


def validate_capture_profile(profile, width, height):
    contract = profile['capture_contract']
    expected = contract['stereo']['left']
    if ([width, height] != [expected['width'], expected['height']]
            or profile['intrinsics'] is None or profile['extrinsics'] is None
            or profile['intrinsics']['K_px'] != expected['K']
            or profile['extrinsics']['T_parent_camera_cv'] != expected['mount']['body_T_camera_cv']
            or profile['override_fields']):
        raise ValueError('Upstream capture requires the pinned resolution, K and full torso camera matrix; '
                         'camera overrides require an explicit diagnostic wall/legacy schedule')


def settings_match(expected, actual):
    # Carb stores float settings in float32; preserve exact source and report readback.
    return all(math.isclose(actual.get(key), value, rel_tol=1e-6, abs_tol=1e-7)
               if isinstance(value, float) and isinstance(actual.get(key), (int, float))
               else actual.get(key) == value for key, value in expected.items())


def apply_render_settings(contract=None):
    """Apply the exact task initialization settings after IsaacLab's defaults."""
    import carb.settings
    contract = contract or load_contract()
    settings = carb.settings.get_settings()
    expected = contract['initial_scene']['render_settings']
    for key, value in expected.items():
        settings.set(key, value)
    actual = {key: settings.get(key) for key in expected}
    if not settings_match(expected, actual):
        raise RuntimeError('Renderer settings did not retain upstream values: ' + repr({k: (expected[k], actual[k]) for k in expected if actual[k] != expected[k]}))
    return {'commit': COMMIT, 'profile': contract['render']['profile'],
            'requested_settings': expected, 'settings': actual, 'source': contract['source_manifest']}


def native_frame_due(physical_step, *, physics_hz=200, camera_hz=25):
    """Pre-action frames at states 0,8,16,...; no wall-time skips or bursts."""
    if physics_hz % camera_hz:
        raise ValueError('Camera frequency must divide physics frequency')
    return physical_step % (physics_hz // camera_hz) == 0


def select_camera_frames(camera_rows, state_host, state_sim, schedule):
    """Join new pre-action frames on physics time; keep old host-clock association."""
    import numpy as np
    times = np.array([float(row['host_monotonic_s']) for row in camera_rows])
    if schedule == 'upstream':
        eligible = np.array([i for i, row in enumerate(camera_rows)
                             if int(row.get('source_physics_step', -1)) >= 0], dtype=np.int64)
        if not len(eligible):
            raise ValueError('Missing upstream camera source physics steps')
        sim = np.array([float(camera_rows[i]['sim_time_s']) for i in eligible])
        chosen = np.searchsorted(sim, np.asarray(state_sim) + 1e-7, side='right') - 1
        if np.any(chosen < 0):
            raise ValueError('Camera does not cover the first physical state')
        selected = eligible[chosen]
    else:
        selected = np.searchsorted(times, state_host, side='right') - 1
    if (np.any(selected < 0) or np.any(np.asarray(state_host) - times[selected] < -1e-7)
            or np.max(np.asarray(state_host) - times[selected]) > .5):
        raise ValueError('Missing/stale causally preceding camera frame')
    return selected
