"""Select a physics runtime before importing Isaac/torch or opening the stage."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
NATIVE_PYTHON = ROOT / '.venv/bin/python'
NATIVE_READY = ROOT / '.runtime/uv_environment.json'
NATIVE_LAB = ROOT / '.runtime/IsaacLab-2.3.2-sparse'


def option(argv, name):
    for i, value in enumerate(argv):
        if value == name:
            return argv[i + 1]
        if value.startswith(name + '='):
            return value.split('=', 1)[1]
    return None


def default_profile(scene):
    if scene and Path(scene).name != 'pico_supermarket_task1_7e5fd9d.usda':
        raise ValueError('Only the current pinned task-1 scene is supported')
    return 'upstream_full'


def selected_profile(argv):
    bundle = option(argv, '--local-token-replay')
    if bundle:
        physics = json.loads(Path(bundle).with_suffix('.json').read_text()).get('physics_contract')
        if physics:
            return physics['profile']
    return option(argv, '--physics-profile') or default_profile(option(argv, '--scene'))


def bootstrap(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if '--help' in argv or '-h' in argv:
        return
    profile = selected_profile(argv)
    if profile != 'upstream_full':
        raise ValueError('Historical physics profiles were removed; old data is preserved but cannot be replayed by this runtime')
    if os.environ.get('PICO_RUNTIME', 'uv') != 'uv':
        raise ValueError('This distribution uses the isolated uv runtime')
    if not NATIVE_READY.is_file():
        raise RuntimeError('uv environment is not configured; run bash setup.sh runtime')
    if not json.loads(NATIVE_READY.read_text()).get('environment_checks_passed'):
        raise RuntimeError('uv runtime verification failed')
    from verify_uv_runtime import verify_lab_extensions
    verify_lab_extensions(NATIVE_LAB)
    expected = NATIVE_PYTHON
    # Compare lexical paths: venv interpreters may symlink to the same binary.
    if Path(sys.executable).absolute() != expected.absolute():
        print(f'[RUNTIME] {profile}: {expected}', flush=True)
        os.execv(str(expected), [str(expected), str(ROOT / "bridge/sonic_isaacsim_bridge.py"), *argv])
