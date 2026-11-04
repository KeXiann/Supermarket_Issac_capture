#!/usr/bin/env python3
"""Verify the isolated uv environment without starting Kit or physics."""
import hashlib
import importlib
from importlib import metadata
import json
import re
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def verify_lab_extensions(lab):
    """Resolve Lab's Kit dependencies locally, never via registry fallback."""
    pending = set()
    for app in (lab / 'apps').glob('*.kit'):
        # Kit permits repeated settings tables, unlike standard TOML. Parse
        # only dependency sections, whose syntax is ordinary TOML.
        for block in re.findall(r'^\[dependencies\]\s*\n(.*?)(?=^\[|\Z)', app.read_text(), re.M | re.S):
            pending.update(name for name in tomllib.loads(block)
                           if name.startswith('isaaclab') and not name.startswith('isaaclab.python'))
    versions = {}
    while pending:
        name = pending.pop()
        if name in versions:
            continue
        manifest = lab / 'source' / name / 'config/extension.toml'
        if not manifest.is_file():
            raise RuntimeError(f'Missing local Kit extension {name}; run bash setup.sh runtime')
        config = tomllib.loads(manifest.read_text())
        versions[name] = config['package']['version']
        for module in config.get('python', {}).get('module', []):
            module_path = manifest.parents[1] / module['name'].replace('.', '/')
            if not module_path.is_dir() and not module_path.with_suffix('.py').is_file():
                raise RuntimeError(f'Missing extension Python source: {module_path}')
        pending.update(dep for dep in config.get('dependencies', {}) if dep.startswith('isaaclab'))
    return versions


def verify():
    expected = ROOT / '.venv'
    assert Path(sys.prefix) == expected, 'Use the project .venv interpreter'
    paths = [sys.base_prefix, *sys.path]
    assert not any('miniconda' in p or '/envs/unitree_sim_env' in p for p in paths), paths
    config = (expected / 'pyvenv.cfg').read_text()
    assert 'include-system-site-packages = false' in config
    packages = {name: metadata.version(name) for name in (
        'isaacsim', 'isaaclab', 'torch', 'torchvision', 'numpy', 'scipy', 'cyclonedds',
        'pyarrow', 'pandas', 'av', 'opencv-python', 'fastapi', 'starlette')}
    required = {'isaacsim': '5.1.0.0', 'isaaclab': '0.54.2', 'torch': '2.7.0+cu128',
                'torchvision': '0.22.0+cu128', 'numpy': '1.26.0', 'scipy': '1.15.3',
                'fastapi': '0.121.3', 'starlette': '0.49.1'}
    assert all(packages[k] == v for k, v in required.items()), packages
    for name in ('numpy', 'scipy', 'torch', 'torchvision', 'pyarrow', 'pandas', 'av',
                 'cv2', 'cyclonedds', 'onnxruntime', 'msgpack_numpy', 'zmq', 'fastapi', 'starlette', 'h5py'):
        importlib.import_module(name)
    lab = ROOT / '.runtime/IsaacLab-2.3.2-sparse'
    commit = subprocess.check_output(['git', '-C', str(lab), 'rev-parse', 'HEAD'], text=True).strip()
    assert commit == '37ddf626871758333d6ed89cf64ad702aef127d0'
    assert (lab / 'VERSION').read_text().strip() == '2.3.2'
    extensions = verify_lab_extensions(lab)
    # Import-only checks missed a native-library abort in dds_create_domain.
    # Exercise it in a subprocess so a native failure cannot produce a valid marker.
    dds_code = '''
import json
from pathlib import Path
from cyclonedds.domain import Domain, DomainParticipant
domain = Domain(0, '<CycloneDDS><Domain Id="0"><General><Interfaces><NetworkInterface name="lo"/></Interfaces></General></Domain></CycloneDDS>')
participant = DomainParticipant(0)
maps = Path('/proc/self/maps').read_text()
libraries = sorted({line.split()[-1] for line in maps.splitlines() if 'libddsc' in line})
assert len(libraries) == 1 and '/.venv/' in libraries[0], libraries
assert 'miniconda' not in maps, 'Conda native dependency leaked into uv runtime'
print(json.dumps({'domain_initialization_passed': True, 'native_libraries': libraries}))
'''
    dds = json.loads(subprocess.check_output([sys.executable, '-c', dds_code], text=True, timeout=30))
    report = {
        'environment_checks_passed': True, 'python': sys.executable, 'base_prefix': sys.base_prefix,
        'python_version': sys.version, 'include_system_site_packages': False,
        'conda_paths_in_sys_path': [], 'packages': packages, 'isaaclab_release': '2.3.2',
        'isaaclab_commit': commit,
        'local_kit_extensions': extensions,
        'dds_runtime': dds,
        'uv_lock_sha256': hashlib.sha256((ROOT / 'uv.lock').read_bytes()).hexdigest(),
        'dependency_override': {'fastapi': '0.121.3',
            'reason': 'Isaac Lab pins Starlette 0.49.1, excluded by Isaac Sim kernel FastAPI 0.115.7; override only in this uv project.'},
        'simulation_app_started': False, 'physics_steps': 0, 'dynamic_stability_tested': False,
    }
    destination = ROOT / '.runtime/uv_environment.json'
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    verify()
