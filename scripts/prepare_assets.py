#!/usr/bin/env python3
"""Install verified source assets locally. Large assets never enter this Git repo."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"bridge"))
from upstream_contract import COMMIT
SOURCE = ROOT/'assets/upstream'/COMMIT[:12]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def copy_checked(source, target, expected):
    if digest(source) != expected:
        raise ValueError(f'Unexpected source hash (check Git LFS): {source}')
    if target.exists():
        if digest(target) != expected:
            raise FileExistsError(f'Refusing to replace modified local file: {target}')
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)

def main():
    revision = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != COMMIT:
        raise ValueError('Upstream revision differs from pinned contract')
    manifest = json.loads((ROOT/'configs/upstream_manifest.json').read_text())
    for entry in manifest['files']:
        source = SOURCE/entry['path']
        if 'source_sha256' in entry:
            if digest(source) != entry['source_sha256']:
                raise ValueError('Resolved hand contract source changed')
            source = ROOT/'configs/revo2_pd_resolved.json'
        copy_checked(source, ROOT/entry['path'], entry['sha256'])
    manifest = json.loads((ROOT/'configs/robot_manifest.json').read_text())
    snapshot = ROOT/'assets/robot_reference'/COMMIT[:12]
    for entry in manifest['files']:
        copy_checked(SOURCE/entry['path'], snapshot/entry['path'], entry['sha256'])
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    subprocess.run([sys.executable, str(SOURCE/'scripts/usd_python.py'), str(ROOT/'bridge/setup_supermarket_scene.py')], check=True, cwd=ROOT)

if __name__ == '__main__':
    main()
