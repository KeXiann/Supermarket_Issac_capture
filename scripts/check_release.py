#!/usr/bin/env python3
"""Reject generated/binary/private files from the Git publication set."""
from pathlib import Path
import re
import json
import subprocess
ROOT = Path(__file__).resolve().parents[1]
files = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
forbidden = {'.mp4','.mkv','.avi','.npz','.npy','.onnx','.pt','.pth','.engine','.whl','.deb','.so','.usd','.usda','.usdc','.zip'}
secrets = [re.compile(rb'gh[pousr]_[A-Za-z0-9]{30,}'), re.compile(rb'github_pat_[A-Za-z0-9_]{30,}'),
           re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')]
errors = []; total = 0
for name in filter(None, files):
    path = ROOT/name
    if path.is_symlink(): errors.append(f'symlink: {name}'); continue
    if not path.is_file(): errors.append(f'missing: {name}'); continue
    if name.startswith(('outputs/','.runtime/','.venv','data/','assets/upstream/','assets/robot_reference/','assets/scenes/')):
        errors.append(f'local/generated path: {name}')
    size=path.stat().st_size;total+=size
    if size>1024*1024: errors.append(f'file exceeds 1 MiB: {name}')
    if path.suffix.lower() in forbidden: errors.append(f'binary/generated file: {name}')
    blob=path.read_bytes()
    if (b'/home/' + b'user/') in blob:errors.append(f'machine-specific home path: {name}')
    if any(pattern.search(blob) for pattern in secrets):errors.append(f'credential material: {name}')
reference = ROOT/'assets/collection_reference_2609/reference.json'
manifest = json.loads(reference.read_text())
required = list(manifest['files']) + list(manifest['semanticdpc_conversion']['files_sha256'])
required += ['dependencies/lerobot_utils.py']
tracked = set(files)
for name in required:
    path = 'assets/collection_reference_2609/' + name
    if path not in tracked:
        errors.append(f'required compatibility source is not tracked: {path}')
if errors:raise SystemExit('\n'.join(errors))
print(f'Publication check OK: {len(list(filter(None,files)))} files, {total/1024/1024:.2f} MiB; no oversized files, captured data or recognized credentials.')
