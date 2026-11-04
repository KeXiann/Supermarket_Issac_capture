#!/usr/bin/env python3
"""Fetch fixed source revisions; never reset an existing user's checkout."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"bridge"))
from project_paths import SONIC_ROOT

SOURCES = {
    'sonic': ('https://github.com/NVlabs/GR00T-WholeBodyControl.git',
              '7f151314d4d1606544bf249d2a7a1cb754c64582', SONIC_ROOT),
    'isaaclab': ('https://github.com/isaac-sim/IsaacLab.git',
                 '37ddf626871758333d6ed89cf64ad702aef127d0', ROOT/'.runtime/IsaacLab-2.3.2-sparse'),
    'assets': ('https://github.com/Kinetra-Humanoids/GPT6-Assist-Humanoid-Robot-Mobile-Manipulation-Data-Generation.git',
               '7e5fd9d0ae2f414249a3f2b83367dbb6d773f946', ROOT/'assets/upstream/7e5fd9d0ae2f'),
}

def run(*args, **kwargs):
    return subprocess.run(args, check=True, **kwargs)

def fetch(name):
    url, revision, dest = SOURCES[name]
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        run('git', 'clone', '--filter=blob:none', '--no-checkout', url, str(dest),
            env=dict(os.environ, GIT_LFS_SKIP_SMUDGE='1'))
        if name == 'isaaclab':
            run('git', '-C', str(dest), 'sparse-checkout', 'set', 'apps', 'source')
        run('git', '-C', str(dest), 'checkout', '--detach', revision,
            env=dict(os.environ, GIT_LFS_SKIP_SMUDGE='1'))
    actual = subprocess.check_output(['git', '-C', str(dest), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != revision:
        raise RuntimeError(f'{dest} is at {actual}; expected {revision}. Use a separate checkout; no files were reset.')
    if name in ('sonic', 'assets'):
        run('git', '-C', str(dest), 'lfs', 'pull')
    if name == 'sonic':
        run('git', '-C', str(dest), 'submodule', 'update', '--init', '--recursive')
        patch = str(ROOT/'patches/sonic-capture.patch')
        reverse = subprocess.run(['git', '-C', str(dest), 'apply', '--reverse', '--check', patch],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if reverse.returncode != 0:
            run('git', '-C', str(dest), 'apply', '--check', patch)
            run('git', '-C', str(dest), 'apply', patch)
        print('SONIC capture patch applied; rebuild the deployment binary.')
    print(name, revision, dest)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sources', nargs='*', choices=tuple(SOURCES), default=list(SOURCES))
    for source in parser.parse_args().sources:
        fetch(source)
