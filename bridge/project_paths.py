"""Portable checkout paths shared by command-line entry points."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SONIC_ROOT = Path(os.environ.get("SONIC_ROOT", ROOT.parent / "GR00T-WholeBodyControl")).expanduser().resolve()


def code_path(name):
    path = ROOT / name
    if name.endswith('.py'):
        return ROOT / 'bridge' / name
    if name.endswith('.sh') and name not in ('capture.sh', 'setup.sh'):
        return ROOT / 'scripts' / name
    return path
