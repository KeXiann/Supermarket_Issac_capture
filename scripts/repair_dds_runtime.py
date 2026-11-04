#!/usr/bin/env python3
"""Bind the project's CycloneDDS Python package to SONIC's matching SDK library."""
import base64
import csv
import hashlib
from importlib import metadata
from pathlib import Path
import shutil
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"bridge"))
from project_paths import SONIC_ROOT

def main():
    if Path(sys.prefix).resolve() != (ROOT/'.venv').resolve():
        raise RuntimeError('Run with this checkout .venv/bin/python; no system or conda packages are modified')
    dist = metadata.distribution('cyclonedds')
    if dist.version != '0.10.2':
        raise RuntimeError('Expected CycloneDDS 0.10.2')
    library = SONIC_ROOT/'gear_sonic_deploy/thirdparty/unitree_sdk2/thirdparty/lib/x86_64/libddsc.so'
    expected = '4038630c231412f7b34a2ea60df192bbdebd0a57f22ac7f35c1b6d28323e695c'
    if hashlib.sha256(library.read_bytes()).hexdigest() != expected:
        raise RuntimeError('Unexpected SONIC native DDS library')
    site = Path(dist.locate_file(''))
    package = site/'cyclonedds'
    native = site/'cyclonedds.libs/libddsc.so.0'
    native.parent.mkdir(exist_ok=True)
    shutil.copy2(library, native)
    for extension in package.glob('*.so'):
        needed = subprocess.check_output(['patchelf', '--print-needed', str(extension)], text=True).splitlines()
        for name in needed:
            if 'libddsc' in name and name != 'libddsc.so.0':
                subprocess.run(['patchelf', '--replace-needed', name, 'libddsc.so.0', str(extension)], check=True)
        subprocess.run(['patchelf', '--set-rpath', '$ORIGIN/../cyclonedds.libs', str(extension)], check=True)
    (package/'__library__.py').write_text("from pathlib import Path\nin_wheel = True\nlibrary_path = Path(__file__).resolve().parent.parent / 'cyclonedds.libs/libddsc.so.0'\n")
    # Keep the installed wheel's file inventory consistent after this local repair.
    record = next(site.glob('cyclonedds-*.dist-info/RECORD'))
    with record.open(newline='') as stream:
        rows = list(csv.reader(stream))
    entries = {row[0] for row in rows}
    entries.add(str(native.relative_to(site)))
    with record.open('w', newline='') as stream:
        writer = csv.writer(stream)
        for name in sorted(entries):
            path = site/name
            if path == record:
                writer.writerow([name, '', ''])
            elif path.is_file():
                blob = path.read_bytes()
                digest = base64.urlsafe_b64encode(hashlib.sha256(blob).digest()).rstrip(b'=').decode()
                writer.writerow([name, 'sha256='+digest, len(blob)])
    print('DDS library repaired inside project venv only:', native)

if __name__ == '__main__':
    main()
