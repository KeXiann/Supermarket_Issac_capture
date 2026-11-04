#!/usr/bin/env python3
"""Extract official SONIC release decoders using verified HTTP ZIP ranges.

Only tensor metadata is unpickled through an allowlist; checkpoint code is never
executed. Optimizer/TRL state is not downloaded. Each selected ZIP member passes
its original CRC32; the resulting NPZ has its own SHA256 manifest.
"""
from __future__ import annotations
import argparse
import collections
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import pickle
import tempfile
import time
import zipfile

import numpy as np
import requests

REVISION = '6733128a3d8a523b1418b06bca3cdf61c8b0987f'
URL = f'https://huggingface.co/nvidia/GEAR-SONIC/resolve/{REVISION}/sonic_release/last.pt'
SIZE = 469418283
CHECKPOINT_SHA256 = 'e6bdab3f64a39336b3d41877d4f497d05f58af275f288ec0e6746c283ded8909'
DEFAULT = Path(__file__).resolve().parents[1]/'assets/sonic_release_reference/decoders.npz'


class MetadataOnly:
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


class Storage:
    def __init__(self, pid):
        self.key, self.size = pid[2], pid[4]


def tensor_metadata(storage, offset, shape, stride, *args):
    return {'storage': storage.key, 'storage_size': storage.size,
            'offset': offset, 'shape': tuple(shape), 'stride': tuple(stride)}


class MetadataReader(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ('collections', 'OrderedDict'):
            return collections.OrderedDict
        if (module, name) == ('collections', 'deque'):
            return collections.deque
        if (module, name) == ('torch._utils', '_rebuild_tensor_v2'):
            return tensor_metadata
        allowed = {('torch', 'FloatStorage'), ('torch', 'device'),
                   ('accelerate.state', 'PartialState'),
                   ('accelerate.utils.dataclasses', 'DistributedType'),
                   ('transformers.trainer_pt_utils', 'AcceleratorConfig'),
                   ('trl.trainer.ppo_config', 'PPOConfig'),
                   ('trl.trainer.utils', 'OnlineTrainerState'),
                   ('transformers.training_args', 'OptimizerNames')}
        allowed.update(('transformers.trainer_utils', x) for x in
                       ['HubStrategy', 'IntervalStrategy', 'SaveStrategy', 'SchedulerType'])
        if (module, name) in allowed:
            return MetadataOnly
        raise ValueError(f'unexpected checkpoint class: {module}.{name}')

    def persistent_load(self, pid):
        if not isinstance(pid, tuple) or len(pid) != 5 or pid[0] != 'storage':
            raise ValueError('unexpected checkpoint storage')
        return Storage(pid)


def fetch_range(start, end, cache):
    path = cache/f'{start}-{end}.bin'
    length = end-start+1
    if path.exists() and path.stat().st_size == length:
        return path
    for attempt in range(4):
        try:
            response = requests.get(URL+f'?byte_range={start}-{end}',
                                    headers={'Range': f'bytes={start}-{end}'}, timeout=(15, 25))
            response.raise_for_status()
            if (response.status_code != 206 or
                    response.headers.get('Content-Range') != f'bytes {start}-{end}/{SIZE}' or
                    len(response.content) != length):
                raise ValueError('server did not return the requested byte range')
            temp = path.with_suffix('.partial')
            temp.write_bytes(response.content)
            temp.replace(path)
            return path
        except (requests.RequestException, ValueError):
            if attempt == 3:
                raise
            time.sleep(attempt+1)


def extract(output=DEFAULT, workers=8):
    output = Path(output).resolve()
    manifest_path = output.with_suffix('.json')
    if output.exists():
        manifest = json.loads(manifest_path.read_text())
        if hashlib.sha256(output.read_bytes()).hexdigest() != manifest['npz_sha256']:
            raise ValueError('existing decoder artifact hash mismatch')
        return manifest
    output.parent.mkdir(parents=True, exist_ok=True)
    cache = output.parent/'download_ranges'
    cache.mkdir(exist_ok=True)
    first = fetch_range(0, 65535, cache)
    tail = fetch_range(SIZE-65536, SIZE-1, cache)
    with tempfile.TemporaryFile() as sparse:
        sparse.truncate(SIZE)
        sparse.seek(0); sparse.write(first.read_bytes())
        sparse.seek(SIZE-65536); sparse.write(tail.read_bytes())
        sparse.flush()
        with zipfile.ZipFile(sparse) as archive:
            prefix = archive.namelist()[0].split('/')[0]
            metadata_bytes = archive.read(prefix+'/data.pkl')
            metadata = MetadataReader(io.BytesIO(metadata_bytes)).load()
            selected = {k: v for k, v in metadata['policy_state_dict'].items()
                        if k.startswith(('actor_module.decoders.g1_kin.',
                                         'actor_module.decoders.g1_dyn.'))}
            if len(selected) != 24:
                raise ValueError('unexpected decoder state dictionary')
            infos = {k: archive.getinfo(prefix+'/data/'+v['storage']) for k, v in selected.items()}
            if any(i.compress_type != zipfile.ZIP_STORED for i in infos.values()):
                raise ValueError('expected uncompressed torch storage')
            start = min(i.header_offset for i in infos.values())
            end = max(i.header_offset+256+i.file_size for i in infos.values())
            chunk_size = 512*1024
            ranges = [(a, min(a+chunk_size-1, end)) for a in range(start, end+1, chunk_size)]
            print(f'Downloading {end-start+1:,} bytes for both decoders in {len(ranges)} ranges', flush=True)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                jobs = {pool.submit(fetch_range, a, b, cache): (a, b) for a, b in ranges}
                for count, job in enumerate(as_completed(jobs), 1):
                    a, b = jobs[job]
                    data = job.result().read_bytes()
                    sparse.seek(a); sparse.write(data)
                    if count % 10 == 0 or count == len(jobs):
                        print(f'Ranges complete: {count}/{len(jobs)}', flush=True)
            sparse.flush()
            weights, members = {}, {}
            for key, value in selected.items():
                info = infos[key]
                data = archive.read(info)  # zipfile verifies original CRC32
                shape = value['shape']
                if value['offset'] != 0 or len(data) != np.prod(shape)*4:
                    raise ValueError(f'unsupported tensor layout: {key}')
                weights[key] = np.frombuffer(data, dtype='<f4').reshape(shape).copy()
                members[key] = {'storage': value['storage'], 'shape': shape,
                                'zip_crc32': f'{info.CRC:08x}',
                                'sha256': hashlib.sha256(data).hexdigest()}
    with output.with_suffix('.partial').open('wb') as stream:
        np.savez_compressed(stream, **weights)
    output.with_suffix('.partial').replace(output)
    manifest = {'source_url': URL, 'revision': REVISION, 'checkpoint_size': SIZE,
                'official_full_checkpoint_sha256': CHECKPOINT_SHA256,
                'full_checkpoint_sha256_verified': False,
                'verification': 'Pinned official HTTPS source, exact Content-Range and ZIP member CRC32; selected tensor SHA256 below. Full training checkpoint was not downloaded.',
                'metadata_sha256': hashlib.sha256(metadata_bytes).hexdigest(),
                'npz_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                'members': members}
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--workers', type=int, default=8, choices=range(1, 17))
    args = parser.parse_args()
    result = extract(args.output, args.workers)
    print(json.dumps({k: v for k, v in result.items() if k != 'members'}, indent=2))
