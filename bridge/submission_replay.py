"""Retain original simulation evidence inside the existing Mentor Parquet file.

The optional binary Arrow schema metadata does not add training columns, files,
or directories. It contains the original finalized files with their own clocks;
it is never a token rollout or an estimate of the recorded world trajectory.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile

import numpy as np

KEY = b'g1_isaacsim_bridge.original_replay.v1'
REQUIRED = ('bridge.npz', 'bridge.json', 'bridge_saved.json', 'world_trace.npz',
            'world_trace.json', 'world_paths.json', 'meta/bridge_alignment.npz')
OPTIONAL = ('ego_view.timestamps.csv', 'recording_control.json', 'capture_bridge.json')


def sha256(blob):
    return hashlib.sha256(blob).hexdigest()


def _unpack(payload):
    """Validate the archive before allowing it to outlive the raw work directory."""
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names) or not set(REQUIRED).issubset(names):
            raise ValueError('original replay archive has missing or duplicate members')
        if set(names) - set(REQUIRED + OPTIONAL + ('manifest.json',)):
            raise ValueError('original replay archive has unexpected members')
        files = {name: archive.read(name) for name in names}
    manifest = json.loads(files.pop('manifest.json'))
    if manifest.get('schema_version') != 1 or set(manifest['files']) != set(files):
        raise ValueError('invalid original replay manifest')
    for name, blob in files.items():
        if manifest['files'][name] != {'sha256': sha256(blob), 'bytes': len(blob)}:
            raise ValueError('original replay member checksum mismatch: ' + name)
    bridge_meta = json.loads(files['bridge.json'])
    saved = json.loads(files['bridge_saved.json'])
    world_status = json.loads(files['world_trace.json'])
    if (not bridge_meta.get('recording_finalized') or not saved.get('recording_finalized')
            or saved.get('recording_error') or not world_status.get('recording_finalized')):
        raise ValueError('original replay contains unfinalized recording evidence')
    paths = json.loads(files['world_paths.json'])
    with np.load(io.BytesIO(files['world_trace.npz']), allow_pickle=False) as world:
        n = len(world['time'])
        if (n < 2 or n != world_status['samples']
                or world['object_transform_xyzw'].shape != (n, len(paths['body_paths']), 7)
                or world['body_link_pos_w'].shape != (n, len(paths['robot_body_names']), 3)
                or world['body_link_quat_wxyz'].shape != (n, len(paths['robot_body_names']), 4)
                or world['foot_net_contact_force_w'].shape != (n, 2, 3)
                or not all(np.isfinite(world[k]).all() for k in world.files)
                or np.any(np.diff(world['time']) <= 0)
                or np.any(np.diff(world['host_monotonic_ns']) <= 0)):
            raise ValueError('invalid original replay world shapes, values or clocks')
    with np.load(io.BytesIO(files['bridge.npz']), allow_pickle=False) as bridge:
        m = len(bridge['time'])
        for key, shape in [('root_pos', (m, 3)), ('root_quat_wxyz', (m, 4)),
                           ('root_lin_vel_w', (m, 3)), ('joint_pos', (m, 29)),
                           ('left_hand_joint_pos', (m, 11)), ('right_hand_joint_pos', (m, 11))]:
            if bridge[key].shape != shape or not np.isfinite(bridge[key]).all():
                raise ValueError('invalid original replay measured state: ' + key)
    summary = {'world_samples': n, 'bridge_samples': m,
               'scene_bodies': len(paths['body_paths']),
               'basket_paths': [p for p in paths['body_paths'] if 'basket' in p.lower()],
               'original_world_positions_retained': True,
               'passive_hand_states_retained': True, 'foot_contacts_retained': True}
    return files, summary


def embed(run, parquet_path):
    """Attach to a copied export only; the original raw recording stays untouched."""
    import pyarrow.parquet as pq
    run, parquet_path = Path(run), Path(parquet_path)
    files = {name: (run / name).read_bytes() for name in REQUIRED}
    files.update({name: (run / name).read_bytes() for name in OPTIONAL if (run / name).is_file()})
    world_phase = json.loads(files['world_trace.json']).get('sampling_phase', 'post_action')
    manifest = {'schema_version': 1, 'kind': 'original_capture_not_resimulation',
                'world_sampling_phase': world_phase,
                'clock': ('original pre-action world and pre-action Bridge clocks; use retained alignment'
                          if world_phase == 'pre_action' else
                          'original post-step world and pre-step Bridge clocks; use retained alignment'),
                'files': {name: {'sha256': sha256(blob), 'bytes': len(blob)} for name, blob in files.items()}}
    buffer = io.BytesIO()
    # NPZ members are already compressed. Avoid another expensive compression pass.
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_STORED) as archive:
        for name, blob in files.items():
            archive.writestr(name, blob)
        archive.writestr('manifest.json', json.dumps(manifest, sort_keys=True))
    payload = buffer.getvalue()
    _, summary = _unpack(payload)
    original = pq.read_table(parquet_path)
    metadata = dict(original.schema.metadata or {})
    if KEY in metadata:
        raise ValueError('original replay metadata already present; refusing replacement')
    metadata[KEY] = payload
    temporary = parquet_path.with_name(parquet_path.name + '.replay.tmp')
    try:
        pq.write_table(original.replace_schema_metadata(metadata), temporary, compression='zstd')
        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        rewritten = pq.read_table(temporary)
        if not original.equals(rewritten, check_metadata=False):
            raise ValueError('adding replay metadata changed training rows')
        temporary.replace(parquet_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {'schema_version': 1, 'storage': 'parquet_schema_metadata',
            'key': KEY.decode(), 'sha256': sha256(payload), 'bytes': len(payload),
            'data_path': 'data/chunk-000/episode_000000.parquet', **summary}


def validate(sample):
    import pyarrow.parquet as pq
    sample = Path(sample)
    info = json.loads((sample / 'meta/info.json').read_text())
    world = info['script_config'].get('world_recording', {})
    description = world.get('embedded_replay')
    if description is None:
        if world.get('retained_in_submission'):
            raise ValueError('world retention is claimed without an embedded replay')
        return None  # Historical seven-file samples remain readable, without inventing poses.
    path = sample / info['data_path'].format(episode_chunk=0, episode_index=0)
    payload = (pq.read_schema(path).metadata or {}).get(KEY)
    if (not payload or sha256(payload) != description['sha256']
            or len(payload) != description['bytes'] or description['key'] != KEY.decode()):
        raise ValueError('embedded original replay missing or checksum mismatch')
    files, summary = _unpack(payload)
    if any(description.get(key) != value for key, value in summary.items()):
        raise ValueError('embedded original replay summary mismatch')
    return files, summary


def extract(sample, destination):
    """Create a new diagnostic render input; never modify an existing sample."""
    checked = validate(sample)
    if checked is None:
        raise ValueError('this historical sample did not retain original world replay evidence')
    files, summary = checked
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for name, blob in files.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(blob)
            stream.flush()
            os.fsync(stream.fileno())
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check', 'extract'))
    parser.add_argument('sample', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.mode == 'extract':
        if args.output is None:
            parser.error('extract requires --output pointing to a new directory')
        result = extract(args.sample, args.output)
    else:
        checked = validate(args.sample)
        result = checked[1] if checked else {'original_world_available': False}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
