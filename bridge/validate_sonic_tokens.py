#!/usr/bin/env python3
"""Offline SONIC token audit and robot-motion replay. Never publishes commands.

Rebuilds the C++ observation from lossless StateLogger records: each history
is flattened oldest-first in its own block, including the current state and
the previous raw action. No quaternion normalization or state interpolation.
Also decodes the official auxiliary robot-motion reference and compares it with
actual recorded joints; their difference is not an equality acceptance test.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import tempfile

import numpy as np
import yaml

from capture_manifest import fingerprint
from capture_utils import atomic_json
from sonic_debug_recorder import validate_recording

ROOT = Path(__file__).resolve().parents[1]


def captured_file(run, manifest, basename, snapshot=False):
    entries = [x for x in manifest['files'] if Path(x['path']).name == basename]
    if len(entries) != 1:
        raise ValueError(f'need exactly one captured {basename}')
    entry = entries[0]
    path = (run / entry['snapshot_relative_to_run'] if snapshot else Path(entry['path']))
    if fingerprint(path)['sha256'] != entry['sha256']:
        raise ValueError(f'capture-time SHA256 mismatch: {path}')
    return path, entry


def gravity_cpp(quat):
    """Exact quat_rotate_d(conjugate(q), [0,0,-1]), including zero padding.

    The C++ formula assumes unit quaternions but does not normalize input.
    A makeZeroEntry_ quaternion yields +Z, not zero or identity gravity.
    """
    q = np.asarray(quat, dtype=np.float64)
    w = q[..., :1]
    xyz = -q[..., 1:]
    v = np.broadcast_to([0., 0., -1.], xyz.shape)
    return v * (2*w*w-1) + np.cross(xyz, v)*w*2 + xyz*np.sum(xyz*v, axis=-1, keepdims=True)*2


def reconstruct_observations(states, config):
    fields = {'base_angular_velocity': ('base_ang_vel', 3),
              'body_joint_positions': ('body_q', 29),
              'body_joint_velocities': ('body_dq', 29),
              'last_actions': ('last_action', 29), 'gravity_dir': ('base_quat', 4)}
    n = len(states)
    arrays = {}
    for field, dim in [*fields.values(), ('token_state', 64)]:
        array = np.asarray([s[field] for s in states], dtype=np.float64)
        if array.shape != (n, dim) or not np.isfinite(array).all():
            raise ValueError(f'invalid recorded {field}: {array.shape}')
        arrays[field] = array
    blocks, layout = [], []
    offset = 0
    for observation in config['observations']:
        if not observation.get('enabled', True):
            continue
        name = observation['name']
        if name == 'token_state':
            block = arrays['token_state']
        else:
            match = re.fullmatch(r'his_(.+)_(\d+)frame_step(\d+)', name)
            if not match or match[1] not in fields:
                raise ValueError(f'unsupported captured observation: {name}')
            kind, frames, step = match[1], int(match[2]), int(match[3])
            if frames < 1 or step < 1:
                raise ValueError('invalid observation history')
            key, dim = fields[kind]
            rows = np.arange(n)[:, None] - np.arange(frames-1, -1, -1)[None, :]*step
            history = arrays[key][np.maximum(rows, 0)].copy()
            history[rows < 0] = 0.
            if kind == 'gravity_dir':
                history = gravity_cpp(history)
            block = history.reshape(n, -1)
        layout.append({'name': name, 'offset': offset, 'dimension': block.shape[1]})
        offset += block.shape[1]
        blocks.append(block)
    return np.concatenate(blocks, axis=1).astype(np.float32), layout


def action_mapping(header):
    text = re.sub(r'/\*.*?\*/|//[^\n]*', '', Path(header).read_text(), flags=re.S)
    match = re.search(r'isaaclab_to_mujoco\s*=\s*\{([^}]+)\}', text)
    if not match:
        raise ValueError('captured action permutation missing')
    order = np.array([int(x.strip()) for x in match[1].split(',') if x.strip()])
    if sorted(order.tolist()) != list(range(29)):
        raise ValueError('invalid captured action permutation')
    return order


def restore_policy_states(states, default, scale, order):
    """Undo ZMQOutputHandler's hardware-order offsets/scales before history.

    Wire body_q is absolute hardware q; wire last_action is scaled hardware
    equilibrium. Neither can be fed directly to the policy observation.
    """
    inverse = np.argsort(order)
    result = []
    for state in states:
        row = dict(state)
        row['body_q'] = (np.asarray(state['body_q'])-default)[inverse]
        row['body_dq'] = np.asarray(state['body_dq'])[inverse]
        row['last_action'] = ((np.asarray(state['last_action'])-default)/scale)[inverse]
        result.append(row)
    return result


def error_stats(a, b):
    difference = np.asarray(a, dtype=float)-np.asarray(b, dtype=float)
    if difference.size == 0 or not np.isfinite(difference).all():
        raise ValueError('comparison requires finite paired samples')
    return {'rmse': float(np.sqrt(np.mean(difference*difference))),
            'mae': float(np.mean(np.abs(difference))), 'max_abs': float(np.abs(difference).max())}


def control_audit(run, out, tolerance_rad=.005):
    import onnxruntime as ort
    manifest = json.loads((run/'capture_sonic.json').read_text())
    model, model_entry = captured_file(run, manifest, 'model_decoder.onnx')
    config_path, _ = captured_file(run, manifest, 'observation_config.yaml', True)
    header, _ = captured_file(run, manifest, 'policy_parameters.hpp', True)
    config = yaml.safe_load(config_path.read_text())
    summary, records = validate_recording(run/'sonic_debug.msgpack')
    configs = [x['payload'] for x in records if x['topic'] == 'robot_config']
    if not configs:
        raise ValueError('missing capture-time robot configuration')
    if any(c != configs[0] for c in configs):
        raise ValueError('robot configuration changed during recording')
    states = [x['payload'] for x in records if x['topic'] == 'g1_debug']
    if not states or states[0]['index'] != 0:
        raise ValueError('exact control history requires recording from control cycle zero')
    order = action_mapping(header)
    default = np.array(json.loads(configs[0]['body_recording_default_q_hardware_json']))
    scale = np.array(json.loads(configs[0]['body_recording_action_scale_hardware_json']))
    if default.shape != (29,) or scale.shape != (29,) or np.any(scale <= 0):
        raise ValueError('invalid capture-time action scale/default')
    policy_states = restore_policy_states(states, default, scale, order)
    observations, layout = reconstruct_observations(policy_states, config)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model), sess_options=options, providers=['CPUExecutionProvider'])
    inputs = session.get_inputs()
    if len(inputs) != 1 or inputs[0].shape != [1, observations.shape[1]]:
        raise ValueError(f'decoder input differs from captured observation layout: {inputs[0].shape}')
    decoded = np.array([session.run(None, {inputs[0].name: row[None]})[0][0]
                        for row in observations], dtype=np.float32)
    if decoded.shape != (len(states), 29) or not np.isfinite(decoded).all():
        raise ValueError('invalid decoder actions')
    recorded = np.asarray([s['body_action'] for s in states], dtype=np.float32)
    targets = (default + decoded[:, order]*scale).astype(np.float32)
    errors = error_stats(targets, recorded)
    np.savez_compressed(out/'control_decoder.npz', observation=observations, raw_action=decoded,
                        decoded_q_hardware=targets, recorded_q_target_hardware=recorded,
                        token=np.array([s['token_state'] for s in states]),
                        state_monotonic_ns=np.array([s['state_monotonic_ns'] for s in states],dtype=np.int64),
                        cycle_index=np.array([s['index'] for s in states],dtype=np.int64),
                        encoder_mode=np.array([s['encoder_mode'] for s in states]),
                        recorded_body_q_hardware=np.array([s['body_q'] for s in states]))
    report = {'status': 'pass' if errors['max_abs'] <= tolerance_rad else 'mismatch',
              'mode': 'teacher-forced control decoder; recorded state/history, not a physics rollout',
              'recording': summary, 'model': model_entry, 'observation_layout': layout,
              'observation_dimension': observations.shape[1], 'action_dimension': 29,
              'target_error_rad': errors, 'max_abs_tolerance_rad': tolerance_rad,
              'comparison_note': 'CPU ONNX versus capture-time TensorRT; numerical equality is tolerance-based'}
    atomic_json(out/'control_decoder_report.json', report)
    return report


def write_summary(out, run, control, motion, unchanged):
    lines = ['# SONIC Token Decoder Validation', '', f'Source capture: `{run}`', '',
             f'- Control decoder: **{control["status"]}**; {control["recording"]["state_action_cycles"]} cycles.',
             f'- Max joint-target error: {control["target_error_rad"]["max_abs"]:.9g} rad.',
             f'- Original source files unchanged: **{unchanged}**.', '']
    if motion is not None:
        stats = motion['pose_q_reference_vs_actual_rad']
        lines += ['## Robot Motion Decoder', '',
            f'- Official token-only `g1_kin`: {motion["source_cycles"]} tokens; {motion["pose_cycles"]} POSE cycles.',
            '- Every 64D token reconstructs 10 future frames: 0.0, 0.1, ..., 0.9 seconds.',
            '- All 14 control-network tensors in the official checkpoint match the capture-time ONNX exactly.',
            f'- POSE current reference / actual joint difference: RMSE **{stats["rmse"]:.6f} rad**, max **{stats["max_abs"]:.6f} rad**.',
            '- This difference includes reconstruction and controller tracking; it is not an equality failure.',
            '- [29-joint curves](joint_trajectory_comparison.png)',
            '- [Full motion report](robot_motion_decoder_report.json)',
            '- [Decoded trajectory arrays](robot_motion_decoder.npz)', '', '## Dataset Token Checks', '']
        if 'preview' in motion:
            lines.insert(lines.index('## Dataset Token Checks')-1, '- [Replay video](robot_motion_comparison.mp4)')
        for exported in motion['export_token_alignment']:
            lines += [f'- `{exported["dataset"]}`: {exported["rows"]} rows; exact token and cycle-time matches: {exported["tokens_exact"]}; source cycles omitted: {exported["source_cycles_not_exported"]}.']
        lines += ['', '## Scope', '',
                  'Video uses URDF forward kinematics. Orange is the decoded reference; blue is recorded actual state.',
                  'The two streams share recorded root translation because this decoder does not output world translation.',
                  'Decoded relative root rotation is composed with recorded pelvis orientation. The right view removes base orientation.',
                  'All 10 future frames are saved in NPZ; continuous video uses the current frame from every token.',
                  'Revo2 fingers, object motion, camera content, contacts and task success are not reconstructed by this body token.',
                  'No DDS publication or live robot / simulator actuation is performed.',
                  'Passing these checks supports token preservation, checkpoint compatibility and deterministic control reconstruction;',
                  'it does not alone certify that a demonstration is suitable for training.', '',
                  'Official references: https://arxiv.org/html/2511.07820v4 and https://github.com/NVlabs/GR00T-WholeBodyControl .']
    (out/'README.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--tolerance-rad', type=float, default=.005)
    from sonic_motion_replay import DEFAULT_WEIGHTS
    parser.add_argument('--motion-weights', type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument('--control-only', action='store_true', help='skip auxiliary motion decoder and video')
    parser.add_argument('--no-video', action='store_true', help='keep trajectories, metrics and plots without encoding MP4')
    parser.add_argument('--fps', type=int, default=20, choices=range(1, 61))
    args = parser.parse_args()
    if not np.isfinite(args.tolerance_rad) or args.tolerance_rad <= 0:
        parser.error('tolerance must be finite and positive')
    run = args.run.resolve()
    out = args.output.resolve() if args.output else Path(tempfile.mkdtemp(
        prefix=datetime.now().strftime('%Y%m%d_%H%M%S')+'_sonic_token_validation_',dir=ROOT/'outputs'))
    if args.output:
        out.mkdir(parents=True, exist_ok=False)
    print('OUTPUT', out, flush=True)
    sources = {name: fingerprint(run/name) for name in
               ['bridge.npz','bridge.json','sonic_debug.msgpack','sonic_debug.msgpack.json',
                'capture_sonic.json','capture_bridge.json','checksums.json']}
    for dataset in [run, run.with_name(run.name+'_mentor_sample')]:
        for folder in ['data', 'meta', 'videos']:
            for path in sorted((dataset/folder).rglob('*')):
                if path.is_file():
                    sources[str(path.resolve())] = fingerprint(path)
    atomic_json(out/'source.json', {'run': str(run), 'files': sources})
    try:
        report = control_audit(run, out, args.tolerance_rad)
        motion = None
        if not args.control_only:
            print('CONTROL COMPLETE; DECODING ROBOT MOTION', flush=True)
            from sonic_motion_replay import motion_audit
            manifest = json.loads((run/'capture_sonic.json').read_text())
            model, _ = captured_file(run, manifest, 'model_decoder.onnx')
            header, _ = captured_file(run, manifest, 'policy_parameters.hpp', True)
            _, records = validate_recording(run/'sonic_debug.msgpack')
            states = [x['payload'] for x in records if x['topic'] == 'g1_debug']
            motion = motion_audit(run, out, states, action_mapping(header), model,
                                  args.motion_weights, not args.no_video, args.fps)
        unchanged = all(fingerprint(run/k)['sha256'] == v['sha256'] for k,v in sources.items())
        report['source_files_unchanged'] = unchanged
        atomic_json(out/'control_decoder_report.json', report)
        if motion is not None:
            motion['source_files_unchanged'] = unchanged
            atomic_json(out/'robot_motion_decoder_report.json', motion)
        write_summary(out, run, report, motion, unchanged)
        artifacts = {p.name: fingerprint(p) for p in sorted(out.iterdir()) if p.is_file()}
        atomic_json(out/'validation_manifest.json', {'source_files_unchanged': unchanged, 'artifacts': artifacts,
                    'validation_code': [fingerprint(ROOT/p) for p in
                     ['validate_sonic_tokens.py','sonic_motion_replay.py','fetch_sonic_motion_decoder.py']]})
        print(json.dumps(report, indent=2), flush=True)
        if motion is not None:
            print('MOTION', json.dumps({k: motion[k] for k in ['status','pose_cycles','pose_q_reference_vs_actual_rad','source_files_unchanged']}), flush=True)
        if report['status'] != 'pass' or not report['source_files_unchanged']:
            raise SystemExit(2)
    except Exception as exc:
        atomic_json(out/'error.json', {'type': type(exc).__name__, 'error': str(exc)})
        raise


if __name__ == '__main__':
    main()
