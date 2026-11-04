#!/usr/bin/env python3
"""Offline SONIC robot-motion decoder and read-only kinematic comparison.

No DDS, Isaac timeline, robot command publisher, or source-file mutation.
The motion decoder reconstructs a reference, not the realized physics trajectory.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from capture_manifest import fingerprint
from capture_utils import atomic_json

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1]/'assets/sonic_release_reference/decoders.npz'


def unpack_motion_output(output, order):
    """Preserve SONIC release's historical position/velocity temporal layout.

    Official command_multi_future concatenates ALL future q then ALL future dq,
    then reshapes that 580-vector to [10,58]. It is NOT [q29,dq29] per frame.
    UniversalTokenModule interleaves that 58-vector and rotation6D per frame.
    See gear_sonic/trl/losses/token_losses.py:decoder_output_to_egocentric_transforms.
    """
    output = np.asarray(output)
    order = np.asarray(order)
    if output.ndim != 2 or output.shape[1] != 640 or not np.isfinite(output).all():
        raise ValueError('motion decoder must return finite [N,640]')
    if sorted(order.tolist()) != list(range(29)):
        raise ValueError('invalid joint permutation')
    frames = output.reshape(-1, 10, 64)
    commands = frames[..., :58].reshape(-1, 580)
    q = commands[:, :290].reshape(-1, 10, 29)[..., order]
    dq = commands[:, 290:].reshape(-1, 10, 29)[..., order]
    return q, dq, frames[..., 58:]


def rotation6d_matrix(rotation6d):
    """SONIC 6D = first TWO COLUMNS, row-major, unlike column-stacked 6D."""
    cols = np.asarray(rotation6d, dtype=np.float64).reshape(*rotation6d.shape[:-1], 3, 2)
    x = cols[..., 0]
    xnorm = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(xnorm < 1e-8):
        raise ValueError('degenerate rotation6d first column')
    x = x/xnorm
    y = cols[..., 1]-np.sum(x*cols[..., 1], axis=-1, keepdims=True)*x
    ynorm = np.linalg.norm(y, axis=-1, keepdims=True)
    if np.any(ynorm < 1e-8):
        raise ValueError('degenerate rotation6d second column')
    y = y/ynorm
    return np.stack((x, y, np.cross(x, y)), axis=-1)


def run_mlp(token, weights, name='g1_kin'):
    import torch
    torch.set_num_threads(2)
    prefix = f'actor_module.decoders.{name}.module.'
    layers = sorted(int(k[len(prefix):].split('.')[0]) for k in weights
                    if k.startswith(prefix) and k.endswith('.weight'))
    expected = [0, 2, 4, 6, 8] if name == 'g1_kin' else [0, 2, 4, 6, 8, 10, 12]
    if layers != expected:
        raise ValueError(f'unexpected {name} layers: {layers}')
    values = []
    with torch.inference_mode():
        for batch in range(0, len(token), 256):
            x = torch.from_numpy(np.asarray(token[batch:batch+256], dtype=np.float32))
            for layer in layers:
                x = torch.nn.functional.linear(x,
                    torch.from_numpy(weights[f'{prefix}{layer}.weight'].copy()),
                    torch.from_numpy(weights[f'{prefix}{layer}.bias'].copy()))
                if layer != layers[-1]:
                    x = torch.nn.functional.silu(x)
            values.append(x.numpy())
    return np.concatenate(values)


def verify_weights(weights_path, onnx_path):
    """Match all control-decoder tensors to capture-time ONNX, not filenames."""
    import onnx
    from onnx import numpy_helper
    manifest_path = weights_path.with_suffix('.json')
    manifest = json.loads(manifest_path.read_text())
    if fingerprint(weights_path)['sha256'] != manifest['npz_sha256']:
        raise ValueError('decoder artifact does not match its SHA256 manifest')
    with np.load(weights_path, allow_pickle=False) as archive:
        weights = {key: archive[key] for key in archive.files}
    for key, value in weights.items():
        if hashlib.sha256(value.astype('<f4', copy=False).tobytes()).hexdigest() != manifest['members'][key]['sha256']:
            raise ValueError(f'selected tensor hash mismatch: {key}')
    initializers = [(t.name, numpy_helper.to_array(t)) for t in onnx.load(str(onnx_path)).graph.initializer]
    matched = {}
    for key, value in weights.items():
        if '.g1_dyn.' not in key:
            continue
        candidates = [name for name, array in initializers if
                      (array.shape == value.shape and np.array_equal(array, value)) or
                      (key.endswith('.weight') and array.ndim == 2 and
                       array.T.shape == value.shape and np.array_equal(array.T, value))]
        if len(candidates) != 1:
            raise ValueError(f'official checkpoint / captured ONNX mismatch: {key}')
        matched[key] = candidates[0]
    if len(matched) != 14:
        raise ValueError('all 14 control-decoder tensors must match')
    return weights, {'artifact': fingerprint(weights_path), 'source': manifest,
                     'capture_control_decoder_exact_weight_matches': matched}


def audit_dataset_tokens(dataset, states):
    import pyarrow.parquet as pq
    paths = sorted((dataset/'data').rglob('*.parquet'))
    if not paths:
        raise ValueError(f'no Parquet in dataset: {dataset}')
    by_index = {int(s['index']): s for s in states}
    seen, errors = [], []
    for path in paths:
        table = pq.read_table(path, columns=['action.motion_token', 'observation.control_cycle_index',
                                            'observation.body_monotonic_ns'])
        token = np.asarray(table['action.motion_token'].to_pylist(), dtype=np.float32)
        index = np.asarray(table['observation.control_cycle_index'].to_pylist()).reshape(-1)
        times = np.asarray(table['observation.body_monotonic_ns'].to_pylist()).reshape(-1)
        if token.shape != (len(index), 64) or not np.isfinite(token).all():
            raise ValueError('invalid exported motion tokens')
        if any(int(i) not in by_index for i in index):
            raise ValueError('dataset references an unrecorded cycle')
        expected = np.array([by_index[int(i)]['token_state'] for i in index], dtype=np.float32)
        expected_times = np.array([by_index[int(i)]['state_monotonic_ns'] for i in index], dtype=np.int64)
        if not np.array_equal(times, expected_times):
            raise ValueError('dataset cycle/timestamp alignment mismatch')
        errors.append(float(np.max(np.abs(token-expected))))
        seen.extend(index.tolist())
    if len(set(seen)) != len(seen) or any(b <= a for a, b in zip(seen, seen[1:])):
        raise ValueError('duplicate or out-of-order dataset cycles')
    missing = sorted(set(by_index)-set(seen))
    return {'dataset': str(dataset), 'rows': len(seen), 'token_max_abs_error': max(errors),
            'tokens_exact': max(errors) == 0, 'cycle_timestamps_exact': True,
            'source_cycles_not_exported': missing, 'parquet_files': [fingerprint(p) for p in paths]}


def motion_audit(run, out, states, order, onnx_path, weights_path=DEFAULT_WEIGHTS, render=True, fps=20):
    from scipy.spatial.transform import Rotation
    from prepare_trajectory import forward_kinematics
    from sonic_contract import SONIC_CPP_MOTOR_ORDER
    from validate_sonic_tokens import error_stats, captured_file
    weights_path = Path(weights_path).resolve()
    weights, provenance = verify_weights(weights_path, onnx_path)
    token = np.asarray([s['token_state'] for s in states], dtype=np.float32)
    output = run_mlp(token, weights)
    q, dq, rot6d = unpack_motion_output(output, order)
    relative_rot = rotation6d_matrix(rot6d)
    actual_q = np.array([s['body_q'] for s in states], dtype=np.float64)
    actual_base = np.array([s['base_quat'] for s in states])
    actual_rot = Rotation.from_quat(actual_base[:, [1, 2, 3, 0]]).as_matrix()
    ref_rot = np.einsum('nij,nhjk->nhik', actual_rot, relative_rot)
    ref_quat_xyzw = Rotation.from_matrix(ref_rot.reshape(-1, 3, 3)).as_quat().reshape(-1, 10, 4)
    ref_quat = ref_quat_xyzw[..., [3, 0, 1, 2]]
    times_ns = np.array([s['state_monotonic_ns'] for s in states], dtype=np.int64)
    t = (times_ns-times_ns[0])/1e9
    pose = np.array([s['encoder_mode'] == 2 for s in states])
    if not np.any(pose):
        raise ValueError('capture contains no SONIC POSE cycles')
    bridge_meta = json.loads((run/'bridge.json').read_text())
    bridge_manifest = json.loads((run/'capture_bridge.json').read_text())
    urdf, urdf_entry = captured_file(run, bridge_manifest, 'main.urdf', True)
    if tuple(bridge_meta['motor_joint_names']) != SONIC_CPP_MOTOR_ORDER:
        raise ValueError('unexpected Bridge motor order')
    for component in ['bridge', 'sonic']:
        boot = json.loads((run/f'capture_{component}.json').read_text()).get('host_boot_id')
        if not boot or (component == 'sonic' and boot != bridge_boot):
            raise ValueError('cannot establish a shared same-boot monotonic clock')
        bridge_boot = boot
    with np.load(run/'bridge.npz', allow_pickle=False) as bridge:
        bridge_time = bridge['state_monotonic_ns']
        if np.any(np.diff(bridge_time) <= 0):
            raise ValueError('Bridge timestamps must increase')
        bridge_rows = np.searchsorted(bridge_time, times_ns, side='right')-1
        if np.any(bridge_rows < 0):
            raise ValueError('SONIC source precedes Bridge recording')
        bridge_age = (times_ns-bridge_time[bridge_rows])/1e9
        if np.any(bridge_age > .1):
            raise ValueError('cannot causally align Bridge root translation within 100ms')
        root_pos = bridge['root_pos'][bridge_rows].copy()
        first_physical_ns = int(bridge_time[np.flatnonzero(~bridge['startup_waiting'].astype(bool))[0]])
    actual_fk, links = forward_kinematics(actual_q, root_pos, actual_base, urdf)
    reference_fk, _ = forward_kinematics(q[:, 0], root_pos, ref_quat[:, 0], urdf)
    zero_root = np.zeros_like(root_pos)
    identity_quat = np.zeros_like(actual_base); identity_quat[:, 0] = 1
    actual_local, _ = forward_kinematics(actual_q, zero_root, identity_quat, urdf)
    ref_local, _ = forward_kinematics(q[:, 0], zero_root, identity_quat, urdf)
    per_joint = np.sqrt(np.mean((q[pose, 0]-actual_q[pose])**2, axis=0))
    # Horizon comparisons use actual timestamps, do not assume a gap-free clock.
    horizons = []
    for h in range(10):
        query = times_ns + h*100_000_000
        right = np.clip(np.searchsorted(times_ns, query), 0, len(times_ns)-1)
        left = np.maximum(right-1, 0)
        nearest = np.where(abs(times_ns[left]-query) <= abs(times_ns[right]-query), left, right)
        aligned = pose & pose[nearest] & (abs(times_ns[nearest]-query) <= 10_000_000)
        # Require continuous POSE mode across the horizon, not across a transition.
        bad = np.r_[0, np.cumsum(~pose)]
        aligned &= (bad[nearest+1]-bad[np.arange(len(pose))]) == 0
        horizons.append({'nominal_horizon_s': h/10, 'paired_pose_cycles': int(aligned.sum()),
                         'q_difference_rad': error_stats(q[aligned, h], actual_q[nearest[aligned]])
                         if aligned.any() else None})
    limits = bridge_meta['position_limits_motor_order']
    low, high = np.array(limits['low']), np.array(limits['high'])
    outside = (q < low) | (q > high)
    exports = [audit_dataset_tokens(run, states)]
    mentor = run.with_name(run.name+'_mentor_sample')
    if (mentor/'data').is_dir():
        exports.append(audit_dataset_tokens(mentor, states))
    if not all(x['tokens_exact'] for x in exports):
        raise ValueError('dataset token values differ from lossless source')
    by_index = {s['index']: s for s in states}
    for exported in exports:
        exported['omitted_source_cycle_details'] = [
            {'cycle': i, 'before_first_physical_bridge_state': by_index[i]['state_monotonic_ns'] < first_physical_ns,
             'previous_action_valid': bool(by_index[i]['previous_body_action_valid'])}
            for i in exported['source_cycles_not_exported']]
    abs_difference = np.abs(q[:, 0]-actual_q)
    masked = np.where(pose[:, None], abs_difference, -1)
    worst_row, worst_joint = np.unravel_index(np.argmax(masked), masked.shape)
    since_entry = np.full(len(t), np.nan)
    entry_time = None
    for i in range(len(t)):
        if pose[i]:
            if i == 0 or not pose[i-1]:
                entry_time = t[i]
            since_entry[i] = t[i]-entry_time
    sustained = pose & (since_entry >= 1.)
    trajectory = out/'robot_motion_decoder.npz'
    np.savez_compressed(trajectory, token=token, cycle_index=np.array([s['index'] for s in states]),
        state_monotonic_ns=times_ns, time_s=t, pose_mode=pose, raw_decoder_output=output,
        q_reference_hardware=q, dq_reference_hardware=dq, relative_rotation6d=rot6d,
        root_quat_reference_wxyz=ref_quat, q_recorded_hardware=actual_q,
        root_quat_recorded_wxyz=actual_base, root_pos_recorded_m=root_pos,
        root_translation_bridge_row=bridge_rows, root_translation_age_s=bridge_age,
        future_offsets_s=np.arange(10)*.1, motor_joint_names=np.array(SONIC_CPP_MOTOR_ORDER),
        joint_link_names=np.array(links),
        link_pos_reference_w_m=reference_fk['joint_link_pos_w_m'],
        link_pos_recorded_w_m=actual_fk['joint_link_pos_w_m'])
    report = {'status': 'decoded_and_compared', 'decoder': 'official g1_kin, token-only MLP',
        'source_cycles': len(states), 'pose_cycles': int(pose.sum()),
        'token_dimension': 64, 'future_frames': 10, 'future_offsets_s': (np.arange(10)*.1).tolist(),
        'motion_layout': 'all q[10,29], then all dq[10,29]; historical SONIC command flattening restored',
        'root_orientation': 'recorded pelvis orientation multiplied by decoded relative first-two-column 6D rotation',
        'root_translation': 'NOT decoded. Same preceding recorded Bridge translation used for both visualizations.',
        'root_translation_max_alignment_age_s': float(bridge_age.max()),
        'kinematic_urdf': urdf_entry,
        'pose_q_reference_vs_actual_rad': error_stats(q[pose, 0], actual_q[pose]),
        'pose_absolute_difference_percentiles_rad': dict(zip(['p50','p90','p95','p99'],
            np.quantile(abs_difference[pose], [.5,.9,.95,.99]).tolist())),
        'largest_pose_joint_difference': {'source_cycle': int(states[worst_row]['index']),
            'time_from_capture_start_s': float(t[worst_row]),
            'time_from_pose_entry_s': float(since_entry[worst_row]),
            'joint': SONIC_CPP_MOTOR_ORDER[worst_joint],
            'absolute_difference_rad': float(abs_difference[worst_row, worst_joint])},
        'pose_at_least_1s_after_each_mode_entry': {
            'note': 'Additional transition diagnostic only; all POSE frames remain in primary metrics.',
            'cycles': int(sustained.sum()),
            'q_difference_rad': error_stats(q[sustained, 0], actual_q[sustained]) if sustained.any() else None},
        'per_joint_pose_rmse_rad': dict(zip(SONIC_CPP_MOTOR_ORDER, per_joint.tolist())),
        'horizon_reference_vs_later_actual': horizons,
        'pose_current_reference_out_of_limit_values': int(outside[pose, 0].sum()),
        'pose_all_future_reference_out_of_limit_values': int(outside[pose].sum()),
        'decoded_finite': bool(np.isfinite(output).all()),
        'export_token_alignment': exports, 'weights': provenance,
        'limitations': ['Motion reconstruction / actual tracking difference has no equality acceptance threshold.',
                       'Auxiliary motion decoder is not a physics simulator; these references need not equal actual joint states.',
                       'No root translation, Revo2 fingers, camera images, contact forces or task success can be recovered from this 64D body token.',
                       'This does not approve training quality, contact accuracy, task success, or full closed-loop reproducibility.']}
    atomic_json(out/'robot_motion_decoder_report.json', report)
    make_plots(out, t, pose, q[:, 0], actual_q, SONIC_CPP_MOTOR_ORDER, horizons)
    if render:
        report['preview'] = render_comparison(out, t, pose, root_pos,
            reference_fk['joint_link_pos_w_m'], actual_fk['joint_link_pos_w_m'],
            ref_local['joint_link_pos_w_m'], actual_local['joint_link_pos_w_m'],
            links, urdf, fps)
        atomic_json(out/'robot_motion_decoder_report.json', report)
    return report


def make_plots(out, t, pose, ref_q, actual_q, names, horizons):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(5, 6, figsize=(18, 12), sharex=True, layout='constrained')
    ref = ref_q.copy(); actual = actual_q.copy()
    ref[~pose] = np.nan; actual[~pose] = np.nan
    for j, ax in enumerate(axes.flat):
        if j >= 29:
            ax.axis('off'); continue
        ax.plot(t, actual[:, j], label='Recorded actual', color='#176b9d', linewidth=.8)
        ax.plot(t, ref[:, j], label='Token -> motion reference', color='#d56c24', linewidth=.8, alpha=.9)
        ax.set_title(names[j].replace('_joint', ''), fontsize=9)
        ax.grid(alpha=.2)
        if j >= 24: ax.set_xlabel('Capture time (s)')
        if j % 6 == 0: ax.set_ylabel('rad')
    axes.flat[0].legend(fontsize=7)
    fig.suptitle('SONIC POSE: decoded current reference vs recorded actual (29 DOF)\nReconstruction + tracking difference; not an equality or task-success test')
    fig.savefig(out/'joint_trajectory_comparison.png', dpi=120)
    plt.close(fig)


def render_comparison(out, t, pose, root, ref_world, actual_world,
                      ref_local, actual_local, links, urdf, fps):
    """Paired URDF FK skeleton at physical capture speed, no physics execution."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    from render_trajectory_preview import skeleton_edges
    edges = skeleton_edges(urdf, links)
    ref_world = np.concatenate((root[:, None], ref_world), axis=1)
    actual_world = np.concatenate((root[:, None], actual_world), axis=1)
    zeros = np.zeros_like(root)[:, None]
    ref_local = np.concatenate((zeros, ref_local), axis=1)
    actual_local = np.concatenate((zeros, actual_local), axis=1)
    centers = root.copy(); centers[:, 2] = 0
    rw, aw = ref_world-centers[:, None], actual_world-centers[:, None]
    fig = plt.figure(figsize=(12.8, 6.4), layout='constrained')
    axes = [fig.add_subplot(121, projection='3d'), fig.add_subplot(122, projection='3d')]
    fig.suptitle('SONIC Universal Token -> Robot Motion Decoder / offline kinematics\nBlue: recorded actual | Orange: decoded current motion reference', fontsize=13)
    groups = []
    for ax, title, points1, points2 in zip(axes,
            ['Recorded root translation shared; decoded root orientation',
             'Pelvis-local joint motion (base orientation removed)'], [rw, ref_local], [aw, actual_local]):
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-.85, .85); ax.set_ylim(-.85, .85)
        ax.set_zlim((0, 1.7) if ax is axes[0] else (-.85, .85))
        ax.set_box_aspect((1.7, 1.7, 1.7))
        ax.view_init(elev=15, azim=-55)
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
        for points, color, style in [(points2, '#176b9d', '-'), (points1, '#d56c24', '--')]:
            lines = [ax.plot([], [], [], color=color, ls=style, lw=2.2, alpha=.9)[0] for _ in edges]
            groups.append((points, lines))
    label = axes[0].text2D(.01, .95, '', transform=axes[0].transAxes)
    fig.text(.5, .01, 'Root translation comes from the recording. Body-only reconstruction; no fingers, objects, camera or contact replay.',
             ha='center', fontsize=9)
    video_t = np.arange(int(np.ceil(t[-1]*fps))+1)/fps
    rows = np.clip(np.searchsorted(t, video_t, side='right')-1, 0, len(t)-1)
    writer = FFMpegWriter(fps=fps, codec='libx264', bitrate=2000, extra_args=['-pix_fmt', 'yuv420p'])
    snapshot_rows = [int(np.flatnonzero(pose)[int(f*(pose.sum()-1))]) for f in (.15, .5, .85)]
    saved = set()
    with writer.saving(fig, str(out/'robot_motion_comparison.mp4'), dpi=100):
        for frame, row in enumerate(rows):
            for points, lines in groups:
                for line, (a, b) in zip(lines, edges):
                    line.set_data_3d(points[row, [a, b], 0], points[row, [a, b], 1], points[row, [a, b], 2])
            stale = min(video_t[frame], t[-1])-t[row] > .05
            label.set_text(f'{min(video_t[frame], t[-1]):.2f} / {t[-1]:.2f} s  '
                           f'{"POSE" if pose[row] else "WARMUP / NOT POSE"}' + ('  CAPTURE GAP' if stale else ''))
            for i, target in enumerate(snapshot_rows):
                if row >= target and i not in saved:
                    fig.savefig(out/f'comparison_pose_{i}.png', dpi=100)
                    saved.add(i)
            writer.grab_frame()
            if frame % 200 == 0:
                print(f'RENDER {frame}/{len(rows)}', flush=True)
    plt.close(fig)
    return {'file': 'robot_motion_comparison.mp4', 'fps': fps, 'frames': len(rows),
            'physical_duration_s': float(t[-1]), 'encoded_duration_s': len(rows)/fps,
            'last_source_row': int(rows[-1]), 'style': 'URDF FK skeleton, not a new Isaac physics rollout',
            'root_translation_shared_from_recording': True}
