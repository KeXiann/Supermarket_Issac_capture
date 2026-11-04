#!/usr/bin/env python3
"""Read-only training audit; saving a capture is not a training-quality approval.

Camera calibration is deliberately out of scope. Video freshness/alignment is
still checked. Thresholds are explicit local screening policy, not a claim of
Mentor acceptance or compatibility with an unprovided training checkpoint.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

import numpy as np

from capture_utils import atomic_json, sync_tree

ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_TASK = 'PICO SONIC POSE development capture'
POLICY = {'min_pose_valid_fraction': .95, 'min_full_history_fraction': .90,
          'max_repeated_body_source_fraction': .01, 'real_time_factor_range': [.95, 1.05],
          'max_human_age_s': .1, 'max_video_age_s': .1}


def assess(run, integrity_status=None):
    import pyarrow.parquet as pq
    run = Path(run).resolve()
    checks = []

    def check(name, passed, detail, severity='blocker'):
        checks.append({'name': name, 'passed': bool(passed), 'severity': severity, 'detail': detail})

    report = {'schema_version': 1, 'checked_utc': datetime.now(timezone.utc).isoformat(),
              'source_run': str(run), 'camera_calibration_excluded': True,
              'policy': POLICY, 'checks': checks, 'formal_collection_approved': False,
              'actual_mentor_training_step_tested': False,
              'scope': 'Revo2-aware imitation/adapter candidates; not BrainCo-equivalent or task-success certification'}
    if integrity_status is None:
        from capture_session import inspect
        integrity, _ = inspect(run)
        integrity_status = integrity['status']
    check('durable_capture_integrity', integrity_status == 'complete', integrity_status)
    required = ['session.json', 'meta/info.json', 'meta/tasks.jsonl', 'meta/bridge_alignment.npz',
                'training/manifest.json', 'training/semanticdpc_0826/manifest.json']
    missing = [name for name in required if not (run / name).is_file()]
    check('training_artifacts_present', not missing, missing)
    if missing:
        report.update(status='blocked_incomplete_capture', machine_quality_passed=False, metrics={})
        return report, {}
    read = lambda rel: json.loads((run / rel).read_text())
    session, info = read('session.json'), read('meta/info.json')
    config = info.get('script_config', {})
    task = json.loads((run / 'meta/tasks.jsonl').read_text().splitlines()[0])['task']
    report.update(task=task, input_kind=session.get('input_kind'), hand_domain='Revo2 simulator',
                  task_outcome=session.get('task_outcome', 'unreviewed'))
    check('live_human_demonstration', session.get('input_kind') == 'live_pico', session.get('input_kind'))
    check('explicit_task_instruction', task != DEVELOPMENT_TASK and session.get('task_source') == 'operator_before_capture'
          and session.get('task') == task, {'task': task, 'source': session.get('task_source')})
    check('body_protocol_v2', config.get('body_action_alignment_version') == 2, config.get('body_action_alignment_version'))
    availability = config.get('field_availability', {})
    unavailable = [key for key, value in availability.items() if not value.get('available')]
    report['unavailable_fields_exclude_from_training'] = unavailable
    needed = ['observation.state', 'observation.joint_velocity', 'action.wbc', 'action.motion_token',
              'teleop.smpl_joints', 'action.hand_joint_position']
    missing_signals = [key for key in needed if not availability.get(key, {}).get('available')]
    check('required_pose_signals_available', not missing_signals, missing_signals)
    path = run / info['data_path'].format(episode_chunk=0, episode_index=0)
    table = pq.read_table(path)
    n = len(table)

    def column(key):
        return np.asarray(table[key].to_pylist())

    row_finite = np.ones(n, dtype=bool)
    for key in table.column_names:
        values = np.isfinite(column(key))
        row_finite &= values.reshape(n, -1).all(axis=1)
    finite = bool(row_finite.all())
    check('finite_numeric_columns', finite, {'rows': n, 'columns': len(table.column_names)})
    with np.load(run / 'meta/bridge_alignment.npz') as a:
        human_valid = a['human_valid'].astype(bool)
        human_age = a['human_age_s'].copy()
        camera_age = a['camera_age_s'].copy()
        sim_time = a['sim_time_s'].copy()
        host_time = a['host_monotonic_s'].copy()
        camera_frame = a['camera_source_frame'].copy()
    body_valid = column('observation.body_valid').reshape(-1).astype(bool)
    pose_valid = (row_finite & body_valid & human_valid
                  & (human_age >= 0) & (human_age <= POLICY['max_human_age_s'])
                  & (column('teleop.stream_mode').reshape(-1) == 1)
                  & column('observation.hand_valid').astype(bool).all(axis=1)
                  & column('action.hand_valid').reshape(-1).astype(bool)
                  & (camera_age >= 0) & (camera_age <= POLICY['max_video_age_s']))
    pose_fraction = float(pose_valid.mean()) if n else 0.
    check('fresh_pose_hand_video_fraction', pose_fraction >= POLICY['min_pose_valid_fraction'], pose_fraction)
    lowtick = column('observation.lowstate_tick').reshape(-1)
    repeated_fraction = float((np.diff(lowtick) <= 0).mean()) if n > 1 else 1.
    check('fresh_body_source_each_cycle', repeated_fraction <= POLICY['max_repeated_body_source_fraction'], repeated_fraction)
    rtf = float((sim_time[-1]-sim_time[0])/(host_time[-1]-host_time[0])) if n > 1 and host_time[-1] > host_time[0] else 0.
    check('simulation_and_control_timebase', POLICY['real_time_factor_range'][0] <= rtf <= POLICY['real_time_factor_range'][1], rtf)
    sm = read('training/semanticdpc_0826/manifest.json')
    masks = {'source_row_index': np.arange(n, dtype=np.int64), 'revo2_pose_bc_valid': pose_valid,
             'body_sample_valid': np.zeros(n, bool), 'full_body_history_valid': np.zeros(n, bool),
             'visual_pose_full_history_valid': np.zeros(n, bool)}
    full = 0
    boundaries = {}
    longest = 0
    if sm.get('status') == 'complete':
        with np.load(run / 'training/semanticdpc_0826' / sm['artifact']) as a:
            valid, history = a['sample_valid'].astype(bool), a['history_mask'].astype(bool).all(axis=1)
            assert len(valid) == len(history) == n, 'converter row count differs from dataset'
            for key in ('state64', 'history10', 'action31'):
                check('finite_' + key, np.isfinite(a[key]).all(), list(a[key].shape))
            masks['body_sample_valid'] = valid
            masks['full_body_history_valid'] = valid & history
            # Every POSE/image row in the ten-frame context must be fresh.
            all_pose_context = np.convolve(pose_valid.astype(np.int32), np.ones(10, np.int32), mode='full')[:n] == 10
            masks['visual_pose_full_history_valid'] = valid & history & all_pose_context
            full = int((valid & history).sum())
            for key, count in zip(*np.unique(a['boundary_reason'], return_counts=True)):
                boundaries[str(key)] = int(count)
            for segment in np.unique(a['segment_id'][valid]):
                longest = max(longest, int(np.sum((a['segment_id'] == segment) & valid)))
    full_fraction = full / n if n else 0.
    check('full_ten_frame_body_history', sm.get('status') == 'complete' and full_fraction >= POLICY['min_full_history_fraction'],
          {'full_history_rows': full, 'fraction': full_fraction, 'boundary_reasons': boundaries})
    check('review_task_success', session.get('task_outcome') == 'success',
          'No automatic success label: review video/object trace, then select successful demonstrations.', 'review')
    bridge = read('bridge.json')
    host_span = float(host_time[-1] - host_time[0]) if n > 1 else 0.
    two_second_rates = []
    if host_span >= 2.:
        anchors = np.searchsorted(host_time, host_time[0] + np.arange(0., host_span, 2.))
        for left, right in zip(anchors[:-1], anchors[1:]):
            if host_time[right] > host_time[left]:
                two_second_rates.append(float((sim_time[right]-sim_time[left]) /
                    (host_time[right]-host_time[left]) / bridge.get('sim_dt', .005)))
    world_available = bool(bridge.get('world_trace_required')) and (run / 'world_trace.npz').is_file()
    check('physical_world_evidence', world_available,
          'Actual body/object poses and foot contacts retained; these do not prove a successful grasp.', 'warning')
    report['metrics'] = {'rows': n, 'pose_valid_rows': int(pose_valid.sum()), 'pose_valid_fraction': pose_fraction,
                         'full_history_rows': full, 'full_history_fraction': full_fraction,
                         'visual_pose_full_history_rows': int(masks['visual_pose_full_history_valid'].sum()),
                         'longest_valid_body_segment_rows': longest, 'repeated_body_source_fraction': repeated_fraction,
                         'real_time_factor_in_recording_window': rtf,
                         'measured_control_rows_per_wall_second': (n-1)/host_span if host_span > 0 else None,
                         'measured_physics_steps_per_wall_second': rtf/bridge.get('sim_dt', .005),
                         'physics_rate_2s_windows': two_second_rates,
                         'physics_rate_2s_window_min': min(two_second_rates) if two_second_rates else None,
                         'source_camera_fps_is_independent_of_export_fps': True,
                         'human_age_max_s': float(human_age.max()), 'video_age_max_s': float(camera_age.max()),
                         'unique_camera_source_frames': len(np.unique(camera_frame)), 'world_recorded': world_available}
    passed = all(c['passed'] for c in checks if c['severity'] == 'blocker')
    technical = all(c['passed'] for c in checks if c['severity'] == 'blocker'
                    and c['name'] not in ('live_human_demonstration', 'explicit_task_instruction'))
    diagnostic = session.get('input_kind') != 'live_pico'
    status = ('diagnostic_pipeline_passed_not_training_data' if technical else 'diagnostic_pipeline_failed') if diagnostic else (
        'candidate_requires_task_review' if passed else 'pilot_only_requires_correction')
    report.update(machine_quality_passed=passed, technical_quality_passed=technical,
                  status=status,
                  whole_unfiltered_mentor_sample_approved=False,
                  use_masks='source-row-indexed boolean masks in row_masks.npz; never bridge invalid/history segment boundaries',
                  hand_adaptation_required=True, brainco_equivalent=False)
    return report, masks


def write_assessment(run, output, integrity_status=None):
    report, masks = assess(run, integrity_status)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if masks:
        with (output / 'row_masks.npz').open('xb') as stream:
            np.savez_compressed(stream, **masks)
    atomic_json(output / 'report.json', report)
    sync_tree(output)
    return report


def ensure_capture_assessment(run, output, integrity_status='complete'):
    """Resume finalization without replacing an already-published assessment.

    A finalizer can be interrupted after writing this report but before the
    capture completion marker. Reuse it only if current inputs and every mask
    reproduce the saved result; a stale or partial assessment fails closed.
    Standalone audits still use write_assessment and always require a new path.
    """
    output = Path(output)
    if not output.exists():
        return write_assessment(run, output, integrity_status)
    expected, masks = assess(run, integrity_status)
    saved = json.loads((output / 'report.json').read_text())
    normalized = lambda report: {k: v for k, v in report.items() if k != 'checked_utc'}
    if normalized(saved) != normalized(expected):
        raise ValueError('existing training assessment differs from current capture; refusing to overwrite')
    with np.load(output / 'row_masks.npz') as actual:
        if set(actual.files) != set(masks) or any(not np.array_equal(actual[k], v) for k, v in masks.items()):
            raise ValueError('existing training masks differ from current capture; refusing to overwrite')
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, help='new output directory; default creates a separate audit under outputs')
    args = parser.parse_args()
    if args.output:
        output = args.output
    else:
        output = Path(tempfile.mkdtemp(prefix=datetime.now().strftime('%Y%m%d_%H%M%S')+'_training_audit_', dir=ROOT/'outputs')) / 'assessment'
    report = write_assessment(args.run, output)
    print(json.dumps({'output': str(output.resolve()), 'status': report['status'], 'metrics': report['metrics'],
                      'failed_checks': [c for c in report['checks'] if not c['passed']]}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report.get('machine_quality_passed') else 2)


if __name__ == '__main__':
    main()
