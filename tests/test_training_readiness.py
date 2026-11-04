"""Bad human labels/history must not be promoted by a successful file save."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from assess_training_readiness import assess, write_assessment, ensure_capture_assessment
from capture_session import initialize


class TrainingReadinessTests(unittest.TestCase):
    def fixture(self, run, kind='live_pico', stale=None, nan=False):
        initialize(run, task='pick up the basket', input_kind=kind)
        (run/'meta').mkdir(); (run/'training/semanticdpc_0826').mkdir(parents=True)
        n=100
        columns={'observation.body_valid':np.ones(n,int), 'teleop.stream_mode':np.ones(n,int),
                 'observation.hand_valid':np.ones((n,2),int), 'action.hand_valid':np.ones(n,int),
                 'observation.lowstate_tick':np.arange(n)*20,
                 'action.wbc':np.zeros((n,29))}
        if nan: columns['action.wbc'][30,0]=np.nan
        pq.write_table(pa.table({k:v.tolist() for k,v in columns.items()}),run/'data.parquet')
        def save(rel,data): (run/rel).write_text(json.dumps(data))
        available = {key: {'available': True} for key in ('observation.state', 'observation.joint_velocity',
                     'action.wbc', 'action.motion_token', 'teleop.smpl_joints', 'action.hand_joint_position')}
        save('meta/info.json',{'data_path':'data.parquet','script_config':{'body_action_alignment_version':2,'field_availability':available}})
        (run/'meta/tasks.jsonl').write_text(json.dumps({'task':'pick up the basket'})+'\n')
        save('training/manifest.json',{})
        save('training/semanticdpc_0826/manifest.json',{'status':'complete','artifact':'episode_000000.npz'})
        save('bridge.json',{'world_trace_required':False})
        human=np.ones(n,bool)
        if stale is not None: human[stale]=False
        np.savez(run/'meta/bridge_alignment.npz',human_valid=human,human_age_s=np.zeros(n),
                 camera_age_s=np.zeros(n),sim_time_s=np.arange(n)*.02,
                 host_monotonic_s=100+np.arange(n)*.02,camera_source_frame=np.arange(n)//2)
        history=np.ones((n,10),bool)
        for i in range(9):history[i,:9-i]=False
        np.savez(run/'training/semanticdpc_0826/episode_000000.npz',sample_valid=np.ones(n,bool),history_mask=history,
                 state64=np.zeros((n,64)),history10=np.zeros((n,10,95)),action31=np.zeros((n,31)),
                 boundary_reason=np.array(['episode_start']+['continuous']*(n-1)),segment_id=np.zeros(n,int))

    def test_clean_candidate_still_requires_task_success_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp); self.fixture(run)
            report,masks=assess(run,integrity_status='complete')
            self.assertEqual(report['status'],'candidate_requires_task_review')
            self.assertTrue(report['machine_quality_passed'])
            self.assertFalse(report['formal_collection_approved'])
            self.assertEqual(masks['visual_pose_full_history_valid'].sum(),91)

    def test_diagnostic_input_cannot_become_a_training_demonstration(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);self.fixture(run,kind='recorded_pose_smoke')
            report,_=assess(run,integrity_status='complete')
            self.assertEqual(report['status'],'diagnostic_pipeline_passed_not_training_data')
            self.assertTrue(report['technical_quality_passed'])
            self.assertFalse(report['machine_quality_passed'])

    def test_stale_human_row_invalidates_whole_ten_frame_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);self.fixture(run,stale=30)
            _,masks=assess(run,integrity_status='complete')
            self.assertFalse(masks['revo2_pose_bc_valid'][30])
            self.assertFalse(masks['visual_pose_full_history_valid'][30:40].any())
            self.assertTrue(masks['visual_pose_full_history_valid'][40])

    def test_nonfinite_labels_are_masked_even_when_files_are_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);self.fixture(run,nan=True)
            report,masks=assess(run,integrity_status='complete')
            self.assertFalse(report['machine_quality_passed'])
            self.assertFalse(masks['revo2_pose_bc_valid'][30])

    def test_missing_provenance_cannot_be_assumed_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);self.fixture(run)
            path=run/'meta/info.json'
            info=json.loads(path.read_text())
            info['script_config']['field_availability'].pop('action.motion_token')
            path.write_text(json.dumps(info))
            report,_=assess(run,integrity_status='complete')
            self.assertFalse(report['machine_quality_passed'])
            check=next(c for c in report['checks'] if c['name']=='required_pose_signals_available')
            self.assertEqual(check['detail'],['action.motion_token'])

    def test_stale_age_is_masked_even_if_an_imported_valid_flag_is_wrong(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);self.fixture(run)
            path=run/'meta/bridge_alignment.npz'
            with np.load(path) as a: arrays={k:a[k].copy() for k in a.files}
            arrays['human_age_s'][30]=.2
            np.savez(path,**arrays)
            _,masks=assess(run,integrity_status='complete')
            self.assertFalse(masks['revo2_pose_bc_valid'][30])
            self.assertFalse(masks['visual_pose_full_history_valid'][30:40].any())

    def test_finalization_retry_reuses_verified_report_without_rewriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)/'run';self.fixture(run)
            output=Path(tmp)/'audit'
            first=ensure_capture_assessment(run,output)
            before={p.name:p.read_bytes() for p in output.iterdir()}
            self.assertEqual(ensure_capture_assessment(run,output),first)
            self.assertEqual({p.name:p.read_bytes() for p in output.iterdir()},before)

    def test_finalization_retry_rejects_changed_inputs_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)/'run';self.fixture(run)
            output=Path(tmp)/'audit'
            ensure_capture_assessment(run,output)
            before={p.name:p.read_bytes() for p in output.iterdir()}
            path=run/'session.json';session=json.loads(path.read_text())
            session['input_kind']='recorded_pose_smoke';path.write_text(json.dumps(session))
            with self.assertRaisesRegex(ValueError,'refusing to overwrite'):
                ensure_capture_assessment(run,output)
            self.assertEqual({p.name:p.read_bytes() for p in output.iterdir()},before)

    def test_audit_never_overwrites_an_existing_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)/'run'; self.fixture(run)
            output=Path(tmp)/'audit'
            write_assessment(run,output,integrity_status='complete')
            original=(output/'report.json').read_bytes()
            with self.assertRaises(FileExistsError):write_assessment(run,output,integrity_status='complete')
            self.assertEqual((output/'report.json').read_bytes(),original)

    def test_blank_task_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):initialize(Path(tmp),task='  ')


if __name__=='__main__':unittest.main()
