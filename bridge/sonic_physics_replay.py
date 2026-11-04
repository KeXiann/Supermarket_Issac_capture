#!/usr/bin/env python3
"""Prepare an immutable token bundle and run g1_dyn on NEW physical states.

The adapter implements the Bridge interface without opening any DDS channels.
Only initial state and the exogenous token/hand-command sequence are recorded
inputs. Body states, action history, targets and contacts then evolve in PhysX.
"""
import argparse
from collections import deque
import json
from pathlib import Path
import re

import numpy as np

from bridge_core import MotorCommand
from capture_manifest import fingerprint
from validate_sonic_tokens import (action_mapping, captured_file, gravity_cpp,
                                   reconstruct_observations, restore_policy_states)


class PolicyHistory:
    def __init__(self, config):
        self.config=config
        self.rows=deque(maxlen=max(int(re.search(r'_(\d+)frame',o['name'])[1]) for o in config['observations'] if 'frame' in o['name']))

    def append(self, q, dq, omega, quat, last_action, token):
        self.rows.append(dict(body_q=np.asarray(q).copy(), body_dq=np.asarray(dq).copy(),
            base_ang_vel=np.asarray(omega).copy(),base_quat=np.asarray(quat).copy(),
            last_action=np.asarray(last_action).copy(),token_state=np.asarray(token).copy()))
        return reconstruct_observations(list(self.rows),self.config)[0][-1:]


def physical_source_rows(control_times, physical_times):
    """Exclude frozen startup/shutdown control cycles outside measured physics."""
    control_times, physical_times = np.asarray(control_times), np.asarray(physical_times)
    selected = np.flatnonzero((control_times >= physical_times[0]) & (control_times <= physical_times[-1]))
    if len(selected) < 2:
        raise ValueError('Fewer than two control cycles covered by recorded physics')
    rows = np.searchsorted(physical_times, control_times[selected], side='right') - 1
    return selected, rows


def prepare(run, validation, output):
    import yaml
    from sonic_contract import policy_parameters
    from sonic_debug_recorder import validate_recording
    from upstream_contract import captured_physics
    source_physics = captured_physics(run)
    manifest=json.loads((run/'capture_sonic.json').read_text())
    model,model_entry=captured_file(run,manifest,'model_decoder.onnx')
    config_path,_=captured_file(run,manifest,'observation_config.yaml',True)
    header,_=captured_file(run,manifest,'policy_parameters.hpp',True)
    config=yaml.safe_load(config_path.read_text());order=action_mapping(header)
    _,records=validate_recording(run/'sonic_debug.msgpack')
    robot_configs=[r['payload'] for r in records if r['topic']=='robot_config']
    states=[r['payload'] for r in records if r['topic']=='g1_debug']
    cfg=robot_configs[0]
    default=np.asarray(json.loads(cfg['body_recording_default_q_hardware_json']))
    scale=np.asarray(json.loads(cfg['body_recording_action_scale_hardware_json']))
    times=np.asarray([s['state_monotonic_ns'] for s in states],dtype=np.int64)
    params=policy_parameters(header)
    reference=np.load(validation/'control_decoder.npz')
    if not np.array_equal(times,reference['state_monotonic_ns']):raise ValueError('source time mismatch')
    all_states = states
    with np.load(run/'bridge.npz') as b:
        selected, rows = physical_source_rows(times, b['state_monotonic_ns'])
        times = times[selected]
        states = [all_states[i] for i in selected]
        arrays=dict(token=np.asarray([s['token_state'] for s in states],dtype=np.float32),
            source_monotonic_ns=times,source_cycle=np.asarray([s['index'] for s in states]),
            encoder_mode=np.asarray([s['encoder_mode'] for s in states]),
            default=default,scale=scale,order=order,kp=np.asarray(params['kps']),kd=np.asarray(params['kds']))
        for key in ['root_pos','root_quat_wxyz','root_lin_vel_w','root_ang_vel_w','joint_pos','joint_vel',
                    'left_hand_joint_pos','right_hand_joint_pos','left_hand_joint_vel','right_hand_joint_vel']:
            arrays['initial_'+key]=b[key][rows[0]].copy()
        for side in ('left','right'):
            arrays[side+'_hand_command']=b[side+'_hand_dds_target'][rows].copy()
    import onnxruntime as ort
    opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
    session=ort.InferenceSession(str(model),sess_options=opts,providers=['CPUExecutionProvider'])
    history=PolicyHistory(config);restored=restore_policy_states(all_states,default,scale,order)
    check=[]
    for row in restored:
        obs=history.append(row['body_q'],row['body_dq'],row['base_ang_vel'],row['base_quat'],row['last_action'],row['token_state'])
        check.append(session.run(None,{session.get_inputs()[0].name:obs})[0][0])
    difference=float(np.max(np.abs(np.asarray(check)-reference['raw_action'])))
    if difference>1e-5:raise ValueError('streaming policy history differs from verified control reconstruction')
    # Retain the genuine pre-start policy history once. The rollout only uses
    # newly simulated body states after initialization, never later recordings.
    first = int(selected[0])
    seed = restored[max(0, first - history.rows.maxlen + 1):first]
    arrays['initial_last_action'] = restored[first]['last_action']
    for key in ('body_q', 'body_dq', 'base_ang_vel', 'base_quat', 'last_action', 'token_state'):
        arrays['initial_history_' + key] = np.asarray([s[key] for s in seed], dtype=np.float32)
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'input.npz',**arrays)
    info={'run':str(run),'validation':str(validation),'decoder':model_entry,'observation_config':config,
          'control_dt':.02,'physics_dt':.005,'control_cycles':len(times),'duration_s':len(times)*.02,
          'history_equivalence_max_abs_error':difference,
          'excluded_control_cycles_outside_physics': len(all_states) - len(states),
          'source_control_row_range': [int(selected[0]), int(selected[-1])],
          'initial_history_rows': len(seed),
          'clock':'one saved token per 20ms simulation time; independent of render/wall speed',
          'recording_window':json.loads((run/'recording_control.json').read_text()),
          'input_fingerprints':[fingerprint(run/n) for n in ['bridge.npz','sonic_debug.msgpack','bridge.json']],
          'semantics':'fresh-state g1_dyn closed loop, no recorded body-state teacher forcing after initialization'}
    info['physics_contract'] = source_physics
    (output/'input.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps({k:v for k,v in info.items() if k not in ['observation_config','input_fingerprints']},indent=2))


class LocalTokenReplay:
    def __init__(self, bundle, output):
        import onnxruntime as ort
        self.info=json.loads(bundle.with_suffix('.json').read_text())
        if fingerprint(Path(self.info['decoder']['path']))['sha256']!=self.info['decoder']['sha256']:
            raise ValueError('captured decoder changed')
        with np.load(bundle) as a:self.a={k:a[k] for k in a.files}
        opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
        self.session=ort.InferenceSession(self.info['decoder']['path'],sess_options=opts,providers=['CPUExecutionProvider'])
        self.input_name=self.session.get_inputs()[0].name
        self.history=PolicyHistory(self.info['observation_config']);self.last_action=np.zeros(29)
        if 'initial_last_action' in self.a:
            self.last_action=self.a['initial_last_action'].copy()
            inverse=np.argsort(self.a['order'])
            for _ in range(self.info.get('seed_history_frames',0)):
                self.history.append((self.a['initial_joint_pos']-self.a['default'])[inverse],
                    self.a['initial_joint_vel'][inverse],self.a['initial_base_ang_vel_b'],
                    self.a['initial_root_quat_wxyz'],self.last_action,self.a['token'][0])
        if 'initial_history_body_q' in self.a:
            for i in range(len(self.a['initial_history_body_q'])):
                self.history.append(*(self.a['initial_history_' + k][i] for k in
                    ('body_q', 'body_dq', 'base_ang_vel', 'base_quat', 'last_action', 'token_state')))
        self.index=-1;self.cmd=None;self.trace=[];self.output=output;self.world=[]

    def publish(self,t,q,dq,ddq,torque,root_pos,root_quat,root_lin_w,root_ang_w,root_ang_b,*rest):
        if 'control_time_s' in self.a:
            from replay_clock import due_rows
            for index in due_rows(self.a['control_time_s'],self.index,t):
                self._infer(index,t,q,dq,root_pos,root_quat,root_ang_b)
            return
        index=min(int(np.floor((t+1e-8)/.02)),len(self.a['token'])-1)
        if index==self.index:return
        if index!=self.index+1:raise ValueError('policy skipped a token')
        self._infer(index,t,q,dq,root_pos,root_quat,root_ang_b)

    def _infer(self,index,t,q,dq,root_pos,root_quat,root_ang_b):
        inverse=np.argsort(self.a['order'])
        obs=self.history.append((q-self.a['default'])[inverse],dq[inverse],root_ang_b,root_quat,
                                self.last_action,self.a['token'][index])
        action=self.session.run(None,{self.input_name:obs})[0][0]
        if action.shape!=(29,) or not np.isfinite(action).all():raise ValueError('invalid live decoder output')
        target=(self.a['default']+action[self.a['order']]*self.a['scale']).astype(np.float32)
        self.cmd=MotorCommand(target,np.zeros(29),self.a['kp'],self.a['kd'],np.zeros(29))
        self.trace.append(dict(time=t,token_row=index,observation=obs[0].copy(),raw_action=action.copy(),target=target,
                               joint_pos=q.copy(),root_pos=root_pos.copy(),root_quat_wxyz=root_quat.copy()))
        self.last_action=action.copy();self.index=index

    def latest(self):return self.cmd,0.,self.index+1,0,''
    def latest_hand(self,is_left,with_clock=False):
        key=('left' if is_left else 'right')+'_hand_command'
        result=(self.a[key][max(0,self.index)].copy(),np.full(7,4.),np.full(7,.12),True,self.index+1)
        return result+(max(0,self.index)*.02,0) if with_clock else result
    def raw_snapshot(self):return (self.cmd.q.copy(),True) if self.cmd else (np.zeros(29),False)
    def rejection_summary(self):return {'motors':{},'categories':{},'outside_physical_limit_count':0}
    def close(self):
        if self.trace:
            np.savez_compressed(self.output/'policy_trace.npz',**{k:np.stack([x[k] for x in self.trace]) for k in self.trace[0]})
        if self.world:
            np.savez_compressed(self.output/'world_trace.npz',**{k:np.stack([x[k] for x in self.world]) for k in self.world[0]})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--validation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    prepare(a.run.resolve(),a.validation.resolve(),a.output.resolve())
