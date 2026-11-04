import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from upstream_contract import COMMIT, model_contract, check_physics_metadata, replay_selection
from upstream_full_robot import NativeRobot, physics_inventory, restore_authored_task_world

try:
    from pxr import Usd, UsdGeom, UsdPhysics, Gf
except ImportError:
    Usd = None


class FakeArticulation:
    def __init__(self, prefix, count):
        self.joint_names = [prefix + str(i) for i in range(count)]
        self.body_names = self.joint_names
        self.data = SimpleNamespace(joint_pos=torch.zeros(1, count))
        self.target = torch.zeros(1, count)
        self.device = 'cpu'

    def set_joint_position_target(self, values, joint_ids):
        self.target[:, joint_ids] = values

    def write_joint_stiffness_to_sim(self, values, joint_ids):
        self.data.joint_pos[:, joint_ids] = values

    def update(self, dt):
        pass


class NativeInterfaceTests(unittest.TestCase):
    def test_routes_reordered_body_and_hand_indices_without_cross_actuation(self):
        parts = [FakeArticulation('body', 3), FakeArticulation('left', 2), FakeArticulation('right', 2)]
        robot = NativeRobot(parts, None, None)
        robot.set_joint_position_target(torch.tensor([[7., 3., 9.]]), joint_ids=torch.tensor([6, 1, 3]))
        self.assertEqual(parts[0].target.tolist(), [[0., 3., 0.]])
        self.assertEqual(parts[1].target.tolist(), [[9., 0.]])
        self.assertEqual(parts[2].target.tolist(), [[0., 7.]])
        robot.set_joint_position_target(torch.arange(7)[None, :].float())
        self.assertEqual(torch.cat([p.target for p in parts], dim=1).tolist(), [list(range(7))])

    def test_combined_state_is_refreshed_after_physics_and_parameter_writes(self):
        parts = [FakeArticulation('body', 3), FakeArticulation('left', 2), FakeArticulation('right', 2)]
        robot = NativeRobot(parts, None, None)
        self.assertEqual(robot.data.joint_pos.sum(), 0.)
        parts[1].data.joint_pos[:] = 2.
        robot.update(.005)
        self.assertEqual(robot.data.joint_pos.sum(), 4.)
        robot.write_joint_stiffness_to_sim(3., joint_ids=[0, 4])
        self.assertEqual(robot.data.joint_pos.tolist(), [[3., 0., 0., 2., 3., 0., 0.]])

    def test_full_profile_requires_native_audit_and_replay_keeps_profile(self):
        model = model_contract()
        meta = {'pd_backend': 'external', 'effort_limits_motor_order': model['g1_29dof']['effort_limit_nm'],
                'physics_contract': {'profile': 'upstream_full', 'commit': COMMIT, 'pd_backend': 'external',
                    'joint_order': model['joint_order'], 'armature_kg_m2': [.01] * 29}}
        self.assertFalse(check_physics_metadata(meta))
        meta['physics_contract'].update(full_model_equivalence=True, checked_scene_configuration_equality=True,
            scene_physics_audit={'passed': True}, native_runtime_audit={
            'passed': True, 'commit': COMMIT, 'native_mimic_constraints': 10, 'articulations': [{}, {}, {}],
            'runtime_versions': {'isaacsim': '5.1.0.0', 'isaaclab_release': '2.3.2'}})
        self.assertTrue(check_physics_metadata(meta))
        drift = copy.deepcopy(meta)
        drift['physics_contract']['scene_physics_audit']['passed'] = False
        self.assertFalse(check_physics_metadata(drift))
        wrong = copy.deepcopy(meta)
        wrong['physics_contract']['native_runtime_audit']['runtime_versions']['isaacsim'] = '5.0.0.0'
        self.assertFalse(check_physics_metadata(wrong))
        with tempfile.TemporaryDirectory() as folder:
            bundle = Path(folder) / 'input.npz'
            bundle.with_suffix('.json').write_text(json.dumps(meta))
            self.assertEqual(replay_selection(bundle), ('upstream_full', 'external'))
            with self.assertRaisesRegex(ValueError, 'differs'):
                replay_selection(bundle, 'upstream', 'external')

    def test_replay_excludes_frozen_startup_and_shutdown_cycles(self):
        from sonic_physics_replay import physical_source_rows
        selected, rows = physical_source_rows([99, 100, 125, 150, 175, 200, 205], [100, 120, 140, 160, 180, 200])
        np.testing.assert_array_equal(selected, [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(rows, [0, 1, 2, 3, 5])

    def test_runtime_profile_preserves_historical_scene_and_replay(self):
        from bridge_runtime import selected_profile
        self.assertEqual(selected_profile(['--scene', 'assets/scenes/pico_supermarket_task1_7e5fd9d.usda']), 'upstream_full')
        self.assertEqual(selected_profile([]), 'upstream_full')
        self.assertEqual(selected_profile(['--scene', 'old.usda', '--physics-profile', 'legacy']), 'legacy')

    def test_native_profile_never_falls_back_to_conda_when_uv_is_incomplete(self):
        import bridge_runtime as runtime
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(runtime, 'NATIVE_READY', Path(folder) / 'missing.json'):
                with self.assertRaisesRegex(RuntimeError, 'uv environment is not configured'):
                    runtime.bootstrap(['--physics-profile', 'upstream_full'])

    def test_native_profile_selects_isolated_python_before_simulation_imports(self):
        import bridge_runtime as runtime
        with tempfile.TemporaryDirectory() as folder:
            marker = Path(folder) / 'environment.json'
            marker.write_text(json.dumps({'environment_checks_passed': True, 'physics_steps': 0}))
            with patch.object(runtime, 'NATIVE_READY', marker), \
                 patch.object(runtime.sys, 'executable', '/unselected/python'), \
                 patch.object(runtime.os, 'execv') as execute:
                runtime.bootstrap(['--physics-profile', 'upstream_full'])
            self.assertEqual(execute.call_args.args[0], str(runtime.ROOT / '.venv/bin/python'))


@unittest.skipIf(Usd is None, 'requires installed USD bindings')
class NativeUsdTests(unittest.TestCase):
    def test_inventory_detects_physics_changes_but_not_world_placement(self):
        s = Usd.Stage.CreateInMemory()
        root = UsdGeom.Xform.Define(s, '/Robot')
        body = UsdGeom.Xform.Define(s, '/Robot/pelvis')
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
        mass.CreateMassAttr(1.)
        original = physics_inventory(s, '/Robot')
        root.AddTranslateOp().Set(Gf.Vec3d(4., 5., 6.))
        self.assertEqual(original, physics_inventory(s, '/Robot'))
        mass.GetMassAttr().Set(2.)
        self.assertNotEqual(original, physics_inventory(s, '/Robot'))

    def test_world_restoration_keeps_all_source_poses_and_nonzero_velocities(self):
        root = Path(__file__).resolve().parents[1]
        s = Usd.Stage.Open(str(root / 'assets/scenes/pico_supermarket_task1_7e5fd9d.usda'))
        s.SetEditTarget(s.GetSessionLayer())
        velocities, report = restore_authored_task_world(s, '/World/Supermarket')
        self.assertEqual(report['body_count'], 111)
        self.assertLess(report['pose_authoring_max_error'], 1e-10)
        mapping = json.loads((root / 'configs/task1/init/state_mapping.json').read_text())
        with np.load(root / 'data/task1/init/initial_state.npz') as source:
            for key, old_path in mapping['extra_rigid_bodies'].items():
                path = old_path.replace('/World/envs/env_0/Room', '/World/Supermarket')
                actual = UsdGeom.Xformable(s.GetPrimAtPath(path)).ComputeLocalToWorldTransform(0)
                np.testing.assert_allclose(actual.ExtractTranslation(), source[key + '__transform_xyzw'][0, :3], atol=1e-7)
                np.testing.assert_array_equal(velocities[path], source[key + '__velocity'][0])
        self.assertGreater(max(np.max(abs(v)) for v in velocities.values()), 1.)


if __name__ == '__main__':
    unittest.main()
