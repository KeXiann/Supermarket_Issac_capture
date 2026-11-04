"""Compare runtime optics/scene against independent, pinned upstream inputs."""
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from camera_extrinsics import load_extrinsics
from camera_intrinsics import load_rectified_intrinsics
from camera_profile import resolve_camera_profile
from reference_robot import verify_snapshot
from upstream_camera import (COMMIT, ROOT, SOURCE, SCENE, load_contract, native_frame_due,
                             launcher_options, settings_match, validate_capture_profile, select_camera_frames)

try:
    from pxr import Usd, UsdGeom, UsdPhysics
except ImportError:
    Usd = None


class UpstreamCameraTests(unittest.TestCase):
    def test_all_five_task_camera_documents_are_byte_identical(self):
        expected=(SOURCE/'configs/collection/camera_stereo.json').read_bytes()
        for n in range(1,6):
            self.assertEqual((SOURCE/f'configs/task{n}/init/camera_stereo.json').read_bytes(),expected)
        contract=load_contract()
        self.assertEqual(contract['stereo'],json.loads(expected))
        self.assertEqual(contract['render'],json.loads((SOURCE/'configs/collection/render_config.json').read_bytes()))

    def test_live_left_optics_preserve_full_matrix_and_updated_raw_stereo(self):
        contract=load_contract();stereo=contract['stereo'];c=resolve_camera_profile(SCENE)
        self.assertEqual(c['extrinsics']['T_parent_camera_cv'],stereo['left']['mount']['body_T_camera_cv'])
        self.assertEqual(c['intrinsics']['K_px'],stereo['left']['K'])
        self.assertEqual(c['intrinsics']['raw_calibration'],stereo['raw_calibration'])
        self.assertEqual(c['intrinsics']['right'],stereo['right'])
        self.assertEqual(c['intrinsics']['baseline_m'],.06555722207234115)
        self.assertFalse(c['extrinsics']['calibrated'])
        self.assertFalse(stereo['camera_alignment_accepted'])
        left=np.asarray(stereo['left']['mount']['body_T_camera_cv'])
        right=np.asarray(stereo['right']['mount']['body_T_camera_cv'])
        np.testing.assert_allclose(np.linalg.inv(left)@right,stereo['left_T_right_cv'],atol=1e-12)
        # The updated screw fit includes small but real roll/yaw: never collapse to pitch.
        self.assertNotEqual(left[0,0],0.)
        self.assertEqual(c['override_fields'],[])

    def test_native_pre_action_camera_states_repeat_in_50hz_video(self):
        physical=np.arange(80)
        native=physical[[native_frame_due(i) for i in physical]]
        np.testing.assert_array_equal(native,np.arange(0,80,8))
        control=np.arange(0,80,4)
        camera_rows=np.searchsorted(native,control,side='right')-1
        np.testing.assert_array_equal(native[camera_rows]//4,(np.arange(20)//2)*2)

    def test_strict_capture_rejects_operator_geometry_changes(self):
        validate_capture_profile(resolve_camera_profile(SCENE),640,360)
        for change in ({'pitch':40.}, {'extrinsics_config':'virtual'}, {'intrinsics_config':'virtual'}):
            with self.assertRaises(ValueError):
                validate_capture_profile(resolve_camera_profile(SCENE,**change),640,360)

    def test_render_launcher_and_float32_readback(self):
        c=load_contract();options=launcher_options(c)
        self.assertEqual(options['renderer'],'RealTimePathTracing')
        self.assertIn('--/persistent/rtx/modes/rt2/enabled=true',options['kit_args'])
        self.assertIn('--/app/settings/persistent=false',options['kit_args'])
        expected=c['initial_scene']['render_settings'];actual=dict(expected)
        actual['/rtx/post/tonemap/exposureKey']=float(np.float32(.6))
        self.assertTrue(settings_match(expected,actual))
        actual['/rtx/rendermode']='RaytracedLighting'
        self.assertFalse(settings_match(expected,actual))

    def test_world_reset_preserves_kinematic_table_and_full_tensor_buffers(self):
        from scene_support import restore_world_transforms
        class View:
            def set_transforms(self, data, indices):
                self.transforms, self.pose_ids = data, indices
            def set_velocities(self, data, indices):
                self.velocities, self.velocity_ids = data, indices
        view = View(); poses = np.zeros((3,7)); poses[:,6] = 1.
        restore_world_transforms(view, poses, np.array([0,2],dtype=np.uint32))
        np.testing.assert_array_equal(view.pose_ids,[0,1,2])
        np.testing.assert_array_equal(view.velocity_ids,[0,2])
        self.assertEqual(view.velocities.shape,(3,6))
        self.assertTrue(np.all(view.velocities==0.))

    def test_frame_join_excludes_ready_preview_and_uses_source_state_clock(self):
        camera=[{'sim_time_s':'0', 'host_monotonic_s':'5', 'source_physics_step':'-1'}]
        camera.extend({'sim_time_s':str(i*.04), 'host_monotonic_s':str(10+i*.04),
                       'source_physics_step':str(i*8), 'render_complete_monotonic_s':str(10+i*.04+.012)}
                      for i in range(5))
        sim=np.arange(10)*.02;host=10+sim
        selected=select_camera_frames(camera,host,sim,'upstream')
        np.testing.assert_array_equal(selected,np.arange(10)//2+1)
        legacy=select_camera_frames(camera,host,sim,'wall')
        self.assertEqual(len(legacy),10)
        with self.assertRaisesRegex(ValueError,'source physics'):
            select_camera_frames(camera[:1],host,sim,'upstream')

    def test_active_scene_and_robot_dependencies_use_current_snapshot(self):
        from setup_supermarket_scene import verify_manifest
        self.assertEqual(verify_manifest(SCENE.with_suffix('.json'))['source_commit'], COMMIT)
        self.assertEqual(verify_snapshot()['commit'], COMMIT)
        # An already-running process may still need an immutable older snapshot.
        # Every dependency selected by the current entry must use the current pin.
        manifest = verify_manifest(SCENE.with_suffix('.json'))
        for entry in manifest['dependency_files']:
            parts = Path(entry['path']).parts
            for anchor in ('upstream', 'robot_reference'):
                if anchor in parts:
                    self.assertEqual(parts[parts.index(anchor) + 1], COMMIT[:12])

    @unittest.skipIf(Usd is None,'requires Isaac USD bindings')
    def test_new_scene_task1_transforms_and_exact_camera_geometry(self):
        from setup_supermarket_scene import verify_manifest
        c=load_contract();m=verify_manifest(SCENE.with_suffix('.json'))
        self.assertEqual(m['source_commit'],COMMIT)
        self.assertEqual(m['environment_counts'],{'visual_meshes':1346,'dynamic_rigid_bodies':110,'fixed_joints':47,'modules':93})
        stage=Usd.Stage.Open(str(SCENE))
        for local,source in [('/World/Robot','root_T'),('/World/Supermarket/Objects/basket','/World/Object'),('/World/Supermarket/Objects/T2','/World/Table')]:
            expected=c['initial_scene']['root_T'] if source=='root_T' else c['initial_scene']['static_transforms'][source]
            actual=np.asarray(UsdGeom.Xformable(stage.GetPrimAtPath(local)).ComputeLocalToWorldTransform(0)).T
            np.testing.assert_allclose(actual,expected,atol=1e-12)
        basket=stage.GetPrimAtPath('/World/Supermarket/Objects/basket')
        self.assertTrue(basket.HasAPI(UsdPhysics.RigidBodyAPI))
        self.assertFalse(stage.GetPrimAtPath(str(basket.GetPath())+'/Content').IsActive())
        table=stage.GetPrimAtPath('/World/Supermarket/Objects/T2')
        self.assertTrue(UsdPhysics.RigidBodyAPI(table).GetKinematicEnabledAttr().Get())
        bounds=UsdGeom.BBoxCache(0,['default','render'])
        b=bounds.ComputeWorldBound(basket).ComputeAlignedRange()
        t=bounds.ComputeWorldBound(table).ComputeAlignedRange()
        for axis in (0,1):
            self.assertGreaterEqual(b.GetMin()[axis],t.GetMin()[axis]-1e-5)
            self.assertLessEqual(b.GetMax()[axis],t.GetMax()[axis]+1e-5)
        from reference_robot import ROBOT
        src=Usd.Stage.Open(str(ROBOT));dst=stage
        cache=UsdGeom.XformCache(0)
        for part in ('Bracket0','Bracket1','Bracket2','CameraHousing'):
            s=src.GetPrimAtPath('/G1Revo2/Robot/head_link/ZEDMiniMount/'+part)
            d=dst.GetPrimAtPath('/World/Robot/Robot/head_link/ZEDMiniMount/'+part)
            torso=src.GetPrimAtPath('/G1Revo2/Robot/torso_link')
            expected=cache.GetLocalToWorldTransform(s)*cache.GetLocalToWorldTransform(torso).GetInverse()
            np.testing.assert_allclose(np.asarray(cache.GetLocalToWorldTransform(d)*cache.GetLocalToWorldTransform(dst.GetPrimAtPath("/World/Robot/Robot/torso_link")).GetInverse()),np.asarray(expected),atol=1e-12)
