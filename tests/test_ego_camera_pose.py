"""Regression for runtime torso heading being erased by Gf.SetTranslate."""
import math
import unittest

try:
    from pxr import Gf, Usd, UsdGeom
except ImportError:
    Gf = Usd = UsdGeom = None

from ego_camera import (EGO_CAMERA_MOUNT_M, EGO_CAMERA_PITCH_DOWN_DEG,
                        setup_ego_camera, update_ego_camera_transform)


@unittest.skipIf(Gf is None, "run with Isaac USD libraries available")
class EgoCameraPose(unittest.TestCase):
    def setUp(self):
        self.stage = Usd.Stage.CreateInMemory()
        self.parent = UsdGeom.Xform.Define(self.stage, "/Robot/torso_link")
        self.info = setup_ego_camera(self.stage, "/Robot/torso_link")

    def matrix(self):
        return UsdGeom.Xformable(self.stage.GetPrimAtPath(self.info["camera_path"])
            ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    def test_robot_turns_change_camera_heading_and_eye_offset(self):
        for degrees in (0, 90, 180, -90):
            with self.subTest(yaw=degrees):
                theta = math.radians(degrees)
                update_ego_camera_transform(self.stage, self.info,
                    world_pos=(2, 5, .8),
                    world_quat_wxyz=(math.cos(theta / 2), 0, 0, math.sin(theta / 2)))
                matrix = self.matrix()
                mx,my,mz = EGO_CAMERA_MOUNT_M
                expected_eye = Gf.Vec3d(2 + mx * math.cos(theta) - my * math.sin(theta),
                    5 + mx * math.sin(theta) + my * math.cos(theta), .8 + mz)
                expected_forward = Gf.Vec3d(math.cos(theta), math.sin(theta),
                    -math.tan(math.radians(EGO_CAMERA_PITCH_DOWN_DEG))).GetNormalized()
                self.assertLess((matrix.ExtractTranslation() - expected_eye).GetLength(), 1e-6)
                self.assertLess((matrix.TransformDir(Gf.Vec3d(0, 0, -1)).GetNormalized()
                    - expected_forward).GetLength(), 1e-6)

    def test_custom_mount_pitch_and_intrinsics_are_used_at_setup_and_update(self):
        self.info = setup_ego_camera(self.stage, "/Robot/torso_link", width=640, height=360,
            mount_position_m=(.12, -.03, .41), pitch_down_deg=42, focal_length_mm=16.)
        self.assertLess((self.matrix().ExtractTranslation()-Gf.Vec3d(.12,-.03,.41)).GetLength(), 1e-6)
        update_ego_camera_transform(self.stage, self.info, world_pos=(1,2,3),
            world_quat_wxyz=(1,0,0,0))
        self.assertLess((self.matrix().ExtractTranslation()-Gf.Vec3d(1.12,1.97,3.41)).GetLength(), 1e-6)
        direction = self.matrix().TransformDir(Gf.Vec3d(0,0,-1)).GetNormalized()
        expected = Gf.Vec3d(math.cos(math.radians(42)),0,-math.sin(math.radians(42)))
        self.assertLess((direction-expected).GetLength(), 1e-6)
        camera=UsdGeom.Camera(self.stage.GetPrimAtPath(self.info["camera_path"]))
        self.assertEqual(camera.GetFocalLengthAttr().Get(),16.)
        self.assertEqual(camera.GetVerticalApertureAttr().Get(),20.25)
        self.assertEqual(self.info["pitch_down_deg"],42.)

    def test_grasp_workspace_moves_towards_image_center(self):
        # A tabletop target .70 m ahead and .55 m below the optical center.
        point=Gf.Vec3d(.06+.70,0,.45-.55)
        offsets=[]
        for mount,pitch,focal in (((.06,0,.45),math.degrees(math.atan(.28)),18.),
                                 (EGO_CAMERA_MOUNT_M,EGO_CAMERA_PITCH_DOWN_DEG,14.)):
            self.info=setup_ego_camera(self.stage,"/Robot/torso_link",pitch_down_deg=pitch,
                                       mount_position_m=mount,focal_length_mm=focal)
            local=self.matrix().GetInverse().Transform(point)
            self.assertLess(local[2],0)
            offsets.append(abs(local[1]/local[2]))
        self.assertLess(offsets[1],offsets[0])

    def test_invalid_configuration_is_rejected_before_camera_changes(self):
        for kwargs in ({"pitch_down_deg":float('nan')}, {"pitch_down_deg":85},
                       {"mount_position_m":(0,0,float('inf'))}, {"focal_length_mm":0},
                       {"width":0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                setup_ego_camera(self.stage,"/Robot/torso_link",**kwargs)

    def test_balanced_view_keeps_close_grasp_points_inside_frame(self):
        # Close palms lie below the old head camera's vertical field of view.
        # Use USD's own projection to verify the new mount and lens together.
        points=[Gf.Vec3d(.21,y,.05) for y in (-.10,.10)]
        old=setup_ego_camera(self.stage,"/Robot/torso_link",width=640,height=360,
            mount_position_m=(.06,0,.45),pitch_down_deg=30,focal_length_mm=18)
        def project(info,point):
            frustum=UsdGeom.Camera(self.stage.GetPrimAtPath(info['camera_path'])).GetCamera().frustum
            return (frustum.ComputeViewMatrix()*frustum.ComputeProjectionMatrix()).Transform(point)
        self.assertTrue(all(abs(project(old,p)[1])>1 for p in points))
        current=setup_ego_camera(self.stage,"/Robot/torso_link",width=640,height=360)
        for point in points:
            projected=project(current,point)
            self.assertLess(abs(projected[0]),1)
            self.assertLess(abs(projected[1]),1)
        self.assertFalse(current['physical_camera_calibration'])

    def test_balanced_view_reduces_floor_bias_without_losing_table_target(self):
        target=Gf.Vec3d(.76,0,-.10)
        def project(mount,pitch):
            info=setup_ego_camera(self.stage,'/Robot/torso_link',width=640,height=360,
                mount_position_m=mount,pitch_down_deg=pitch,focal_length_mm=14.)
            frustum=UsdGeom.Camera(self.stage.GetPrimAtPath(info['camera_path'])).GetCamera().frustum
            return (frustum.ComputeViewMatrix()*frustum.ComputeProjectionMatrix()).Transform(target)
        previous=project((.12,0,.30),50.)
        current=project(EGO_CAMERA_MOUNT_M,EGO_CAMERA_PITCH_DOWN_DEG)
        self.assertLess(abs(current[1]),abs(previous[1]))
        self.assertLess(abs(current[1]),1.)

    def test_usd_parent_and_runtime_pose_produce_same_camera(self):
        rotation = Gf.Rotation(Gf.Vec3d(0, 0, 1), 90) * Gf.Rotation(Gf.Vec3d(0, 1, 0), 15)
        body = Gf.Matrix4d().SetRotate(rotation)
        body.SetTranslateOnly(Gf.Vec3d(2, 5, .8))
        self.parent.AddTransformOp().Set(body)
        update_ego_camera_transform(self.stage, self.info)
        expected = self.matrix()
        q = rotation.GetQuat()
        update_ego_camera_transform(self.stage, self.info, world_pos=(2, 5, .8),
            world_quat_wxyz=(q.GetReal(), *q.GetImaginary()))
        actual = self.matrix()
        self.assertLess(max(abs(actual[i][j] - expected[i][j])
            for i in range(4) for j in range(4)), 1e-6)

    def test_rectified_K_matches_USD_projection_including_principal_point(self):
        from camera_intrinsics import load_rectified_intrinsics
        c = load_rectified_intrinsics(); k = c['K_px']
        info = setup_ego_camera(self.stage, '/Robot/torso_link', width=640, height=360, intrinsics=c)
        camera = UsdGeom.Camera(self.stage.GetPrimAtPath(info['camera_path']))
        pose = UsdGeom.Xformable(camera).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        frustum = camera.GetCamera().frustum
        projection = frustum.ComputeViewMatrix()*frustum.ComputeProjectionMatrix()
        plan = info['render_plan']
        for x, y, z in ((0., 0., 2.), (.2, -.1, 1.), (-.3, .2, 1.5)):
            world = pose.Transform(Gf.Vec3d(x, -y, -z))
            ndc = projection.Transform(world)
            dx,dy = plan['output_to_render_offset_px']
            uv = ((ndc[0]+1)*plan['render_width']/2-.5-dx,
                  (1-ndc[1])*plan['render_height']/2-.5-dy)
            expected = (k[0][0]*x/z+k[0][2], k[1][1]*y/z+k[1][2])
            self.assertLess(max(abs(a-b) for a,b in zip(uv,expected)), 2e-5)
        self.assertTrue(info['intrinsics_match_real_rectified_output'])
        self.assertFalse(info['physical_camera_calibration'])
        self.assertFalse(info['extrinsics_calibrated'])

    def test_rectified_projection_preserves_mount_and_can_restore_legacy(self):
        from camera_intrinsics import load_rectified_intrinsics
        before = self.matrix()
        info = setup_ego_camera(self.stage, '/Robot/torso_link', width=640, height=360,
                                intrinsics=load_rectified_intrinsics())
        self.assertLess(max(abs(self.matrix()[i][j]-before[i][j]) for i in range(4) for j in range(4)), 1e-6)
        self.assertIsNotNone(info['render_plan'])
        setup_ego_camera(self.stage, '/Robot/torso_link', width=640, height=360)
        self.assertFalse(self.stage.GetPrimAtPath(info['camera_path']).HasAttribute('omni:lensdistortion:model'))

    def test_estimated_mount_stays_rigid_through_torso_roll_pitch_and_yaw(self):
        import numpy as np
        from camera_extrinsics import load_extrinsics
        mount=load_extrinsics();expected=np.asarray(mount['T_parent_camera_cv'])
        cv_to_usd=np.diag([1.,-1.,-1.,1.])
        info=setup_ego_camera(self.stage,'/Robot/torso_link',width=640,height=360,extrinsics=mount)
        for angles in ((0,0,0),(15,-20,90),(-30,15,180),(45,25,-90)):
            rotation=Gf.Rotation(Gf.Vec3d(1,0,0),angles[0])*Gf.Rotation(Gf.Vec3d(0,1,0),angles[1])*Gf.Rotation(Gf.Vec3d(0,0,1),angles[2])
            torso=Gf.Matrix4d().SetRotate(rotation);torso.SetTranslateOnly(Gf.Vec3d(2,5,.8))
            q=rotation.GetQuat()
            update_ego_camera_transform(self.stage,info,world_pos=(2,5,.8),world_quat_wxyz=(q.GetReal(),*q.GetImaginary()))
            relative=np.asarray(torso.GetInverse()).T@np.asarray(self.matrix()).T@cv_to_usd
            with self.subTest(angles=angles):np.testing.assert_allclose(relative,expected,rtol=0,atol=1e-12)
        self.assertEqual(info['mount_rotation_semantics'],'rigid_parent')
        self.assertFalse(info['extrinsics_calibrated'])

    def test_estimated_mount_uses_same_transform_from_USD_parent_or_live_state(self):
        import numpy as np
        from camera_extrinsics import load_extrinsics
        rotation=Gf.Rotation(Gf.Vec3d(1,0,0),25)*Gf.Rotation(Gf.Vec3d(0,1,0),-15)
        body=Gf.Matrix4d().SetRotate(rotation);body.SetTranslateOnly(Gf.Vec3d(1,2,3))
        self.parent.AddTransformOp().Set(body)
        info=setup_ego_camera(self.stage,'/Robot/torso_link',width=640,height=360,extrinsics=load_extrinsics())
        expected=np.asarray(self.matrix()).copy();q=rotation.GetQuat()
        update_ego_camera_transform(self.stage,info,world_pos=(1,2,3),world_quat_wxyz=(q.GetReal(),*q.GetImaginary()))
        np.testing.assert_allclose(np.asarray(self.matrix()),expected,rtol=0,atol=1e-12)

    def test_legacy_roll_stabilization_is_retained_only_for_historical_geometry(self):
        import numpy as np
        rotation=Gf.Rotation(Gf.Vec3d(1,0,0),15)
        q=rotation.GetQuat();body=Gf.Matrix4d().SetRotate(rotation)
        update_ego_camera_transform(self.stage,self.info,world_pos=(0,0,0),world_quat_wxyz=(q.GetReal(),*q.GetImaginary()))
        relative=np.asarray(body.GetInverse()).T@np.asarray(self.matrix()).T
        local=setup_ego_camera(self.stage,'/Robot/torso_link',width=640,height=360)
        self.assertGreater(np.max(np.abs(relative-np.asarray(self.matrix()).T)),.1)
        self.assertEqual(local['mount_rotation_semantics'],'legacy_world_up_stabilized')
