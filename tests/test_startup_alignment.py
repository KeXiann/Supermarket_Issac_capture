import unittest
import numpy as np
from startup_alignment import assert_nominal_body

try:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
except ImportError:
    Usd = None


class NominalStateTests(unittest.TestCase):
    def test_named_body_state_cannot_be_replaced_by_imported_zero_pose(self):
        names=['finger','knee','hip'];nominal=[-.312,.669]
        self.assertEqual(assert_nominal_body(names,[1.5,.669,-.312],['hip','knee'],nominal)['motor_count'],2)
        with self.assertRaisesRegex(ValueError,'differs from SONIC nominal'):
            assert_nominal_body(names,[1.5,0.,0.],['hip','knee'],nominal)


@unittest.skipIf(Usd is None,'run with installed Isaac USD bindings')
class InitialAssemblyTests(unittest.TestCase):
    def test_foot_bound_does_not_apply_sphere_translation_twice(self):
        from startup_alignment import foot_floor_alignment, pose_matrix
        stage = Usd.Stage.CreateInMemory()
        floor = UsdGeom.Cube.Define(stage, '/World/Room/Objects/floor')
        floor.CreateSizeAttr(2.)
        floor.AddScaleOp().Set(Gf.Vec3f(1, 1, .01))
        poses = {}
        for side in ('left', 'right'):
            name = side + '_ankle_roll_link'
            link = UsdGeom.Xform.Define(stage, '/World/Robot/' + name)
            link.AddTranslateOp().Set(Gf.Vec3d(0, 0, .1))
            sphere = UsdGeom.Sphere.Define(stage, str(link.GetPath()) + '/contact')
            sphere.CreateRadiusAttr(.005)
            sphere.AddTranslateOp().Set(Gf.Vec3d(0, 0, -.03))
            UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
            poses[name] = pose_matrix([0, 0, .1], [1, 0, 0, 0])
        result = foot_floor_alignment(stage, {'environment_root': '/World/Room'}, '/World/Robot', poses)
        self.assertAlmostEqual(result['feet'][0]['sole_bottom_before_m'], .065, places=7)
        self.assertAlmostEqual(result['feet'][0]['gap_before_m'], .055, places=7)

    def test_fixed_mount_uses_both_local_frames_and_actual_body_poses(self):
        from startup_alignment import pose_matrix, audit_joint_frames
        s=Usd.Stage.CreateInMemory();UsdGeom.Xform.Define(s,'/Robot')
        for name in ('wrist','palm'):UsdGeom.Xform.Define(s,'/Robot/'+name)
        j=UsdPhysics.FixedJoint.Define(s,'/Robot/Mount')
        j.CreateBody0Rel().SetTargets(['/Robot/wrist']);j.CreateBody1Rel().SetTargets(['/Robot/palm'])
        j.CreateLocalPos0Attr(Gf.Vec3f(.025,0,0))
        poses={'wrist':pose_matrix([1,2,3],[1,0,0,0]),'palm':pose_matrix([1.025,2,3],[1,0,0,0])}
        result=audit_joint_frames(s,['/Robot'],poses)
        self.assertLess(result['max_pivot_gap_m'],1e-7)
        poses['palm']=pose_matrix([1.035,2,3],[1,0,0,0])
        with self.assertRaisesRegex(ValueError,'joint-frame mismatch'):
            audit_joint_frames(s,['/Robot'],poses)

    def test_revolute_rotation_is_allowed_only_about_its_axis(self):
        from startup_alignment import pose_matrix,audit_joint_frames
        s=Usd.Stage.CreateInMemory();UsdGeom.Xform.Define(s,'/Robot')
        j=UsdPhysics.RevoluteJoint.Define(s,'/Robot/Hinge')
        j.CreateBody0Rel().SetTargets(['/Robot/a']);j.CreateBody1Rel().SetTargets(['/Robot/b']);j.CreateAxisAttr('Z')
        poses={'a':pose_matrix([0,0,0],[1,0,0,0]),'b':pose_matrix([0,0,0],[np.cos(.4),0,0,np.sin(.4)])}
        self.assertLess(audit_joint_frames(s,['/Robot'],poses)['max_constraint_rotation_error_deg'],.001)
        poses['b']=pose_matrix([0,0,0],[np.cos(.4),np.sin(.4),0,0])
        with self.assertRaisesRegex(ValueError,'joint-frame mismatch'):
            audit_joint_frames(s,['/Robot'],poses)

    def test_sole_alignment_uses_colliders_and_preserves_scene_layout(self):
        from startup_alignment import pose_matrix,foot_floor_alignment
        s=Usd.Stage.CreateInMemory()
        floor=UsdGeom.Cube.Define(s,'/World/Market/Objects/floor')
        floor.CreateSizeAttr(1.);floor.AddTranslateOp().Set(Gf.Vec3d(0,0,-.05));floor.AddScaleOp().Set(Gf.Vec3f(4,4,.1))
        for side in ('left','right'):
            link=UsdGeom.Xform.Define(s,'/World/Robot/'+side+'_ankle_roll_link')
            link.AddTranslateOp().Set(Gf.Vec3d(3,0,10)) # intentionally stale authored transform
            shape=UsdGeom.Cube.Define(s,str(link.GetPath())+'/Collision');shape.CreateSizeAttr(.2)
            UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
            # Match imported URDF collision instances, whose children are
            # absent from ordinary PrimRange traversal.
            source=UsdGeom.Xform.Define(s,'/Shapes/'+side)
            collider=UsdGeom.Cube.Define(s,str(source.GetPath())+'/Cube')
            collider.CreateSizeAttr(.2);UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
            collider.CreatePurposeAttr('guide')
            s.RemovePrim(shape.GetPath())
            instance=UsdGeom.Xform.Define(s,str(link.GetPath())+'/Collision').GetPrim()
            instance.GetReferences().AddInternalReference(source.GetPath())
            instance.SetInstanceable(True)
        poses={side+'_ankle_roll_link':pose_matrix([0,.12 if side=='left' else -.12,.15],[1,0,0,0])
               for side in ('left','right')}
        before=s.GetRootLayer().ExportToString()
        result=foot_floor_alignment(s,{'environment_root':'/World/Market'},'/World/Robot',poses)
        self.assertAlmostEqual(result['vertical_shift_m'],-.049,places=6)
        self.assertTrue(all(abs(f['gap_after_m']-.001)<1e-6 for f in result['feet']))
        self.assertEqual(s.GetRootLayer().ExportToString(),before)

    def test_missing_floor_fails_instead_of_guessing_a_height(self):
        from startup_alignment import foot_floor_alignment
        s=Usd.Stage.CreateInMemory();UsdGeom.Xform.Define(s,'/World/Market/Objects')
        with self.assertRaisesRegex(ValueError,'floor modules'):
            foot_floor_alignment(s,{'environment_root':'/World/Market'},'/World/Robot',{})


if __name__=='__main__':unittest.main()
