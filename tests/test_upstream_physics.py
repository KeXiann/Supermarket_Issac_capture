"""Offline USD checks for source physics settings and material preservation."""
import unittest

try:
    from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, UsdShade, Plug
    from pathlib import Path
    for path in Path(PhysxSchema.__file__).resolve().parents[2].glob('plugins/*/resources/plugInfo.json'):
        Plug.Registry().RegisterPlugins(str(path))
except ImportError:
    Usd = None

from upstream_physics import apply_scene_physics, check_scene_physics


@unittest.skipIf(Usd is None, 'requires Isaac USD bindings')
class SourcePhysicsTests(unittest.TestCase):
    def fixture(self):
        stage = Usd.Stage.CreateInMemory()
        scene = UsdPhysics.Scene.Define(stage, '/World/Physics').GetPrim()
        default = UsdShade.Material.Define(stage, '/World/Physics/defaultMaterial')
        UsdPhysics.MaterialAPI.Apply(default.GetPrim()).CreateStaticFrictionAttr(1.0)
        UsdShade.MaterialBindingAPI.Apply(scene).Bind(default, materialPurpose='physics')
        body = UsdGeom.Xform.Define(stage, '/World/Body').GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
        material = UsdShade.Material.Define(stage, '/World/ObjectMaterial')
        UsdPhysics.MaterialAPI.Apply(material.GetPrim()).CreateStaticFrictionAttr(.65)
        UsdShade.MaterialBindingAPI.Apply(body).Bind(material, materialPurpose='physics')
        return stage, body

    def test_removes_lab_default_without_changing_object_materials(self):
        stage, body = self.fixture()
        before = body.GetRelationship('material:binding:physics').GetTargets()
        result = apply_scene_physics(stage, '/World/Physics')
        self.assertTrue(result['passed'])
        self.assertTrue(result['enable_ccd'])
        self.assertEqual(result['bounce_threshold_velocity'], 0.)
        self.assertEqual(result['rigid_bodies_with_zero_contact_report_threshold'], 1)
        self.assertEqual(body.GetRelationship('material:binding:physics').GetTargets(), before)
        self.assertAlmostEqual(UsdPhysics.MaterialAPI(stage.GetPrimAtPath('/World/ObjectMaterial')).GetStaticFrictionAttr().Get(), .65)
        self.assertFalse(stage.GetPrimAtPath('/World/Physics/defaultMaterial').IsActive())
        self.assertEqual(apply_scene_physics(stage, '/World/Physics'), result)

    def test_rejects_a_later_setting_or_contact_report_override(self):
        stage, body = self.fixture()
        apply_scene_physics(stage, '/World/Physics')
        PhysxSchema.PhysxSceneAPI(stage.GetPrimAtPath('/World/Physics')).GetEnableCCDAttr().Set(False)
        with self.assertRaisesRegex(ValueError, 'settings differ'):
            check_scene_physics(stage, '/World/Physics')
        apply_scene_physics(stage, '/World/Physics')
        PhysxSchema.PhysxContactReportAPI(body).GetThresholdAttr().Set(1.)
        with self.assertRaisesRegex(ValueError, 'contact report'):
            check_scene_physics(stage, '/World/Physics')
