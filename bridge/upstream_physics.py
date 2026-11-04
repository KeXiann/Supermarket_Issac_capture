"""CPU physics configuration matching the pinned upstream inspection entry.

USD operations are shared by production startup and the offline audit.
"""

PHYSX_SETTINGS = {
    'enable_ccd': True,
    'enable_stabilization': False,
    'bounce_threshold_velocity': 0.0,
    'solver_type': 1,
}


def apply_scene_physics(stage, scene_path):
    from pxr import UsdPhysics, PhysxSchema
    scene = stage.GetPrimAtPath(scene_path)
    if not scene or not scene.IsA(UsdPhysics.Scene):
        raise ValueError('Missing authored physics scene: ' + scene_path)
    api = PhysxSchema.PhysxSceneAPI.Apply(scene)
    api.CreateEnableCCDAttr(True)
    api.CreateEnableStabilizationAttr(False)
    api.CreateBounceThresholdAttr(0.0)
    api.CreateSolverTypeAttr('TGS')
    api.CreateEnableGPUDynamicsAttr(False)
    api.CreateBroadphaseTypeAttr('MBP')
    # Isaac Lab always creates/binds a scene default material. The upstream
    # inspector does not. Remove that binding before PhysX initializes; retain
    # all object-level physics material bindings unchanged.
    relation = scene.GetRelationship('material:binding:physics')
    if relation:
        relation.SetTargets([])
    generated = stage.GetPrimAtPath(scene_path + '/defaultMaterial')
    if generated:
        generated.SetActive(False)
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
    return check_scene_physics(stage, scene_path)


def check_scene_physics(stage, scene_path):
    from pxr import UsdPhysics, PhysxSchema
    scene = stage.GetPrimAtPath(scene_path)
    api = PhysxSchema.PhysxSceneAPI(scene)
    expected = {'enable_ccd': True, 'enable_stabilization': False,
                'bounce_threshold_velocity': 0.0, 'solver_type': 'TGS',
                'gpu_dynamics': False, 'broadphase': 'MBP'}
    actual = dict(zip(expected, [api.GetEnableCCDAttr().Get(), api.GetEnableStabilizationAttr().Get(),
        api.GetBounceThresholdAttr().Get(), api.GetSolverTypeAttr().Get(),
        api.GetEnableGPUDynamicsAttr().Get(), api.GetBroadphaseTypeAttr().Get()]))
    if actual != expected:
        raise ValueError('Physics scene settings differ from upstream: ' + repr(actual))
    relation = scene.GetRelationship('material:binding:physics')
    if relation and relation.GetTargets():
        raise ValueError('Unexpected global contact material binding')
    count = 0
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            if (not prim.HasAPI(PhysxSchema.PhysxContactReportAPI)
                    or PhysxSchema.PhysxContactReportAPI(prim).GetThresholdAttr().Get() != 0.0):
                raise ValueError('Missing source contact report configuration: ' + str(prim.GetPath()))
            count += 1
    return {'passed': True, **actual, 'scene_default_material_binding': [],
            'rigid_bodies_with_zero_contact_report_threshold': count,
            'scope': 'authored scene settings, material bindings and contact reporting; not dynamic behavior'}
