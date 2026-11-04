"""Authored scene selection; all lighting/camera overrides stay in session memory."""
def render_config():
    import isaaclab.sim as sim_utils
    return sim_utils.RenderCfg(
        rendering_mode="quality", enable_shadows=True,
        enable_ambient_occlusion=True, enable_reflections=True,
        enable_global_illumination=True, enable_dlssg=False,
        carb_settings={"rtx.sceneDb.ambientLightIntensity": 0.0})



SUPERMARKET_EYE = (6.35, 3.2, 2.35)
SUPERMARKET_TARGET = (5.35, 4.95, .9)


def restore_world_transforms(view, transforms, dynamic_indices, velocities=None):
    """Restore every pose, but only write velocities to dynamic PhysX actors."""
    import numpy as np
    targets = np.asarray(transforms, dtype=np.float32)
    view.set_transforms(targets.copy(), np.arange(len(targets), dtype=np.uint32))
    if len(dynamic_indices):
        # Tensor setters require a full-view buffer even for selected indices.
        values = np.zeros((len(targets), 6), dtype=np.float32) if velocities is None else np.asarray(velocities, dtype=np.float32)
        if values.shape != (len(targets), 6) or not np.isfinite(values).all():
            raise ValueError('Invalid world initialization velocities')
        view.set_velocities(values.copy(), dynamic_indices)


def scene_profile_for_stage(stage):
    from pxr import UsdPhysics
    name, root, label = 'supermarket', '/World/Supermarket', 'supermarket'
    if not stage.GetPrimAtPath(root):
        raise ValueError('Only the current supermarket scene is supported')
    scenes = [str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    expected = root+'/PhysicsScene'
    if scenes != [expected]:
        raise ValueError(f'Expected a single authored physics scene {expected}; found {scenes}')
    return {'name': name, 'label': label, 'environment_root': root,
            'physics_scene_path': expected,
            'source_commit': stage.GetRootLayer().customLayerData.get('upstream_commit')}


def setup_scene_view(stage, profile, *, activate_viewport=False):
    from upstream_camera import COMMIT
    if profile.get('source_commit') != COMMIT:
        raise ValueError('Scene commit differs from the current pinned version')
    return setup_upstream_view(stage, activate_viewport=activate_viewport)


def setup_upstream_view(stage, *, activate_viewport=False):
    """Exact upstream task-1 external camera; keep authored scene lighting."""
    import numpy as np
    from pxr import Gf, UsdGeom
    from upstream_camera import load_contract
    from camera_intrinsics import load_rectified_intrinsics, usd_projection_parameters
    contract = load_contract()
    record = next(r for r in contract['initial_scene']['view_records'] if r['name'] == 'external')
    stage.SetEditTarget(stage.GetSessionLayer())
    path = '/World/TeleopRender/Overview'
    camera = UsdGeom.Camera.Define(stage, path)
    camera.CreateFocalLengthAttr(14.)
    k = load_rectified_intrinsics()
    projection = usd_projection_parameters(k, k['width'], k['height'], 14.)
    for name, value in [('HorizontalAperture', projection['horizontal_aperture_mm']),
                        ('VerticalAperture', projection['vertical_aperture_mm']),
                        ('HorizontalApertureOffset', projection['horizontal_aperture_offset_mm']),
                        ('VerticalApertureOffset', projection['vertical_aperture_offset_mm'])]:
        getattr(camera, 'Create' + name + 'Attr')(value)
    camera.CreateClippingRangeAttr(Gf.Vec2f(*contract['render']['clipping_range_m']))
    cv = np.asarray(record['world_T_camera_cv'])
    usd = cv @ np.diag([1., -1., -1., 1.])
    UsdGeom.Xformable(camera).MakeMatrixXform().Set(Gf.Matrix4d(*usd.T.reshape(-1).tolist()))
    if activate_viewport:
        from omni.kit.viewport.utility import get_active_viewport
        viewport = get_active_viewport()
        if viewport is None:
            raise RuntimeError('No active viewport for supermarket overview')
        viewport.set_active_camera(path)
    return {'profile': 'kinetra_task1_external_7e5fd9d', 'camera_path': path,
            'source_commit': contract['commit'], **record,
            'lighting': 'unchanged upstream scene; no extra fill lights', 'session_layer_only': True}
