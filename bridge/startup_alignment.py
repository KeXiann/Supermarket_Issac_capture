"""One-time initial pose/contact and joint-frame checks; no physics stepping.

These operate before DDS starts. They never reset/support a moving free base.
"""
import math
import numpy as np


def assert_nominal_body(joint_names, positions, motor_names, nominal, atol=1e-6):
    names = list(joint_names)
    actual = np.asarray(positions)[[names.index(name) for name in motor_names]]
    error = np.abs(actual - np.asarray(nominal))
    if not np.isfinite(actual).all() or error.max() > atol:
        raise ValueError(f'initial body pose differs from SONIC nominal by {error.max():.6f} rad')
    return {'max_body_nominal_error_rad': float(error.max()), 'motor_count': len(motor_names)}


def pose_matrix(position, quaternion_wxyz):
    from pxr import Gf
    q = np.asarray(quaternion_wxyz, dtype=float)
    m = Gf.Matrix4d(1).SetRotate(Gf.Quatd(float(q[0]), Gf.Vec3d(*q[1:])))
    m.SetTranslateOnly(Gf.Vec3d(*np.asarray(position, dtype=float)))
    return m


def audit_joint_frames(stage, roots, body_poses, position_tolerance_m=1e-4, angle_tolerance_deg=.1):
    """Use actual PhysX actor poses, not stale USD authored transforms."""
    from pxr import Gf, Usd, UsdPhysics
    rows = []
    for root in roots:
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
            if not (prim.IsA(UsdPhysics.FixedJoint) or prim.IsA(UsdPhysics.RevoluteJoint)):
                continue
            joint = UsdPhysics.Joint(prim)
            if joint.GetJointEnabledAttr().Get() is False:
                continue
            frames = []
            for i in (0, 1):
                paths = (joint.GetBody0Rel() if i == 0 else joint.GetBody1Rel()).GetTargets()
                if len(paths) != 1:
                    raise ValueError(f'free-base joint must bind two bodies: {prim.GetPath()}')
                name = paths[0].name
                if name not in body_poses:
                    raise ValueError(f'joint binds a body outside the articulation: {paths[0]}')
                position = (joint.GetLocalPos0Attr() if i == 0 else joint.GetLocalPos1Attr()).Get()
                rotation = Gf.Quatd((joint.GetLocalRot0Attr() if i == 0 else joint.GetLocalRot1Attr()).Get())
                frame = Gf.Matrix4d(1).SetRotate(rotation)
                frame.SetTranslateOnly(Gf.Vec3d(position))
                frames.append(frame * body_poses[name])
            relative = frames[1] * frames[0].GetInverse()
            gap = float(relative.ExtractTranslation().GetLength())
            rotation = Gf.Rotation(relative.ExtractRotationQuat())
            if prim.IsA(UsdPhysics.FixedJoint):
                angle = float(rotation.GetAngle())
            else:
                axis = Gf.Vec3d(*{'X': (1,0,0), 'Y': (0,1,0), 'Z': (0,0,1)}[
                    UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get()])
                dot = Gf.Dot(rotation.TransformDir(axis), axis)
                angle = math.degrees(math.acos(float(np.clip(dot, -1, 1))))
            rows.append({'joint': str(prim.GetPath()), 'type': prim.GetTypeName(),
                         'pivot_gap_m': gap, 'constraint_rotation_error_deg': angle})
    if not rows:
        raise ValueError('no articulation joints audited')
    worst_gap = max(row['pivot_gap_m'] for row in rows)
    worst_angle = max(row['constraint_rotation_error_deg'] for row in rows)
    if worst_gap > position_tolerance_m or worst_angle > angle_tolerance_deg:
        raise ValueError(f'initial joint-frame mismatch: gap={worst_gap:.6g} m angle={worst_angle:.6g} deg')
    return {'joint_count': len(rows), 'max_pivot_gap_m': worst_gap,
            'max_constraint_rotation_error_deg': worst_angle, 'joints': rows}


def foot_floor_alignment(stage, profile, robot_path, body_poses, clearance_m=.001, *, legacy_bounds=False):
    """Compute the initial vertical shift from real foot collision shapes.

Explicitly find authored floor modules below both feet; all other objects
are excluded. The replacement visuals are never used as contact geometry.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    # Imported URDF colliders use purpose=guide and are hidden from renders.
    cache = UsdGeom.BBoxCache(0, ['default', 'render', 'proxy', 'guide'])
    floors = []
    objects = stage.GetPrimAtPath(profile['environment_root'] + '/Objects')
    for p in objects.GetChildren():
        if not p.GetName().lower().startswith('floor'):
            continue
        bounds = cache.ComputeWorldBound(p).ComputeAlignedRange()
        if not bounds.IsEmpty():
            floors.append((str(p.GetPath()), bounds))
    if not floors:
        raise ValueError('cannot identify authored floor modules for initial foot alignment')
    feet = []
    for side in ('left','right'):
        name = side + '_ankle_roll_link'
        link = stage.GetPrimAtPath(robot_path + '/' + name)
        authored = UsdGeom.Xformable(link).ComputeLocalToWorldTransform(0)
        shape_bottoms = []
        # URDF import stores primitive colliders below instanceable Xforms.
        # Default traversal silently skips their instance-proxy children.
        for p in Usd.PrimRange(link, Usd.TraverseInstanceProxies()):
            if not p.HasAPI(UsdPhysics.CollisionAPI) or not UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get():
                continue
            # Transform each collider's own box to the real FK link frame.
            # Imported primitive bounds include the URDF cylinder->capsule
            # conversion, unlike the original visual mesh bounds.
            # ComputeLocalBound already includes this prim's transform.
            # The explicit transform below must start from untransformed bounds.
            local = (cache.ComputeLocalBound(p) if legacy_bounds else cache.ComputeUntransformedBound(p)).ComputeAlignedRange()
            if local.IsEmpty():
                continue
            transform = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(0) * authored.GetInverse() * body_poses[name]
            points = [transform.Transform(Gf.Vec3d(x,y,z))
                      for x in (local.GetMin()[0],local.GetMax()[0])
                      for y in (local.GetMin()[1],local.GetMax()[1])
                      for z in (local.GetMin()[2],local.GetMax()[2])]
            shape_bottoms.append(min(float(point[2]) for point in points))
        if not shape_bottoms:
            raise ValueError(f'no foot collision geometry found: {name}')
        xy = body_poses[name].ExtractTranslation()
        beneath = [(path, bounds) for path, bounds in floors
                   if bounds.GetMin()[0] <= xy[0] <= bounds.GetMax()[0]
                   and bounds.GetMin()[1] <= xy[1] <= bounds.GetMax()[1]]
        if not beneath:
            raise ValueError(f'no floor beneath {name} at {tuple(xy)}')
        floor_top = max(float(bounds.GetMax()[2]) for _, bounds in beneath)
        feet.append({'side':side, 'sole_bottom_before_m':min(shape_bottoms),
                     'floor_top_m':floor_top, 'gap_before_m':min(shape_bottoms)-floor_top,
                     'floor_modules':[path for path,_ in beneath]})
    # Keep the lower sole just above contact, without initial penetration.
    dz = clearance_m - min(foot['gap_before_m'] for foot in feet)
    if abs(dz) > .15:
        raise ValueError(f'initial floor adjustment unexpectedly large: {dz:.4f} m')
    for foot in feet:
        foot['gap_after_m'] = foot['gap_before_m'] + dz
    return {'vertical_shift_m':dz, 'clearance_m':clearance_m, 'feet':feet}
