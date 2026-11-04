"""Native pinned USD robot, with a flat recording interface over three articulations.

The adapter only concatenates tensors and routes commands. It does not merge
articulations, replace collision geometry, or implement finger coupling.
"""
import importlib.util
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from upstream_contract import COMMIT, document, verified_path
from reference_robot import ROBOT, verify_snapshot


def physics_inventory(stage, root):
    """Comparable source properties; exclude placement and initial motion state."""
    from pxr import Usd, UsdPhysics
    result = {}
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        collision = prim.HasAPI(UsdPhysics.CollisionAPI)
        if not (collision or prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.IsA(UsdPhysics.Joint)
                or prim.HasAPI(UsdPhysics.MaterialAPI)):
            continue
        attributes = {}
        for attr in prim.GetAuthoredAttributes():
            name = attr.GetName()
            if name in ('physics:velocity', 'physics:angularVelocity') or name.startswith('state:') or 'targetPosition' in name or 'targetVelocity' in name:
                continue
            if name.startswith(('physics:', 'physx', 'drive:')) or (collision and name.startswith(('xformOp', 'points', 'faceVertex', 'radius', 'height', 'size', 'axis'))):
                attributes[name] = str(attr.Get()).replace(root + '/', '/ROBOT/')
        relations = {r.GetName(): [str(p).replace(root + '/', '/ROBOT/') for p in r.GetTargets()]
                     for r in prim.GetRelationships() if r.GetName().startswith(('physics:', 'physx', 'material:'))}
        result[str(prim.GetPath())[len(root):]] = {'type': prim.GetTypeName(), 'attributes': attributes,
                                                   'relationships': relations,
                                                   'applied_schemas': sorted(prim.GetAppliedSchemas())}
    return result


def verify_native_asset(stage, root):
    from pxr import Usd, UsdPhysics, PhysxSchema
    reference = Usd.Stage.Open(str(ROBOT))
    expected = physics_inventory(reference, '/G1Revo2')
    actual = physics_inventory(stage, root)
    if actual != expected:
        changed = sorted(k for k in set(actual) | set(expected) if actual.get(k) != expected.get(k))
        raise ValueError('Runtime USD differs from the pinned full physical asset: ' + repr(changed[:10]))
    roots, mimic, collisions = [], [], []
    for p in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if p.HasAPI(UsdPhysics.ArticulationRootAPI):
            roots.append(str(p.GetPath()))
        if p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get():
            collisions.append(str(p.GetPath()))
        for axis in ('rotX', 'rotY'):
            if p.HasAPI(PhysxSchema.PhysxMimicJointAPI, axis):
                mimic.append({'joint': str(p.GetPath()), 'axis': axis})
    if len(roots) != 3 or len(mimic) != 10:
        raise ValueError('Unexpected canonical articulation/mimic inventory')
    return {'sha256': hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest(),
            'matched_source_prims': len(actual), 'articulation_roots': roots,
            'mimic_constraints': mimic, 'enabled_collision_count': len(collisions)}


def pinned_pose_writer():
    source = Path(__file__).resolve().parents[1] / 'assets/upstream' / COMMIT[:12]
    spec = importlib.util.spec_from_file_location('_pinned_scene_state', source / 'scripts/scene_state.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PoseWriter


def runtime_versions():
    from importlib.metadata import version
    import isaaclab
    release_file = Path(isaaclab.__file__).resolve().parents[3] / 'VERSION'
    return {'isaacsim': version('isaacsim'), 'isaaclab_package': version('isaaclab'),
            'isaaclab_release': release_file.read_text().strip() if release_file.is_file() else None,
            'torch': version('torch'), 'source_isaacsim': '5.1.0.0'}


def restore_authored_task_world(stage, environment_root):
    """Restore all 111 named task bodies, including saved background velocities."""
    from pxr import UsdPhysics, Gf
    mapping = document('configs/task1/init/state_mapping.json')
    poses, velocities = {}, {}
    with np.load(verified_path('data/task1/init/initial_state.npz'), allow_pickle=False) as state:
        for key, path in mapping['extra_rigid_bodies'].items():
            path = path.replace('/World/envs/env_0/Room', environment_root)
            if not path.startswith(environment_root + '/'):
                raise ValueError('Unsupported canonical background body: ' + path)
            value = state[key + '__transform_xyzw'][0]
            poses[path] = (value[:3], value[[6, 3, 4, 5]])
            velocities[path] = state[key + '__velocity'][0].copy()
        for asset, name in (('object', 'basket'), ('table', 'T2')):
            path = environment_root + '/Objects/' + name
            poses[path] = (state[asset + '__body_pos_w'][0, 0], state[asset + '__body_quat_w'][0, 0])
            velocities[path] = state[asset + '__body_vel_w'][0, 0].copy()
        error = pinned_pose_writer()(stage, list(poses)).apply(poses)
    for path, value in velocities.items():
        api = UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(path))
        if not api:
            raise ValueError('Canonical world body missing: ' + path)
        api.CreateVelocityAttr(Gf.Vec3f(*map(float, value[:3])))
        api.CreateAngularVelocityAttr(Gf.Vec3f(*map(float, np.rad2deg(value[3:]))))
    return velocities, {'body_count': len(poses), 'pose_authoring_max_error': error,
                        'source': 'data/task1/init/initial_state.npz', 'velocities': 'recorded, not zeroed'}


class KinematicTree:
    """The same joint-frame FK used by the upstream initialization builder."""

    def __init__(self, stage, root):
        from pxr import Usd, UsdPhysics
        from scipy.spatial.transform import Rotation
        self.root = root
        self.edges = []
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
            if not prim.IsA(UsdPhysics.Joint):
                continue
            joint = UsdPhysics.Joint(prim)
            parents, children = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
            if len(parents) != 1 or len(children) != 1:
                raise ValueError('Incomplete native joint: ' + str(prim.GetPath()))
            frames = []
            for i in (0, 1):
                m = np.eye(4)
                q = getattr(joint, f'GetLocalRot{i}Attr')().Get()
                m[:3, :3] = Rotation.from_quat([*q.GetImaginary(), q.GetReal()]).as_matrix()
                m[:3, 3] = getattr(joint, f'GetLocalPos{i}Attr')().Get()
                frames.append(m)
            axis = UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get() if prim.IsA(UsdPhysics.RevoluteJoint) else None
            self.edges.append((str(parents[0]), str(children[0]), prim.GetName(), axis, frames[0], np.linalg.inv(frames[1])))
        self.bodies = {str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath(root))
                       if p.HasAPI(UsdPhysics.RigidBodyAPI)}

    def poses(self, root_state, joints):
        from scipy.spatial.transform import Rotation
        m = np.eye(4)
        m[:3, 3] = root_state[:3]
        m[:3, :3] = Rotation.from_quat(np.asarray(root_state[3:7])[[1, 2, 3, 0]]).as_matrix()
        matrices = {self.root + '/Robot/pelvis': m}
        pending = list(self.edges)
        while pending:
            remaining = []
            for parent, child, name, axis, frame0, inv1 in pending:
                if parent not in matrices:
                    remaining.append((parent, child, name, axis, frame0, inv1))
                    continue
                motion = np.eye(4)
                if axis:
                    v = np.zeros(3)
                    v['XYZ'.index(axis)] = joints[name]
                    motion[:3, :3] = Rotation.from_rotvec(v).as_matrix()
                matrices[child] = matrices[parent] @ frame0 @ motion @ inv1
            if len(remaining) == len(pending):
                raise ValueError('Disconnected upstream robot kinematic tree')
            pending = remaining
        if set(matrices) != self.bodies:
            raise ValueError('Native robot FK body coverage differs from USD')
        return {path: (m[:3, 3], Rotation.from_matrix(m[:3, :3]).as_quat()[[3, 0, 1, 2]])
                for path, m in matrices.items()}


class CombinedData:
    def __init__(self, owner):
        self.owner = owner
        self.cache = {}

    def __getattr__(self, name):
        import torch
        if name.startswith(('joint_', 'default_joint_', 'soft_joint_', 'body_')):
            if name not in self.cache:
                self.cache[name] = torch.cat([getattr(a.data, name) for a in self.owner.parts], dim=1)
            return self.cache[name]
        return getattr(self.owner.parts[0].data, name)


class NativeRobot:
    def __init__(self, parts, tree, cfg, inventory=None):
        self.parts, self.tree, self.cfg = parts, tree, cfg
        self.data = CombinedData(self)
        self._routes = {}
        self.inventory = inventory

    @property
    def device(self):
        return self.parts[0].device

    @property
    def joint_names(self):
        return [n for a in self.parts for n in a.joint_names]

    @property
    def body_names(self):
        return [n for a in self.parts for n in a.body_names]

    def update(self, dt):
        self.data.cache.clear()
        for a in self.parts:
            a.update(dt)

    def reset(self):
        self.data.cache.clear()
        for a in self.parts:
            a.reset()

    def write_data_to_sim(self):
        for a in self.parts:
            a.write_data_to_sim()

    def _route(self, method, *values, joint_ids=None, **kwargs):
        import torch
        key = None if joint_ids is None else tuple(joint_ids.tolist() if torch.is_tensor(joint_ids) else joint_ids)
        if key not in self._routes:
            ids = list(range(len(self.joint_names))) if key is None else key
            offset, routes = 0, []
            for a in self.parts:
                selected = [(i, j - offset) for i, j in enumerate(ids) if offset <= j < offset + len(a.joint_names)]
                if selected:
                    source, local = map(list, zip(*selected))
                    routes.append((a, source, local))
                offset += len(a.joint_names)
            self._routes[key] = routes
        for a, source, local in self._routes[key]:
            args = [v[..., source] if torch.is_tensor(v) and v.ndim else v for v in values]
            getattr(a, method)(*args, joint_ids=local, **kwargs)
        if method.startswith('write_joint_'):
            self.data.cache.clear()

    def __getattr__(self, name):
        if name.startswith(('set_joint_', 'write_joint_')):
            return lambda *v, **kw: self._route(name, *v, **kw)
        raise AttributeError(name)

    def _align_hand_roots(self):
        """Only explicit state resets may reposition roots; never called per step."""
        import torch
        body = self.parts[0]
        root = torch.cat((body.data.root_link_pos_w, body.data.root_link_quat_w,
                          body.data.root_link_lin_vel_w, body.data.root_link_ang_vel_w), dim=-1)
        joints = dict(zip(self.joint_names, self.data.joint_pos[0].cpu().numpy()))
        poses = self.tree.poses(root[0].cpu().numpy(), joints)
        for a, side in zip(self.parts[1:], ('left', 'right')):
            p, q = poses[f'{self.tree.root}/BrainCo_{side}/{side}_base_link']
            state = root.clone()
            state[0, :7] = torch.as_tensor(np.r_[p, q], device=self.device, dtype=state.dtype)
            # Transport the root velocity through the fixed wrist mount.
            state[:, 7:10] += torch.linalg.cross(root[:, 10:13], state[:, :3] - root[:, :3])
            a.write_root_link_state_to_sim(state)

    def write_root_link_state_to_sim(self, state):
        self.parts[0].write_root_link_state_to_sim(state)
        self._align_hand_roots()

    def write_joint_state_to_sim(self, positions, velocities, joint_ids=None):
        self._route('write_joint_state_to_sim', positions, velocities, joint_ids=joint_ids)
        self._align_hand_roots()

    def runtime_audit(self, stage):
        """Read actual PhysX mass/inertia/DOFs and check against the source prims."""
        from pxr import UsdPhysics
        from scipy.spatial.transform import Rotation
        report = []
        hand_pd = document('configs/control/revo2_pd.json')
        for a in self.parts:
            view = a.root_physx_view
            masses = view.get_masses()[0].cpu().numpy()
            inertias = view.get_inertias()[0].cpu().numpy().reshape(-1, 3, 3)
            rows = []
            for i, name in enumerate(a.body_names):
                prim = stage.GetPrimAtPath(a.cfg.prim_path + '/' + name)
                api = UsdPhysics.MassAPI(prim)
                mass, diag, q = api.GetMassAttr().Get(), api.GetDiagonalInertiaAttr().Get(), api.GetPrincipalAxesAttr().Get()
                axes = Rotation.from_quat([*q.GetImaginary(), q.GetReal()]).as_matrix()
                expected = axes @ np.diag(diag) @ axes.T
                if not np.isclose(masses[i], mass, atol=1e-6, rtol=1e-5) or not np.allclose(inertias[i], expected, atol=1e-7, rtol=1e-4):
                    raise ValueError(f'Native PhysX mass/inertia differs from USD: {prim.GetPath()}, mass={masses[i]}/{mass}, inertia error={np.max(abs(inertias[i]-expected))}')
                rows.append({'name': name, 'mass_kg': float(masses[i]), 'inertia_max_error': float(np.max(abs(inertias[i]-expected)))})
            gains = {}
            if a is not self.parts[0]:
                for i, name in enumerate(a.joint_names):
                    passive = '_distal_' in name
                    expected = hand_pd['passive_mimic' if passive else 'active']
                    values = {k: float(getattr(a.data, 'joint_' + k)[0, i]) for k in ('stiffness', 'damping', 'armature')}
                    for k, v in values.items():
                        if not np.isclose(v, expected[k], rtol=1e-5, atol=1e-8):
                            raise ValueError(f'Native hand {name} {k} differs: {v} != {expected[k]}')
                    values['effort_limit'] = float(a.data.joint_effort_limits[0, i])
                    if passive:
                        values['velocity_limit_rad_s'] = float(a.data.joint_vel_limits[0, i])
                        if not np.isclose(values['velocity_limit_rad_s'], expected['velocity_limit_by_joint_rad_s'][name], rtol=1e-5):
                            raise ValueError('Passive velocity limit differs from resolved USD contract: ' + name)
                    if passive and values['effort_limit'] != 0.:
                        raise ValueError('Passive native mimic joint must have zero drive force')
                    gains[name] = values
            report.append({'prim': a.cfg.prim_path, 'joints': list(a.joint_names), 'bodies': rows, 'hand_drives': gains})
        return {'passed': True, 'articulations': report,
                'native_mimic_constraints': len(self.inventory['mimic_constraints']),
                'source_asset_inventory': self.inventory,
                'source': str(ROBOT), 'commit': COMMIT,
                'runtime_versions': runtime_versions(),
                'dynamic_stability_validated': False,
                'scope': 'canonical USD physics plus upstream initialization builder solver and Revo2 armature; body PD applied explicitly at 200 Hz'}


def prepare_native_stage(stage, root, root_position, root_quaternion, task_state):
    """Author the production initialization in memory; no SimulationApp or physics."""
    from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema
    ACTIVE_SUFFIXES = ('thumb_metacarpal_joint', 'thumb_proximal_joint', 'index_proximal_joint',
                       'middle_proximal_joint', 'ring_proximal_joint', 'pinky_proximal_joint')
    verify_snapshot()
    inventory = verify_native_asset(stage, root)
    if task_state is None:
        raise ValueError('Full upstream robot requires the canonical task initialization')
    q = dict(zip(task_state['joint_names'].tolist(), task_state['observation.state'][0].tolist()))
    info = {}
    for side in ('left', 'right'):
        q.update(zip(task_state[side + '_hand_joint_names'].tolist(), task_state[side + '_hand_joint_position'].tolist()))
        active = [f'{side}_{s}' for s in ACTIVE_SUFFIXES]
        passive = [f'{side}_{f}_distal_joint' for f in ('thumb', 'index', 'middle', 'ring', 'pinky')]
        path = f'{root}/BrainCo_{side}'
        info[side] = {'prim': path, 'mount': path + '/WristMount', 'active_joint_names': active,
                      'mimic_joint_names': passive, 'initial_joint_positions_rad': [q[n] for n in active + passive],
                      'distal_coupling': 'unmodified native PhysX mimic constraints; passive drive effort/stiffness/damping zero'}
        for name in passive:
            joint = stage.GetPrimAtPath(path + '/joints/' + name)
            axis = 'rotX' if 'thumb' in name else 'rotY'
            mimic = PhysxSchema.PhysxMimicJointAPI(joint, axis)
            if not joint.HasAPI(PhysxSchema.PhysxMimicJointAPI, axis) or not mimic.GetReferenceJointRel().GetTargets():
                raise ValueError('Native mimic constraint missing: ' + str(joint.GetPath()))
        if not UsdPhysics.FixedJoint(stage.GetPrimAtPath(path + '/WristMount')).GetExcludeFromArticulationAttr().Get():
            raise ValueError('Upstream wrist articulation topology changed')
    tree = KinematicTree(stage, root)
    poses = tree.poses(np.r_[root_position, root_quaternion], q)
    error = pinned_pose_writer()(stage, list(poses)).apply(poses)
    if error > 1e-8:
        raise ValueError('Native source FK pose authoring failed')
    hand_pd = document('configs/control/revo2_pd.json')
    for side in ('left', 'right'):
        for name in info[side]['mimic_joint_names']:
            prim = stage.GetPrimAtPath(info[side]['prim'] + '/joints/' + name)
            actual = math.radians(PhysxSchema.PhysxJointAPI(prim).GetMaxJointVelocityAttr().Get())
            if not np.isclose(actual, hand_pd['passive_mimic']['velocity_limit_by_joint_rad_s'][name], rtol=1e-7):
                raise ValueError('Local hand configuration differs from native USD: ' + name)
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if prim.IsA(UsdPhysics.RevoluteJoint):
            name = prim.GetName()
            state = PhysxSchema.JointStateAPI.Apply(prim, 'angular')
            state.CreatePositionAttr(math.degrees(q[name]))
            state.CreateVelocityAttr(0.)
            if prim.HasAPI(UsdPhysics.DriveAPI, 'angular'):
                UsdPhysics.DriveAPI(prim, 'angular').CreateTargetPositionAttr(math.degrees(q[name]))
            if '/BrainCo_' in str(prim.GetPath()):
                PhysxSchema.PhysxJointAPI.Apply(prim).CreateArmatureAttr(hand_pd['active']['armature'])
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            api = PhysxSchema.PhysxArticulationAPI.Apply(prim)
            api.CreateSolverPositionIterationCountAttr(32)
            api.CreateSolverVelocityIterationCountAttr(4)
    appearance = {'profile': 'kinetra_full_usd_' + COMMIT[:12], 'commit': COMMIT, 'asset': str(ROBOT),
                  'method': 'native source USD assembly, collision geometry, masses, inertias and mimic constraints',
                  'visual_mesh_count': sum(p.IsA(UsdGeom.Mesh) for p in Usd.PrimRange(stage.GetPrimAtPath(root)))}
    return tree, info, appearance, inventory, q, poses, hand_pd


def create_native_robot(stage, root, root_position, root_quaternion, task_state, source_model):
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    tree, info, appearance, inventory, q, poses, hand_pd = prepare_native_stage(
        stage, root, root_position, root_quaternion, task_state)
    params = source_model['g1_29dof']
    body_cfg = ArticulationCfg(prim_path=root + '/Robot', spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(pos=tuple(root_position), rot=tuple(root_quaternion),
                    joint_pos={n: q[n] for n in source_model['joint_order']}, joint_vel={'.*': 0.}),
        actuators={'body': ImplicitActuatorCfg(joint_names_expr=source_model['joint_order'], stiffness=0., damping=0.,
                    effort_limit_sim=dict(zip(source_model['joint_order'], params['effort_limit_nm'])),
                    armature=dict(zip(source_model['joint_order'], params['joint_armature_kg_m2'])))})
    parts = [Articulation(body_cfg)]
    for side in ('left', 'right'):
        p, quat = poses[f'{root}/BrainCo_{side}/{side}_base_link']
        names = info[side]['active_joint_names'] + info[side]['mimic_joint_names']
        hand_cfg = ArticulationCfg(prim_path=info[side]['prim'], spawn=None,
            init_state=ArticulationCfg.InitialStateCfg(pos=tuple(p), rot=tuple(quat), joint_pos={n: q[n] for n in names}),
            actuators={'native': ImplicitActuatorCfg(joint_names_expr=['.*'], stiffness=None, damping=None,
                        armature=hand_pd['active']['armature'])})
        parts.append(Articulation(hand_cfg))
    return NativeRobot(parts, tree, body_cfg, inventory), body_cfg, info, appearance
