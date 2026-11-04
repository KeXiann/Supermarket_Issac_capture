#!/usr/bin/env python3
"""Compose a new PICO supermarket USD without editing either source project.

USD-only command: run with the installed Isaac USD bindings, not SimulationApp.
The supermarket layout remains authored; the proven local G1/Revo2 assembly
is placed beside checkout T2, facing the original basket.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from capture_manifest import fingerprint

ROOT = Path(__file__).resolve().parents[1]
from upstream_camera import SOURCE, SCENE as OUTPUT, load_contract
from reference_robot import ROBOT
# Current pinned installation. The older pico_supermarket.usda remains the
# source for historical captures and is intentionally never overwritten.
# OUTPUT is versioned by upstream_camera; historical installs stay immutable.


def verify_manifest(path):
    manifest = json.loads(path.read_text())
    for entry in manifest['dependency_files']:
        if fingerprint(Path(entry['path']))['sha256'] != entry['sha256']:
            raise ValueError(f'Installed scene dependency changed: {entry["path"]}')
    return manifest


def build(source=SOURCE, output=OUTPUT):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils
    source, output = Path(source).resolve(), Path(output).resolve()
    sidecar = output.with_suffix('.json')
    if output.exists() or sidecar.exists():
        if not output.is_file() or not sidecar.is_file():
            raise FileExistsError('Partial scene install; preserve it and use a NEW --output path')
        return verify_manifest(sidecar)
    market = source/'assets/supermarket/scene.usda'
    for p in [market, ROBOT, source/'configs/scene/objects.json']:
        if not p.is_file():
            raise FileNotFoundError(p)
    checksum = json.loads((source/'configs/assets/assets.sha256.json').read_text())
    market_files = [source/p for p in checksum if p.startswith('assets/supermarket/')]
    for path in market_files:
        if fingerprint(path)['sha256'] != checksum[str(path.relative_to(source))]:
            raise ValueError(f'Source asset checksum mismatch: {path}')
    market_stage = Usd.Stage.Open(str(market))
    floor = market_stage.GetPrimAtPath('/World/Objects/floor')
    floor_top = float(UsdGeom.BBoxCache(0, ['default','render']).ComputeWorldBound(floor).ComputeAlignedRange().GetMax()[2])
    robot_stage = Usd.Stage.Open(str(ROBOT))
    cache = UsdGeom.BBoxCache(0, ['default','render'])
    root = '/'+robot_stage.GetDefaultPrim().GetName()+'/Robot'
    foot_bottom = min(float(cache.ComputeWorldBound(robot_stage.GetPrimAtPath(
        f'{root}/{side}_ankle_roll_link')).ComputeAlignedRange().GetMin()[2]) for side in ['left','right'])
    contract = load_contract()
    initial = contract['initial_scene']
    origin = tuple(row[3] for row in initial['root_T'][:3])
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageMetersPerUnit(stage, 1.); UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    world = UsdGeom.Xform.Define(stage, '/World'); stage.SetDefaultPrim(world.GetPrim())
    stage.GetRootLayer().customLayerData = {'layout_id': 'kinetra_task1_initialization',
                                                 'upstream_commit': contract['commit']}
    env = UsdGeom.Xform.Define(stage, '/World/Supermarket')
    env.GetPrim().GetReferences().AddReference(Path(os.path.relpath(market, output.parent)).as_posix())
    # Task init references the body-centred basket wrapper, not the module wrapper
    # (which includes an extra 90-degree local transform). Match the upstream
    # build_initialization_scene.py object_asset and kinematic table exactly.
    basket = stage.GetPrimAtPath('/World/Supermarket/Objects/basket')
    basket.GetReferences().SetReferences([Sdf.Reference(
        Path(os.path.relpath(source/'assets/supermarket/basket.usda', output.parent)).as_posix())])
    # The original module reference arrives through the environment composition
    # arc, so its children survive a stronger local reference list. Deactivate
    # that whole branch to avoid a duplicate nested rigid body and visual mesh.
    stage.OverridePrim(str(basket.GetPath()) + '/Content').SetActive(False)
    UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath('/World/Supermarket/Objects/T2')).CreateKinematicEnabledAttr(True)
    # Apply the upstream task-1 transforms exactly; keep all shelf/collision assets intact.
    for name, source_path in [('basket', '/World/Object'), ('T2', '/World/Table')]:
        matrix = initial['static_transforms'][source_path]
        prim = stage.GetPrimAtPath('/World/Supermarket/Objects/' + name)
        if not prim:
            raise ValueError('Missing task initialization object: ' + name)
        UsdGeom.Xformable(prim).MakeMatrixXform().Set(
            Gf.Matrix4d(*[matrix[j][i] for i in range(4) for j in range(4)]))
    robot = UsdGeom.Xform.Define(stage, '/World/Robot')
    robot.GetPrim().GetReferences().AddReference(Path(os.path.relpath(ROBOT, output.parent)).as_posix())
    robot.MakeMatrixXform().Set(Gf.Matrix4d(*[initial['root_T'][j][i] for i in range(4) for j in range(4)]))
    robot.GetPrim().SetCustomData({'role': 'Existing PICO SONIC G1/Revo2; free-base proxy replaces authored body at runtime',
        'placement': 'Exact upstream task1 root_T and named initial_proprio; no vertical relocation'})
    stage.GetRootLayer().Save()
    layers, assets, missing = UsdUtils.ComputeAllDependencies(str(output))
    if missing:
        raise RuntimeError(f'Unresolved scene dependencies: {list(missing)}')
    from scene_support import scene_profile_for_stage
    profile = scene_profile_for_stage(stage)
    environment = stage.GetPrimAtPath('/World/Supermarket')
    count = {'visual_meshes': 0, 'dynamic_rigid_bodies': 0, 'fixed_joints': 0}
    for p in Usd.PrimRange(environment):
        count['visual_meshes'] += int(p.IsA(UsdGeom.Mesh) and UsdGeom.Imageable(p).GetPurposeAttr().Get() != 'guide')
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            rb = UsdPhysics.RigidBodyAPI(p)
            count['dynamic_rigid_bodies'] += int(rb.GetRigidBodyEnabledAttr().Get() and not rb.GetKinematicEnabledAttr().Get())
        count['fixed_joints'] += int(p.IsA(UsdPhysics.FixedJoint))
    count['modules'] = len(stage.GetPrimAtPath('/World/Supermarket/Objects').GetChildren())
    dependencies = [Path(layer.realPath) for layer in layers if layer.realPath]
    dependencies += [Path(p) for p in assets]
    dependencies += [source/'configs/scene/objects.json', source/'configs/assets/assets.sha256.json',
                     ROBOT.parent/'g1_revo2.physics.usda',
                     source/'configs/task1/init/initial_scene.json', source/'configs/collection/camera_stereo.json',
                     source/'configs/collection/render_config.json']
    dependencies = list(dict.fromkeys(p.resolve() for p in dependencies))
    manifest = {'schema_version': 1, 'kind': 'pico_supermarket_scene_install',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'scene': str(output),
        'source_url': 'https://github.com/Kinetra-Humanoids/GPT6-Assist-Humanoid-Robot-Mobile-Manipulation-Data-Generation',
        'source_commit': subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
        'source_repository': str(source), 'source_layout_modified': True,
        'source_layout_changes': 'Exact upstream task1 basket wrapper, table kinematic flag and task1 transforms',
        'task_initialization': 'configs/task1/init/initial_scene.json',
        'task_initialization_transforms': initial['static_transforms'],
        'robot_z_semantics': 'exact upstream root_T; runtime restores named initial_proprio without relocating the root',
        'robot_asset': str(ROBOT), 'robot_root_position_m': origin, 'robot_yaw_deg': 180.,
        'floor_top_m': floor_top, 'source_robot_foot_bottom_m': foot_bottom,
        'scene_profile': profile, 'environment_counts': count,
        'missing_dependencies': list(missing), 'dependency_files': [fingerprint(p) for p in dependencies]}
    sidecar.write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = build(args.source, args.output)
    print(json.dumps({k:v for k,v in result.items() if k != 'dependency_files'}, indent=2))
