from project_paths import SONIC_ROOT, code_path
#!/usr/bin/env python3
"""Capture selected code/config/model hashes before a local development run.

No environment secrets or sensor/network connections are read. Text sources
are copied for review; large models/assets are fingerprinted rather than copied.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata as package_metadata
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PROJECT = SONIC_ROOT
DEPLOY = PROJECT / "gear_sonic_deploy"


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def capture(run, component, profile="release", pose_file=None, parameters=None, scene_file=None):
    run = Path(run).resolve()
    destination = run / f"capture_{component}.json"
    if destination.exists():
        raise FileExistsError(f"capture manifest already exists: {destination}")
    common = [code_path(name) for name in ("capture_manifest.py", "start_pico_test.sh",
                                      "capture_session.py", "capture_utils.py", "batch_runtime.py", "batch_capture.py", "prepare_trajectory.py", "pose_replay.py",
                                      "export_capture_dataset.py", "prepare_mentor_sample.py", "smoke_capture.py",
                                      "recording_window.py", "assess_training_readiness.py",
                                      "runtime_cpu.py", "simulation_timing.py", "frame_output.py", "submission_replay.py")]
    common += [ROOT / "assets/collection_reference_2609" / name for name in
               ("reference.json", "info.template.json", "modality.template.json")]
    common += [ROOT / "assets/collection_reference_2609/gear_sonic/data" / name for name in
               ("features_sonic_vla.py", "body_alignment.py", "brainco_collection.py")]
    common += [ROOT / "assets/collection_reference_2609/dependencies/lerobot_utils.py",
               ROOT / "assets/collection_reference_2609/gear_sonic/pyproject.toml"]
    from upstream_contract import provenance, MANIFEST as UPSTREAM_MANIFEST
    common += [ROOT / "bridge/upstream_contract.py", ROOT / "bridge/data_standard.py", UPSTREAM_MANIFEST]
    common += [ROOT / entry["path"] for entry in provenance()["files"]]
    if component == "bridge":
        files = [code_path(name) for name in ["sonic_isaacsim_bridge.py", "bridge_core.py", "sonic_contract.py",
                                          "free_base_startup.py", "startup_alignment.py", "upstream_full_robot.py", "upstream_physics.py", "bridge_runtime.py", "bridge_recording.py", "reference_robot.py",
                                          "scene_support.py", "ego_camera.py", "camera_profile.py", "camera_intrinsics.py", "camera_extrinsics.py", "upstream_camera.py"]]
        from camera_intrinsics import DEFAULT_CALIBRATION
        from camera_extrinsics import DEFAULT_EXTRINSICS
        files += [DEFAULT_CALIBRATION]
        files += [DEFAULT_EXTRINSICS]
        from upstream_camera import MANIFEST
        files += [MANIFEST]
        files += [PROJECT / "gear_sonic/envs/manager_env/robots/g1.py",
                  DEPLOY / 'src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp',
                  PROJECT / "gear_sonic/utils/teleop/revo2_hand_control.py",
                  PROJECT / "gear_sonic/data/assets/robot_description/urdf/g1/main.urdf"]
        from reference_robot import SNAPSHOT, verify_snapshot
        reference = verify_snapshot()
        files += [SNAPSHOT / 'manifest.json']
        files += [SNAPSHOT / item['path'] for item in reference['files']]
        from upstream_contract import COMMIT
        files += [ROOT / "assets/upstream" / COMMIT[:12] / "scripts/scene_state.py"]
        files += [code_path(name) for name in ('pyproject.toml', 'uv.lock', '.runtime/uv_environment.json')
                  if (code_path(name)).is_file()]
    elif component == "sonic":
        files = [DEPLOY / f"policy/{profile}/{name}" for name in
                 ["model_encoder.onnx", "model_decoder.onnx", "observation_config.yaml"]]
        files += [DEPLOY / "planner/target_vel/V2/planner_sonic.onnx",
                  DEPLOY / "target/release/g1_deploy_onnx_ref",
                  DEPLOY / "src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp",
                  DEPLOY / "src/g1/g1_deploy_onnx_ref/include/robot_parameters.hpp",
                  DEPLOY / "src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp"]
        files += [ROOT / "bridge/run_sonic_with_debug.py", ROOT / "bridge/sonic_debug_recorder.py"]
    elif component == "manager":
        files = [PROJECT / "gear_sonic/scripts/pico_manager_thread_server.py"]
        files += sorted((PROJECT / "gear_sonic/utils/teleop").rglob("*.py"))
    else:
        raise ValueError(f"unknown component {component}")
    files += common
    if pose_file is not None:
        files += [Path(pose_file).resolve(), ROOT / "bridge/pose_replay.py"]
    if scene_file is not None:
        scene = Path(scene_file).resolve()
        files += [scene]
        camera_config = scene.with_suffix('.camera.json')
        if camera_config.is_file():
            files += [camera_config]
        if component == "bridge":
            sidecar = scene.with_suffix(".json")
            scene_info = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
            if scene_info.get("kind") == "pico_supermarket_scene_install":
                from setup_supermarket_scene import verify_manifest
                scene_info = verify_manifest(sidecar)
                files += [sidecar, ROOT / "bridge/setup_supermarket_scene.py"]
                files += [Path(entry["path"]) for entry in scene_info["dependency_files"]]
            else:
                robot_asset = scene.parents[1] / "robot/g1_revo2"
                files += [robot_asset / name for name in ("g1_revo2.usda", "g1_revo2.physics.usda",
                          "model/g1_revo2_assembly.usdc", "model/g1_revo2_assembly.json")]
    files = list(dict.fromkeys(files))
    # Missing required source must be visible before launching a process.
    entries = [fingerprint(path) for path in files]
    body_fk = PROJECT / 'gear_sonic/data/assets/robot_description/urdf/g1/main.urdf'
    for path, entry in zip(files, entries):
        if path.resolve() == body_fk.resolve():
            entry['role'] = 'body_fk_urdf'
    snapshot = run / "provenance" / component
    snapshot.mkdir(parents=True, exist_ok=False)
    for path, entry in zip(files, entries):
        if path.suffix in (".py", ".sh", ".hpp", ".yaml", ".urdf", ".json", ".usda", ".toml"):
            if path.stat().st_size > 8 * 1024 * 1024:
                entry['snapshot_omitted_reason'] = 'large pinned asset; full SHA256 and source path retained'
                continue
            target = snapshot / (entry["sha256"][:12] + "_" + path.name)
            shutil.copyfile(path, target)
            entry["snapshot_relative_to_run"] = str(target.relative_to(run))
    boot_id = Path("/proc/sys/kernel/random/boot_id")
    packages = {}
    for name in ("numpy", "scipy", "pyarrow", "pandas", "opencv-python", "pin", "torch", "isaacsim", "isaaclab"):
        try:
            packages[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            packages[name] = None
    result = {"schema_version": 1, "stage": "development", "formal_asset_pending": True,
              "component": component, "profile": profile, "created_utc": datetime.now(timezone.utc).isoformat(),
              "monotonic_s": time.monotonic(), "host_boot_id": boot_id.read_text().strip() if boot_id.exists() else None,
              "python_executable": sys.executable, "python_version": sys.version,
              "manifest_interpreter_packages": packages,
              "parameters": parameters or {}, "files": entries,
              "scope": "selected source/config/model files only; not a complete dependency or mesh-asset lockfile"}
    with destination.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--component", choices=["bridge", "sonic", "manager"], required=True)
    parser.add_argument("--profile", default="release", choices=["release", "low_latency"])
    parser.add_argument("--pose-file", type=Path)
    parser.add_argument("--scene-file", type=Path)
    args = parser.parse_args()
    result = capture(args.run, args.component, args.profile, args.pose_file, scene_file=args.scene_file)
    print(f"Captured {args.component} provenance: {len(result['files'])} files", flush=True)


if __name__ == "__main__":
    main()
