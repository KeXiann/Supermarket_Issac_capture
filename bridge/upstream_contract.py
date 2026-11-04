"""Verified upstream configuration shared by capture, export and physics replay."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "7e5fd9d0ae2f414249a3f2b83367dbb6d773f946"
MANIFEST = ROOT / "configs/upstream_manifest.json"


def verified_path(relative):
    manifest = json.loads(MANIFEST.read_text())
    if manifest["commit"] != COMMIT:
        raise ValueError("Upstream configuration commit mismatch")
    expected = {entry["path"]: entry["sha256"] for entry in manifest["files"]}
    path = ROOT / relative
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected[relative]:
        raise ValueError("Upstream configuration changed: " + relative)
    return path


def document(relative):
    return json.loads(verified_path(relative).read_text())


def model_contract():
    from sonic_contract import SONIC_CPP_MOTOR_ORDER, policy_parameters
    import numpy as np
    model = document("configs/control/sonic/g1_revo2_model.json")
    if model["joint_order"] != list(SONIC_CPP_MOTOR_ORDER):
        raise ValueError("Upstream model joint order differs from DDS")
    for key, source in (("default_angles", "default_angles_rad"),
                        ("kps", "stiffness_kp_nm_per_rad"),
                        ("kds", "damping_kd_nm_s_per_rad")):
        if not np.allclose(policy_parameters()[key], model["g1_29dof"][source], atol=1e-7, rtol=1e-6):
            raise ValueError("SONIC deployment differs from upstream model: " + key)
    if (model["simulation_dt_s"], model["control_frequency_hz"], model["decimation"]) != (.005, 50, 4):
        raise ValueError("Unsupported upstream control clock")
    return model


def task_initialization():
    import numpy as np
    initial = document("configs/task1/init/initial_scene.json")
    with np.load(verified_path("data/task1/init/initial_proprio.npz"), allow_pickle=False) as source:
        state = {key: source[key].copy() for key in source.files}
    return initial, state


def provenance():
    manifest = json.loads(MANIFEST.read_text())
    for entry in manifest["files"]:
        verified_path(entry["path"])
    return manifest


def check_physics_metadata(metadata):
    """Check actual runtime arrays, not just a named profile or a PD label."""
    import numpy as np
    physics = metadata.get("physics_contract") or {}
    if physics.get("profile") not in ("upstream_full", "upstream"):
        return metadata.get("pd_backend") == "implicit"
    model = model_contract()
    if (physics.get("commit") != COMMIT or metadata.get("pd_backend") != "external"
            or physics.get("pd_backend") != "external" or physics.get("joint_order") != model["joint_order"]):
        return False
    for key, actual in (("effort_limit_nm", metadata.get("effort_limits_motor_order")),
                        ("armature_kg_m2", physics.get("armature_kg_m2"))):
        expected = model["g1_29dof"]["joint_armature_kg_m2" if key == "armature_kg_m2" else key]
        if np.shape(actual) != (29,) or not np.allclose(actual, expected, atol=1e-7, rtol=1e-6):
            return False
    if physics.get('profile') == 'upstream_full':
        audit = physics.get('native_runtime_audit') or {}
        runtime = audit.get('runtime_versions') or {}
        if not (physics.get('full_model_equivalence') and audit.get('passed')
                and physics.get('checked_scene_configuration_equality') is True
                and (physics.get('scene_physics_audit') or {}).get('passed') is True
                and audit.get('commit') == COMMIT and audit.get('native_mimic_constraints') == 10
                and len(audit.get('articulations', [])) == 3
                and runtime.get('isaacsim') == '5.1.0.0' and runtime.get('isaaclab_release') == '2.3.2'):
            return False
    return True


def captured_physics(run):
    """Recover the recorded profile; never silently relabel an old episode."""
    run = Path(run)
    if (run / "bridge.json").is_file():
        metadata = json.loads((run / "bridge.json").read_text())
    else:
        from submission_replay import validate
        retained = validate(run)
        if retained is None:
            info = json.loads((run / "meta/info.json").read_text())
            physics = info.get("script_config", {}).get("physics_contract")
            if physics:
                return physics
            raise ValueError("Source physics not retained; cannot choose a replay PD backend/limits automatically")
        metadata = json.loads(retained[0]["bridge.json"])
    return metadata.get("physics_contract") or {
        "profile": "legacy", "pd_backend": metadata["pd_backend"],
        "effort_limit_nm": metadata["effort_limits_motor_order"],
        "source": "original recorded bridge.json; predates upstream profile",
    }


def replay_selection(bundle, profile=None, backend=None):
    """Reject accidental capture/replay drift before initializing Isaac Sim."""
    physics = json.loads(Path(bundle).with_suffix(".json").read_text()).get("physics_contract")
    if not physics:
        raise ValueError('Replay bundle lacks physics contract; rebuild it from a current-version recording')
    if physics["profile"] != 'upstream_full' or physics['pd_backend'] != 'external':
        raise ValueError('Historical physics profile removed; cannot replay old data using the latest model')
    if (profile is not None and profile != physics["profile"]) or (backend is not None and backend != physics["pd_backend"]):
        raise ValueError("Requested replay physics differs from the recorded capture")
    if physics["profile"] in ("upstream_full", "upstream") and physics.get("commit") != COMMIT:
        raise ValueError("Replay needs the capture's pinned upstream model version")
    return physics["profile"], physics["pd_backend"]
