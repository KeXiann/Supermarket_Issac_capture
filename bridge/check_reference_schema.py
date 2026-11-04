#!/usr/bin/env python3
"""Rebuild schema dictionaries from pinned upstream functions, without sensors.

Only the named pure schema builders are evaluated. Robot/Sonic/LFS assets and
upstream module imports are not needed. The body order is checked independently
against upstream body_alignment.py, rather than guessed from articulation order.
"""
import ast
import hashlib
import json
from pathlib import Path

from sonic_contract import SONIC_CPP_MOTOR_ORDER

REFERENCE = Path(__file__).resolve().parents[1] / "assets/collection_reference_2609"


def builders(path, names, namespace):
    tree = ast.parse((REFERENCE / path).read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise ValueError(f"missing pinned schema builders in {path}")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)


def contract():
    body_tree = ast.parse((REFERENCE / "gear_sonic/data/body_alignment.py").read_text())
    body_assignment = next(node for node in body_tree.body if isinstance(node, ast.Assign)
                           and any(isinstance(t, ast.Name) and t.id == "BODY_JOINT_NAMES" for t in node.targets))
    namespace = {}
    exec(compile(ast.Module(body=[body_assignment], type_ignores=[]), "body_alignment", "exec"), namespace)
    if namespace["BODY_JOINT_NAMES"] != list(SONIC_CPP_MOTOR_ORDER):
        raise ValueError("local DDS order differs from the pinned upstream body contract")

    class SchemaRobot:
        # Upstream G1+ThreeFinger inserts seven left-hand joints after the
        # left arm. BrainCo's builder then removes both Dex3 groups from body.
        joint_names = (list(SONIC_CPP_MOTOR_ORDER[:22]) + [f"left_hand_{i}" for i in range(7)]
                       + list(SONIC_CPP_MOTOR_ORDER[22:]) + [f"right_hand_{i}" for i in range(7)])
        num_joints = 43
        groups = {"left_leg": list(range(6)), "right_leg": list(range(6, 12)),
                  "waist": list(range(12, 15)), "left_arm": list(range(15, 22)),
                  "left_hand": list(range(22, 29)), "right_arm": list(range(29, 36)),
                  "right_hand": list(range(36, 43)), "body": list(range(22)) + list(range(29, 36))}

        def get_joint_group_indices(self, group):
            return self.groups[group]

    namespace.update(RobotModel=SchemaRobot, EGO_VIEW_HEIGHT=360, EGO_VIEW_WIDTH=640,
                     _JOINT_GROUPS_FOR_STATE=["left_leg", "right_leg", "waist", "left_arm", "left_hand", "right_arm", "right_hand"],
                     SIDES=("left", "right"), MOTOR_NAMES=[f"{side}_brainco_motor_{i}" for side in ("left", "right") for i in range(6)])
    builders("gear_sonic/data/features_sonic_vla.py",
             ("_get_joint_group_slices", "get_modality_config_sonic_vla", "get_features_sonic_vla"), namespace)
    builders("gear_sonic/data/body_alignment.py", ("configure_body_alignment_schema",), namespace)
    builders("gear_sonic/data/brainco_collection.py", ("configure_brainco_schema",), namespace)
    robot = SchemaRobot()
    features = namespace["get_features_sonic_vla"](robot)
    modality = namespace["get_modality_config_sonic_vla"](robot)
    namespace["configure_body_alignment_schema"](features)
    namespace["configure_brainco_schema"](features, modality, robot)
    # The five LeRobot standard columns are added by LeRobotDatasetMetadata.
    for key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        features[key] = {"dtype": "float32" if key == "timestamp" else "int64", "shape": [1], "names": None}
    return json.loads(json.dumps(features)), json.loads(json.dumps(modality))


def check():
    reference = json.loads((REFERENCE / "reference.json").read_text())
    for name, digest in reference["sha256"].items():
        if hashlib.sha256((REFERENCE / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"pinned upstream snapshot differs: {name}")
    dependency = reference["dependency_sources"]["lerobot_utils"]
    if hashlib.sha256((REFERENCE / dependency["path"]).read_bytes()).hexdigest() != dependency["sha256"]:
        raise ValueError("pinned LeRobot helper differs")
    features, modality = contract()
    info = json.loads((REFERENCE / "info.template.json").read_text())
    if features != info["features"] or modality != json.loads((REFERENCE / "modality.template.json").read_text()):
        raise ValueError("templates differ from pinned upstream schema builders")
    return {"reference_commit": reference["commit"],
            "features": len(features), "numeric_columns": len(features) - 1,
            "video": "640x360 BrainCo collection profile", "schema_matches": True}


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
