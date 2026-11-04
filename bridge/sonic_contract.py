from project_paths import SONIC_ROOT
"""SONIC DDS ordering contract shared by preflight and the Isaac bridge.

The C++ deployment source is the protocol authority for ``rt/lowcmd`` and
``rt/lowstate``.  The YAML ``WeakMotorJointIndex`` table in this checkout uses
another (yaw-first) order and is therefore only used as a named limits table.
"""
from pathlib import Path
import ast
import operator
import re


SONIC_CPP_MOTOR_ORDER = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)

CPP_HEADER = SONIC_ROOT / (
    "gear_sonic_deploy/"
    "src/g1/g1_deploy_onnx_ref/include/robot_parameters.hpp"
)
POLICY_HEADER = CPP_HEADER.with_name("policy_parameters.hpp")


def policy_parameters(path=POLICY_HEADER):
    """Read nominal C++ numeric arrays for local diagnostics, without Kit imports."""
    source = Path(path).read_text(encoding="utf-8")
    source = re.sub(r"/\*.*?\*/|//[^\n]*", "", source, flags=re.S)
    constants = {}
    operations = {ast.Add: operator.add, ast.Sub: operator.sub,
                  ast.Mult: operator.mul, ast.Div: operator.truediv}
    def number(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            return constants[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -number(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in operations:
            return operations[type(node.op)](number(node.left), number(node.right))
        raise ValueError("unsupported expression in C++ policy constants")
    def evaluate(expression):
        return number(ast.parse(expression.strip(), mode="eval").body)
    for name, expression in re.findall(r"const double (\w+)\s*=\s*([^;]+);", source):
        constants[name] = evaluate(expression)
    result = {}
    for name in ("default_angles", "kps", "kds"):
        match = re.search(r"const std::array<[^>]+>\s+" + name + r"\s*=\s*\{([^}]+)\}", source)
        if not match:
            raise RuntimeError(f"C++ policy array {name} missing")
        values = tuple(evaluate(value) for value in match.group(1).split(",") if value.strip())
        if len(values) != 29:
            raise RuntimeError(f"Expected 29 C++ values in {name}, got {len(values)}")
        result[name] = values
    return result


def yaml_names_by_index(config):
    mapping = config.get("WeakMotorJointIndex")
    if not isinstance(mapping, dict) or not mapping:
        raise RuntimeError("SONIC config is missing WeakMotorJointIndex")
    index_to_name = {}
    for name, raw_index in mapping.items():
        index = int(raw_index)
        if index in index_to_name:
            raise RuntimeError(f"Duplicate YAML motor index {index}")
        index_to_name[index] = str(name)
    if sorted(index_to_name) != list(range(29)):
        raise RuntimeError(f"WeakMotorJointIndex must define indices 0..28; got {sorted(index_to_name)}")
    return tuple(index_to_name[i] for i in range(29))


def reorder_yaml_values(values, config):
    """Convert a YAML index-array to the C++ DDS motor order by joint name."""
    if len(values) != 29:
        raise RuntimeError(f"Expected a 29-element YAML motor array, got {len(values)}")
    by_name = dict(zip(yaml_names_by_index(config), values))
    missing = set(SONIC_CPP_MOTOR_ORDER) - set(by_name)
    if missing:
        raise RuntimeError(f"YAML motor table is missing joints: {sorted(missing)}")
    return tuple(by_name[name] for name in SONIC_CPP_MOTOR_ORDER)


def validate_cpp_header(path=CPP_HEADER):
    """Fail closed if the local deployment header no longer matches this map."""
    text = Path(path).read_text(encoding="utf-8")
    enum_tokens = (
        "LeftHipPitch = 0", "LeftHipRoll = 1", "LeftHipYaw = 2",
        "RightHipPitch = 6", "RightHipRoll = 7", "RightHipYaw = 8",
        "WaistYaw = 12", "LeftShoulderPitch = 15", "RightShoulderPitch = 22",
        "RightWristYaw = 28",
    )
    missing = [token for token in enum_tokens if token not in text]
    if missing:
        raise RuntimeError("C++ SONIC motor contract changed; missing: " + ", ".join(missing))
