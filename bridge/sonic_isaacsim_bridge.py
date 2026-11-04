from project_paths import SONIC_ROOT
#!/usr/bin/env python3
"""Experimental Unitree G1 (29-DOF) Isaac Lab 2.3.2 / SONIC DDS bridge.

ONLY for local Isaac Sim simulation (DDS domain 0, interface 'lo').
No MuJoCo Python runtime is imported or used. The 29-motor DDS index order is
validated against SONIC's C++ G1JointIndex contract. URDF and MuJoCo XML are
read only to verify joint inventories, never to infer the DDS index order.

Launch with uv run --frozen --offline python this_file.py [--headless] ...
Test in --mode listen first. --mode drive enables external torque commands.
"""
import argparse
import json
import math
import os
import re
import signal
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

from bridge_runtime import bootstrap, default_profile
bootstrap()

from runtime_cpu import configure_cpu_affinity
CPU_RUNTIME = configure_cpu_affinity()

import numpy as np

from sonic_contract import (
    CPP_HEADER,
    SONIC_CPP_MOTOR_ORDER,
    reorder_yaml_values,
    validate_cpp_header,
    yaml_names_by_index,
    policy_parameters,
)
from bridge_core import (
    CommandRejected,
    MotorCommand,
    compute_torque,
    decode_lowcmd,
    first_command_error,
    guard_joint_effort,
    guard_implicit_command,
    implicit_feedback_budget,
    hold_torque,
    slew_pd_command,
    inverse_quaternion_matrix,
)
from bridge_recording import Recorder
from simulation_timing import LoopPerformance, PhysicsPacer, StartupPostureReadiness
from upstream_contract import model_contract, task_initialization, COMMIT as CONFIG_COMMIT
from camera_profile import resolve_camera_profile
from capture_utils import atomic_json
from free_base_startup import PolicyStartupGate, base_tilt_radians
from startup_alignment import (assert_nominal_body, pose_matrix, audit_joint_frames,
                               foot_floor_alignment)
from ego_camera import EgoCameraCapture, setup_ego_camera, update_ego_camera_transform
from scene_support import render_config as authored_render_config, scene_profile_for_stage, setup_scene_view

PROJECT = SONIC_ROOT
WORKDIR = Path(__file__).resolve().parents[1]
URDF = PROJECT / "gear_sonic/data/assets/robot_description/urdf/g1/main.urdf"
MJCF = PROJECT / "gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml"
CFG = PROJECT / "gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml"
SDK = PROJECT / "external_dependencies/unitree_sdk2_python"


def motor_names_from_urdf(path):
    root = ET.parse(path).getroot()
    return [j.attrib["name"] for j in root.findall("joint")
            if j.attrib.get("type") not in ("fixed", "floating")]


def motor_names_from_sonic_config(config, urdf_path):
    """Return names in the actual C++ DDS motor order, validated by source."""
    validate_cpp_header(CPP_HEADER)
    yaml_names = yaml_names_by_index(config)
    names = list(SONIC_CPP_MOTOR_ORDER)
    urdf_names = motor_names_from_urdf(urdf_path)
    if len(urdf_names) != 29 or len(set(urdf_names)) != 29:
        raise RuntimeError(f"Expected 29 unique movable joints in SONIC URDF, got {len(urdf_names)}")
    if set(names) != set(urdf_names):
        missing_in_urdf = sorted(set(names) - set(urdf_names))
        missing_in_config = sorted(set(urdf_names) - set(names))
        raise RuntimeError(
            "SONIC motor map and URDF joint inventory differ. "
            f"Not in URDF: {missing_in_urdf}; not in motor map: {missing_in_config}"
        )
    if tuple(yaml_names) != SONIC_CPP_MOTOR_ORDER:
        print("[WARN] YAML WeakMotorJointIndex is not the C++ DDS order; using the "
              "verified C++ robot_parameters.hpp contract.")
    return names


def preflight():
    import yaml
    if not SDK.is_dir():
        raise FileNotFoundError("GR00T customized unitree_sdk2_python missing: " + str(SDK))
    with CFG.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if (int(config["NUM_MOTORS"]) != 29 or int(config["DOMAIN_ID"]) != 0
            or config["INTERFACE"] != "lo" or float(config["SIMULATE_DT"]) != 0.005):
        raise RuntimeError("Expected G1 29DOF; DDS domain=0; interface=lo; dt=0.005. Inspect YAML.")
    # Use the local C++ deployment contract, not YAML/URDF declaration order.
    motor_names = motor_names_from_sonic_config(config, URDF)
    if len(motor_names) != 29 or len(set(motor_names)) != 29:
        raise RuntimeError(f"Expected exactly 29 unique SONIC motor joint names, got {len(motor_names)}")
    # Validate only that the read-only MJCF contains the same joint inventory.
    # XML declaration order is NOT the DDS motor order. No MuJoCo runtime is used.
    if MJCF.is_file():
        scene = ET.parse(MJCF).getroot()
        mjcf_names = [j.attrib["name"] for j in scene.findall(".//worldbody//joint")
                      if j.attrib.get("name") in set(motor_names)]
        if len(mjcf_names) != 29 or set(mjcf_names) != set(motor_names):
            raise RuntimeError("SONIC motor map and official sim XML joint inventory differ. "
                               "Do not publish DDS until the joint inventory is reviewed.\n"
                               f"Motor map: {motor_names}\nMJCF joints: {mjcf_names}")
    else:
        print("[WARN] Official MJCF XML not found; URDF order only (review motor mapping manually).")
    for k in ("MOTOR_KP", "MOTOR_KD", "motor_pos_lower_limit_list", "motor_pos_upper_limit_list"):
        if len(config[k]) != 29:
            raise RuntimeError(f"Unexpected {k} length: {len(config[k])}")
    if len(config.get("motor_vel_limit_list", [])) != 29:
        raise RuntimeError("Unexpected motor_vel_limit_list length")
    return config, motor_names


parser = argparse.ArgumentParser(description="Isaac Sim local-only experimental SONIC G1 DDS bridge")
parser.add_argument("--mode", choices=("listen", "drive"), default="listen",
                    help="listen = publish observations, ignore SONIC actuation; drive = accept lowcmd torque")
parser.set_defaults(fixed_base=False)
parser.add_argument("--wait-for-policy-log", type=Path,
                    help="Free-base startup: publish actual initial state without stepping physics until fresh SONIC policy actuation exists")
parser.add_argument("--startup-timeout", type=float, default=180.,
                    help="Maximum wall seconds waiting for free-base policy takeover")
parser.add_argument("--stop-request-file", type=Path,
                    help="Local Manager shutdown handshake: end physics before SONIC disconnects")
parser.add_argument('--batch-control-file', type=Path,
                    help='Local batch coordinator: rotate recordings and reset the scene without closing Isaac')
parser.add_argument("--max-base-tilt-deg", type=float, default=60.,
                    help="Stop free-base simulation when pelvis tilt exceeds this angle")
parser.add_argument("--target-policy", choices=("sonic-pd", "strict"), default="sonic-pd",
                    help="sonic-pd: validate PD equilibria separately from physical joint limits; strict: require targets inside physical limits")
parser.add_argument("--physics-profile", choices=("upstream_full",), default=None,
                    help="Complete current native USD; retired physics profiles are not supported")
parser.add_argument("--pd-backend", choices=("external",), default=None,
                    help="external: explicit PD effort; implicit: PhysX solves PD once, only feed-forward is external")
parser.add_argument("--extra-waist-damping", type=float, default=0.,
                    help="Additional measured-velocity damping for the three waist joints in fixed-base diagnostics")
parser.add_argument("--pd-target-slew", choices=("joint-speed", "none"), default="joint-speed",
                    help="Optional PD equilibrium slew; actual joint effort/velocity guards always remain active")
parser.add_argument("--duration", type=float, default=40.0, help="Simulation seconds before exit (0 = unlimited)")
parser.add_argument("--watchdog", type=float, default=0.3, help="Seconds without lowcmd before fallback pose hold")
parser.add_argument("--kp-scale", type=float, default=1.0, help="Scale external motor kp in drive mode")
parser.add_argument("--kd-scale", type=float, default=1.0, help="Scale external motor kd in drive mode")
parser.add_argument("--torque-scale", type=float, default=1.0,
                    help="Scale SONIC feed-forward torque only; kp/kd scales affect feedback")
parser.add_argument("--hold-scale", type=float, default=0.5, help="Fallback pose holding PD scale")
parser.add_argument("--record", type=str, default="", help="NPZ trajectory output path (optional)")
parser.add_argument("--record-every", type=int, default=4, help="Capture every N physics steps (4 -> 50 Hz)")
parser.add_argument("--profile-timing", action="store_true", default=os.environ.get('PICO_PROFILE_TIMING') == '1',
                    help="Record measured loop phase costs in bridge metadata")
parser.add_argument("--ego-render-schedule", choices=("upstream", "wall", "legacy"), default="upstream",
                    help="upstream: 25 Hz pre-action physics schedule; wall/legacy: explicitly requested diagnostic overrides")
parser.add_argument("--render-every", type=int, default=4,
                    help="Viewport/legacy render cadence; camera wall schedule instead uses ego-camera-fps")
parser.add_argument("--checkpoint-seconds", type=float, default=30.0,
                    help="Atomically checkpoint an active recording at this interval (0 disables)")
parser.add_argument("--max-initial-error", type=float, default=0.75,
                    help="Maximum first target/current joint error in rad before latching a fault")
parser.add_argument("--max-command-step", type=float, default=0.75,
                    help="Strict: reject larger frame jumps; sonic-pd: cap applied target step with joint velocity limits")
parser.add_argument("--no-pace", action="store_true", help="Do NOT pace the physics steps to wall time")
parser.add_argument("--no-hand-state", action="store_true", help="Disable DDS hand state publishers")
from upstream_camera import SCENE
parser.add_argument("--scene", type=Path, default=SCENE,
                    help="Current pinned task-1 scene with the complete native robot")
parser.add_argument("--scene-robot-root", type=str, default="/World/Robot",
                    help="Root xform containing the complete native robot")
parser.add_argument("--ego-camera", action="store_true",
                    help="Attach a simulated first-person camera to G1 torso_link")
parser.add_argument("--ego-camera-width", type=int, default=640,
                    help="Simulated ego camera width in pixels")
parser.add_argument("--ego-camera-height", type=int, default=360,
                    help="Simulated ego camera height in pixels")
parser.add_argument("--ego-camera-fps", type=int, default=25,
                    help="Ego camera output rate for video/ZMQ/RTP")
parser.add_argument("--ego-camera-config", type=Path,
                    help="Camera JSON configuration; otherwise use scene-adjacent .camera.json when present")
parser.add_argument("--ego-camera-intrinsics", default=None,
                    help="Rectified calibration JSON; default is supplied real-camera 640x360 K; 'virtual' restores legacy projection")
parser.add_argument("--ego-camera-extrinsics", default=None,
                    help="Photo/CAD rigid torso mount JSON; default is the current pinned upstream estimate; 'virtual' restores historical world-up scene view")
parser.add_argument("--ego-camera-pitch-down-deg", type=float, default=None,
                    help="Override the scene camera downward pitch (legacy default: 45 degrees)")
parser.add_argument("--ego-camera-mount", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "Z"), help="Camera offset from torso_link in meters")
parser.add_argument("--ego-camera-focal-length-mm", type=float, default=None,
                    help="Nominal USD focal length; with calibrated K the film apertures scale to preserve pixel intrinsics")
parser.add_argument("--ego-camera-record", type=str, default="",
                    help="Optional ego_view MP4 path")
parser.add_argument("--ego-camera-zmq-port", type=int, default=0,
                    help="Publish GR00T ego_view msgpack frames on this port (0 disables)")
parser.add_argument("--ego-camera-rtp-host", type=str, default="",
                    help="camera_viz RTP receiver host (empty disables RTP)")
parser.add_argument("--ego-camera-rtp-port", type=int, default=0,
                    help="camera_viz RTP receiver port (0 disables RTP)")
parser.add_argument("--dexterous-hands", action="store_true", default=True,
                    help="Enable authored Revo2 five-finger hands and rt/dex3 command mapping in scene mode")
parser.add_argument('--local-token-replay',type=Path,
                    help='Offline g1_dyn bundle: fresh-state closed loop, opens NO DDS connections')

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device='cpu')
args = parser.parse_args()
if args.device != 'cpu':
    parser.error('The pinned upstream inspection contract uses CPU physics')
if args.scene.resolve() != SCENE.resolve():
    parser.error("Only the current pinned task-1 scene is supported")
if args.local_token_replay:
    from upstream_contract import replay_selection
    try:
        args.physics_profile, args.pd_backend = replay_selection(args.local_token_replay, args.physics_profile, args.pd_backend)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
else:
    args.physics_profile = args.physics_profile or default_profile(args.scene)
    args.pd_backend = args.pd_backend or "external"
try:
    camera_profile = resolve_camera_profile(args.scene, config=args.ego_camera_config,
        mount=args.ego_camera_mount, pitch=args.ego_camera_pitch_down_deg, focal=args.ego_camera_focal_length_mm,
        intrinsics_config=args.ego_camera_intrinsics,extrinsics_config=args.ego_camera_extrinsics)
    if args.ego_camera and camera_profile['intrinsics']:
        from camera_intrinsics import validate_intrinsics
        validate_intrinsics(camera_profile['intrinsics'], args.ego_camera_width, args.ego_camera_height)
except (ValueError, KeyError, OSError, TypeError) as exc:
    parser.error(f"Invalid ego camera configuration: {exc}")
args.ego_camera_mount = camera_profile['mount_position_m']
args.ego_camera_pitch_down_deg = camera_profile['pitch_down_deg']
args.ego_camera_focal_length_mm = camera_profile['focal_length_mm']
upstream_sampling = args.ego_camera and args.ego_render_schedule == 'upstream'
if upstream_sampling and (args.ego_camera_fps != camera_profile['capture_contract']['render']['camera_native_hz']
                          or args.record_every != 4):
    parser.error('Upstream sampling requires 25 Hz camera and 50 Hz state recording; choose wall/legacy only for diagnostics')
if upstream_sampling:
    from upstream_camera import launcher_options, validate_capture_profile
    try:
        validate_capture_profile(camera_profile, args.ego_camera_width, args.ego_camera_height)
    except ValueError as exc:
        parser.error(str(exc))
    options = launcher_options(camera_profile['capture_contract'])
    args.renderer = options['renderer']
    args.kit_args = (args.kit_args + ' ' + options['kit_args']).strip()

if (args.duration < 0 or args.watchdog <= 0 or args.record_every < 1 or args.render_every < 1
        or args.checkpoint_seconds < 0 or args.max_initial_error <= 0
        or args.max_command_step <= 0):
    parser.error("duration>=0, watchdog>0, record-every>=1, checkpoint>=0, error thresholds>0")
if not 0.0 < args.torque_scale <= 1.0:
    parser.error("torque-scale must be in (0,1]")
if not 0.0 <= args.hold_scale <= 1.0:
    parser.error("hold-scale must be in [0,1]")
if not 0.0 < args.kp_scale <= 1.0 or not 0.0 < args.kd_scale <= 1.0:
    parser.error("kp-scale and kd-scale must be in (0,1] during initial testing")
if not 0 <= args.extra_waist_damping <= 20 or (args.extra_waist_damping and not args.fixed_base):
    parser.error("extra-waist-damping must be in [0,20] and is only supported with --fixed-base")
pass
if args.wait_for_policy_log and (args.fixed_base or args.mode != "drive"):
    parser.error("wait-for-policy-log requires free-base drive mode")
if args.startup_timeout <= 0 or not 10 <= args.max_base_tilt_deg <= 90:
    parser.error("startup-timeout must be positive and max-base-tilt-deg must be in [10,90]")
if args.scene is not None and not args.scene.is_file():
    parser.error(f"Missing scene: {args.scene}")
pass
if args.dexterous_hands and args.scene is None:
    parser.error("--dexterous-hands requires a scene containing the authored Revo2 hands")
if args.dexterous_hands and args.no_hand_state:
    parser.error("Revo2 requires measured hand feedback; do not combine --dexterous-hands with --no-hand-state")
if args.ego_camera and (args.ego_camera_width < 16 or args.ego_camera_height < 16
                         or args.ego_camera_fps < 1 or args.ego_camera_fps > 120):
    parser.error("ego camera dimensions must be >=16 and fps must be in [1,120]")
if (not math.isfinite(args.ego_camera_pitch_down_deg) or not 0 <= args.ego_camera_pitch_down_deg < 85
        or not all(math.isfinite(v) for v in args.ego_camera_mount)
        or not math.isfinite(args.ego_camera_focal_length_mm) or args.ego_camera_focal_length_mm <= 0):
    parser.error("ego camera requires finite xyz mount, downward pitch in [0,85), positive focal length")
if args.ego_camera_rtp_port < 0 or args.ego_camera_zmq_port < 0:
    parser.error("ego camera ports must be >=0")
if args.ego_camera_rtp_port and not args.ego_camera_rtp_host:
    parser.error("--ego-camera-rtp-port requires --ego-camera-rtp-host")
if args.ego_camera:
    # Replicator camera products need the Kit camera extension even in
    # headless mode.  AppLauncher adds this attribute to the namespace.
    args.enable_cameras = True

# Preflight uses only the standard library + YAML; do before booting Kit.
config, motor_names = preflight()

# Start Isaac Sim before importing Isaac Lab simulator modules or DDS bindings.
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import omni.usd
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import quat_apply_inverse
from pxr import Gf, Usd, UsdGeom, UsdPhysics

# Ensure this user's repository and customized SDK take precedence.
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(SDK))
from gear_sonic.utils.teleop.simulation_stop import SimulationStopGate
from gear_sonic.utils.teleop.revo2_hand_control import (
    ACTIVE_SUFFIXES, controller_hand_command, dex3_to_revo2, revo2_to_dex3, revo2_full_target,
)

# Import the customized SDK from the same GR00T checkout.
import unitree_sdk2py
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__LowState_ as LowState_default,
    unitree_hg_msg_dds__OdoState_ as OdoState_default,
    unitree_hg_msg_dds__IMUState_ as IMUState_default,
    unitree_hg_msg_dds__HandState_ as HandState_default,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (
    LowState_, LowCmd_, OdoState_, IMUState_, HandState_, HandCmd_
)

SIM_DT = 0.005


def as_np(tensor):
    return tensor.detach().cpu().numpy().astype(np.float64, copy=True)


def hand_state(hand, indices, is_left):
    q = as_np(hand.data.joint_pos[0])[indices]
    dq = as_np(hand.data.joint_vel[0])[indices]
    return revo2_to_dex3(q, is_left), revo2_to_dex3(dq, is_left, velocity=True)


class DdsBridge:
    """Small protocol adapter; DDS callback never interacts with Isaac Sim API."""
    def __init__(self, position_low, position_high, velocity_limits, dummy_hands=True,
                 position_policy="sonic-pd"):
        self.lock = threading.Lock()
        self.cmd = None
        self.cmd_when = 0.0
        self.command_count = 0
        self.reject_count = 0
        self.last_reject = ""
        self.last_raw_q = np.full(29, np.nan)
        self.last_raw_valid = False
        self.outside_physical_limit_count = 0
        self.position_policy = position_policy
        self.reject_categories = Counter()
        self.reject_motors = Counter()
        self.position_low = np.asarray(position_low, dtype=np.float64)
        self.position_high = np.asarray(position_high, dtype=np.float64)
        self.velocity_limits = np.asarray(velocity_limits, dtype=np.float64)
        self.low_state = LowState_default()
        self.odo = OdoState_default()
        self.imu = IMUState_default()
        self.low_pub = ChannelPublisher("rt/lowstate", LowState_)
        self.odo_pub = ChannelPublisher("rt/odostate", OdoState_)
        self.imu_pub = ChannelPublisher("rt/secondary_imu", IMUState_)
        for p in (self.low_pub, self.odo_pub, self.imu_pub):
            p.Init()
        self.hands = dummy_hands
        if dummy_hands:
            self.lhand = HandState_default()
            self.rhand = HandState_default()
            self.left_pub = ChannelPublisher("rt/dex3/left/state", HandState_)
            self.right_pub = ChannelPublisher("rt/dex3/right/state", HandState_)
            self.left_pub.Init()
            self.right_pub.Init()
        # Lock and buffers exist before the callback is registered.
        self.sub = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self.sub.Init(self.on_cmd, 10)
        # SONIC publishes the two 7-DOF Dex3 commands separately from the
        # 29-body LowCmd. Keep the callback lightweight and consume snapshots
        # from the Isaac loop, just like body commands.
        self.hand_lock = threading.Lock()
        self.hand_q = {True: np.zeros(7, dtype=np.float64), False: np.zeros(7, dtype=np.float64)}
        self.hand_kp = {True: np.zeros(7, dtype=np.float64), False: np.zeros(7, dtype=np.float64)}
        self.hand_kd = {True: np.zeros(7, dtype=np.float64), False: np.zeros(7, dtype=np.float64)}
        self.hand_valid = {True: False, False: False}
        self.hand_command_count = {True: 0, False: 0}
        self.hand_received_monotonic = {True: 0., False: 0.}
        self.hand_received_unix_ns = {True: -1, False: -1}
        self.left_hand_sub = ChannelSubscriber("rt/dex3/left/cmd", HandCmd_)
        self.right_hand_sub = ChannelSubscriber("rt/dex3/right/cmd", HandCmd_)
        self.left_hand_sub.Init(lambda msg: self.on_hand_cmd(True, msg), 10)
        self.right_hand_sub.Init(lambda msg: self.on_hand_cmd(False, msg), 10)

    def on_cmd(self, msg):
        raw_q = np.asarray([m.q for m in msg.motor_cmd[:29]], dtype=np.float64)
        with self.lock:
            if raw_q.shape == (29,):
                self.last_raw_q = raw_q
                self.outside_physical_limit_count += int(np.any(
                    (raw_q < self.position_low) | (raw_q > self.position_high)))
            self.last_raw_valid = False
        try:
            command = decode_lowcmd(msg, self.position_low, self.position_high,
                                    self.velocity_limits, self.position_policy)
            with self.lock:
                self.cmd = command
                self.cmd_when = time.monotonic()
                self.command_count += 1
                self.last_raw_valid = True
        except CommandRejected as exc:
            reason = str(exc)
            with self.lock:
                self.reject_count += 1
                self.last_reject = reason
                if "target" in reason and "outside" in reason:
                    self.reject_categories["joint_position_limit"] += 1
                elif "velocity target" in reason:
                    self.reject_categories["joint_velocity_limit"] += 1
                elif "gain" in reason:
                    self.reject_categories["gain_range"] += 1
                elif "mode" in reason:
                    self.reject_categories["motor_mode"] += 1
                else:
                    self.reject_categories[reason.split(":", 1)[0]] += 1
                match = re.search(r"LowCmd motor (\d+)", reason)
                if match:
                    self.reject_motors[match.group(1)] += 1
            if self.reject_count <= 3 or self.reject_count % 100 == 0:
                print(f"[DDS] rejected LowCmd #{self.reject_count}: {exc}", flush=True)
        except Exception as exc:
            print(f"[DDS] LowCmd callback error: {exc}")

    def latest(self):
        with self.lock:
            age = (time.monotonic() - self.cmd_when) if self.cmd_when else math.inf
            return self.cmd, age, self.command_count, self.reject_count, self.last_reject

    def on_hand_cmd(self, is_left, msg):
        try:
            motors = list(msg.motor_cmd)
            if len(motors) != 7:
                return
            q = np.asarray([float(m.q) for m in motors], dtype=np.float64)
            kp = np.asarray([float(m.kp) for m in motors], dtype=np.float64)
            kd = np.asarray([float(m.kd) for m in motors], dtype=np.float64)
            if not (np.isfinite(q).all() and np.isfinite(kp).all() and np.isfinite(kd).all()):
                return
            with self.hand_lock:
                self.hand_q[is_left] = q
                self.hand_kp[is_left] = kp
                self.hand_kd[is_left] = kd
                self.hand_valid[is_left] = True
                self.hand_command_count[is_left] += 1
                self.hand_received_monotonic[is_left] = time.monotonic()
                self.hand_received_unix_ns[is_left] = time.time_ns()
        except Exception as exc:
            print(f"[DDS] hand command callback error: {exc}", flush=True)

    def latest_hand(self, is_left, with_clock=False):
        with self.hand_lock:
            snapshot = (self.hand_q[is_left].copy(), self.hand_kp[is_left].copy(),
                    self.hand_kd[is_left].copy(), self.hand_valid[is_left],
                    self.hand_command_count[is_left])
            return (snapshot + (self.hand_received_monotonic[is_left],
                                self.hand_received_unix_ns[is_left])) if with_clock else snapshot

    def rejection_summary(self):
        with self.lock:
            return {
                "categories": dict(self.reject_categories),
                "motors": dict(self.reject_motors),
                "outside_physical_limit_count": self.outside_physical_limit_count,
            }

    def raw_snapshot(self):
        with self.lock:
            return self.last_raw_q.copy(), self.last_raw_valid

    def close(self):
        self.sub.Close()
        self.left_hand_sub.Close()
        self.right_hand_sub.Close()
        publishers = [self.low_pub, self.odo_pub, self.imu_pub]
        if self.hands:
            publishers.extend((self.left_pub, self.right_pub))
        for publisher in publishers:
            publisher.Close()

    def reset_episode_buffers(self):
        """Clear the previous policy's targets before a paused scene reset."""
        with self.lock:
            self.cmd = None
            self.cmd_when = 0.
            self.command_count = self.reject_count = self.outside_physical_limit_count = 0
            self.last_reject = ''
            self.last_raw_q = np.full(29, np.nan)
            self.last_raw_valid = False
            self.reject_categories.clear()
            self.reject_motors.clear()
        with self.hand_lock:
            for side in (True, False):
                self.hand_q[side] = np.zeros(7)
                self.hand_valid[side] = False
                self.hand_command_count[side] = 0
                self.hand_received_monotonic[side] = 0.
                self.hand_received_unix_ns[side] = -1

    def publish(self, t, q, dq, ddq, torque, root_pos, root_quat,
                root_lin_w, root_ang_w, root_ang_b, root_acc_b,
                torso_quat, torso_gyro_b, torso_acc_b,
                hand_states=None):
        # q/dq/... motor order (not IsaacLab joint order).
        for idx in range(29):
            motor = self.low_state.motor_state[idx]
            motor.q = float(q[idx])
            motor.dq = float(dq[idx])
            motor.ddq = float(ddq[idx])
            # Estimated applied command, not a physical measured motor torque.
            motor.tau_est = float(torque[idx])
        self.low_state.tick = int(t * 1000) % (2**32)
        self.low_state.imu_state.quaternion[:] = [float(x) for x in root_quat]
        self.low_state.imu_state.gyroscope[:] = [float(x) for x in root_ang_b]
        self.low_state.imu_state.accelerometer[:] = [float(x) for x in root_acc_b]

        self.odo.tick = int(t * 1000) % (2**32)
        self.odo.position[:] = [float(x) for x in root_pos]
        self.odo.orientation[:] = [float(x) for x in root_quat]
        self.odo.linear_velocity[:] = [float(x) for x in root_lin_w]
        self.odo.angular_velocity[:] = [float(x) for x in root_ang_w]
        self.imu.quaternion[:] = [float(x) for x in torso_quat]
        self.imu.gyroscope[:] = [float(x) for x in torso_gyro_b]
        self.imu.accelerometer[:] = [float(x) for x in torso_acc_b]
        self.low_pub.Write(self.low_state)
        self.odo_pub.Write(self.odo)
        self.imu_pub.Write(self.imu)
        if self.hands:
            # Publish the simulated Revo2 positions back to SONIC. This keeps
            # the policy's hand feedback path coherent with the visual hand.
            for is_left, state in ((True, self.lhand), (False, self.rhand)):
                hand_q = None if hand_states is None else hand_states.get(is_left)
                if hand_q is not None:
                    hand_dq = hand_q[1]
                    for idx in range(min(7, len(hand_q[0]))):
                        state.motor_state[idx].q = float(hand_q[0][idx])
                        state.motor_state[idx].dq = float(hand_dq[idx])
                        state.motor_state[idx].ddq = 0.0
                        state.motor_state[idx].tau_est = 0.0
            self.left_pub.Write(self.lhand)
            self.right_pub.Write(self.rhand)


def main():
    print("[INFO] local DDS ONLY: domain 0 / interface lo")
    print("[INFO] SDK:", unitree_sdk2py.__file__)
    if not str(Path(unitree_sdk2py.__file__).resolve()).startswith(str(SDK.resolve())):
        raise RuntimeError("Wrong Unitree SDK imported; stop rather than risk incompatible DDS types")

    native_full = args.physics_profile == "upstream_full"
    scene_mode = args.scene is not None
    if native_full and (not scene_mode or not args.dexterous_hands or args.fixed_base):
        raise ValueError("Full upstream model requires the canonical scene, Revo2 hands and free base")
    full_model_audit = None
    authored_world_velocities = {}
    native_world_initialization = None
    scene_profile = None
    scene_root_pos = None
    scene_root_quat = None
    scene_path = args.scene.resolve()
    print(f"[INFO] opening authored scene: {scene_path}", flush=True)
    if not omni.usd.get_context().open_stage(str(scene_path)):
        raise RuntimeError(f"Could not open authored scene: {scene_path}")
    # Allow all referenced USD layers and PhysX schemas to resolve before
    # Isaac Lab adds the controlled proxy to the scene.
    for _ in range(30):
        simulation_app.update()
    stage = omni.usd.get_context().get_stage()
    # Keep preview lights, disabled authored robot prims and proxy materials
    # in the session layer; the downloaded living-room asset stays pristine.
    stage.SetEditTarget(stage.GetSessionLayer())
    scene_profile = scene_profile_for_stage(stage)
    scene_root_prim = stage.GetPrimAtPath(args.scene_robot_root)
    if not scene_root_prim or not scene_root_prim.IsValid():
        raise RuntimeError(f"Missing scene robot root: {args.scene_robot_root}")
    scene_root_matrix = UsdGeom.Xformable(scene_root_prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default())
    scene_root_translation = scene_root_matrix.ExtractTranslation()
    scene_root_rotation = scene_root_matrix.ExtractRotationQuat()
    scene_root_pos = tuple(float(v) for v in scene_root_translation)
    scene_root_quat = (float(scene_root_rotation.GetReal()),
                       *[float(v) for v in scene_root_rotation.GetImaginary()])
    pass

    from upstream_physics import PHYSX_SETTINGS, apply_scene_physics, check_scene_physics
    sim_cfg = sim_utils.SimulationCfg(
        dt=SIM_DT,
        device=args.device,
        physx=sim_utils.PhysxCfg(**PHYSX_SETTINGS),
        physics_prim_path=(scene_profile["physics_scene_path"]),
        render=authored_render_config(),
    )
    sim = SimulationContext(sim_cfg)
    scene_physics_audit = apply_scene_physics(stage, scene_profile['physics_scene_path'])
    upstream_renderer = None
    if upstream_sampling:
        from upstream_camera import apply_render_settings
        upstream_renderer = apply_render_settings(camera_profile['capture_contract'])
    scene_render = None
    ego_camera_info = None
    ego_capture = None
    scene_render = setup_scene_view(
        omni.usd.get_context().get_stage(), scene_profile,
        activate_viewport=not args.headless)
    pass

    source_model = model_contract()
    task_state = None
    if scene_mode and source_model and stage.GetRootLayer().customLayerData.get("layout_id") == "kinetra_task1_initialization":
        task_scene, task_state = task_initialization()
    from upstream_full_robot import create_native_robot, restore_authored_task_world
    authored_world_velocities, native_world_initialization = restore_authored_task_world(stage, scene_profile["environment_root"])
    robot, cfg, native_hand_info, appearance = create_native_robot(
        stage, args.scene_robot_root, scene_root_pos, scene_root_quat, task_state, source_model)
    print(f"[PHYSICS] complete upstream USD: {appearance['asset']}; native body and two hand articulations", flush=True)

    hand_articulations = {}
    hand_joint_ids = {}
    hand_full_joint_ids = {}
    hand_info = {}
    startup_alignment = None
    hand_info = native_hand_info
    hand_articulations = {True: robot, False: robot}
    # PhysX initialization can run a bootstrap step before the nominal robot
    # posture is written. Preserve authored task objects so that this bootstrap
    # cannot become an unrecorded collision impulse in the first episode.
    from pxr import Gf, UsdPhysics
    authored_world_poses = {}
    cache = UsdGeom.XformCache(0)
    for prim in Usd.PrimRange(stage.GetPrimAtPath(scene_profile['environment_root'])):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            transform = Gf.Transform(cache.GetLocalToWorldTransform(prim))
            q_world = transform.GetRotation().GetQuat()
            authored_world_poses[str(prim.GetPath())] = [*transform.GetTranslation(),
                                                       *q_world.GetImaginary(), q_world.GetReal()]
    sim.reset()
    scene_physics_audit = check_scene_physics(stage, scene_profile['physics_scene_path'])
    if hand_articulations:
        robot.update(0.0)
        for is_left in hand_articulations:
            side = "left" if is_left else "right"
            names = list(robot.joint_names)
            active_names = hand_info[side]["active_joint_names"]
            full_names = active_names + hand_info[side]["mimic_joint_names"]
            if len(names) != 51 or not set(full_names).issubset(names):
                raise RuntimeError(f"Invalid combined body/hand articulation: {names}")
            hand_joint_ids[is_left] = [names.index(name) for name in active_names]
            hand_full_joint_ids[is_left] = [names.index(name) for name in full_names]
            robot.write_joint_state_to_sim(
                torch.as_tensor(hand_info[side]['initial_joint_positions_rad'],
                                dtype=torch.float32, device=robot.device)[None, :].clamp(min=0.),
                torch.zeros((1, 11), device=robot.device),
                joint_ids=hand_full_joint_ids[is_left])
            print(f"[HANDS] {side}: 11 joints, active={hand_joint_ids[is_left]}", flush=True)
    robot.update(0.0)
    proxy_root_state = robot.data.default_root_state.clone()
    proxy_root_state[:, 7:] = 0.0
    proxy_joint_pos = robot.data.default_joint_pos.clone()
    for is_left, indices in hand_full_joint_ids.items():
        side = 'left' if is_left else 'right'
        proxy_joint_pos[:, indices] = torch.as_tensor(
            hand_info[side]['initial_joint_positions_rad'], device=robot.device,
            dtype=proxy_joint_pos.dtype).clamp(min=0.)
    proxy_joint_vel = torch.zeros_like(robot.data.joint_vel)
    robot.write_root_link_state_to_sim(proxy_root_state)
    robot.write_joint_state_to_sim(proxy_joint_pos, proxy_joint_vel)
    sim.forward()
    nominal_check = assert_nominal_body(
        robot.joint_names, as_np(robot.data.joint_pos[0]), motor_names,
        (task_state['observation.state'][0] if task_state is not None else policy_parameters()['default_angles']))
    def initial_body_poses():
        positions = as_np(robot.data.body_link_pos_w[0])
        rotations = as_np(robot.data.body_link_quat_w[0])
        return {name: pose_matrix(p, q) for name, p, q in
                zip(robot.body_names, positions, rotations)}
    stage = omni.usd.get_context().get_stage()
    feet = foot_floor_alignment(stage, scene_profile, cfg.prim_path, initial_body_poses(), legacy_bounds=False)
    if task_state is not None:
        feet['diagnostic_suggested_shift_m'] = feet['vertical_shift_m']
        feet['vertical_shift_m'] = 0.
        for foot in feet['feet']:
            foot['gap_after_m'] = foot['gap_before_m']
    proxy_root_state[:, 2] += feet['vertical_shift_m']
    robot.write_root_link_state_to_sim(proxy_root_state)
    sim.forward()
    roots = [cfg.prim_path] + [hand_info[side]['prim'] for side in ('left','right') if side in hand_info]
    joint_check = audit_joint_frames(stage, roots, initial_body_poses())
    startup_alignment = {**nominal_check, 'floor_contact': feet, 'joint_frames': joint_check,
                         'initial_root_position_m': as_np(proxy_root_state[0, :3]).tolist(),
                         'upstream_commit': CONFIG_COMMIT if task_state is not None else None,
                         'initial_body_joint_names': motor_names,
                         'initial_body_joint_positions_rad': as_np(proxy_joint_pos[0])[[robot.joint_names.index(n) for n in motor_names]].tolist(),
                         'method': ('exact upstream root_T and named initial_proprio; zero velocities; no root shift' if task_state is not None else
                                    'one-time nominal joint state and collision-sole floor alignment before DDS; no physics steps')}
    print(f"[STARTUP ALIGN] nominal error={nominal_check['max_body_nominal_error_rad']:.3g} rad, "
          f"floor dz={feet['vertical_shift_m']:.5f} m, joints={joint_check['joint_count']}, "
          f"max pivot gap={joint_check['max_pivot_gap_m']:.3g} m", flush=True)
    robot.reset()
    # Clear body drives for explicit PD; native hand drives retain upstream values.
    drive_body = robot.parts[0]
    drive_body.write_joint_stiffness_to_sim(0.0)
    drive_body.write_joint_damping_to_sim(0.0)
    pass

    full_model_audit = robot.runtime_audit(stage)
    print("[PHYSICS] native mass/inertia readback verified for all three articulations", flush=True)

    isaac_names = list(robot.joint_names)
    if not set(motor_names).issubset(set(isaac_names)) or len(isaac_names) < 29:
        raise RuntimeError("SONIC proxy articulation is missing required joints: " + repr(isaac_names))
    motor_to_isaac = np.asarray([isaac_names.index(name) for name in motor_names], dtype=int)
    body_isaac_indices = torch.as_tensor(motor_to_isaac, device=robot.device, dtype=torch.long)
    print("\nMotor index -> IsaacLab joint index")
    for motor_idx, (isaac_idx, name) in enumerate(zip(motor_to_isaac, motor_names)):
        print(f"  {motor_idx:02d} -> {isaac_idx:02d}  {name}")
    extra_names = [name for name in isaac_names if name not in set(motor_names)]
    print(f"[INFO] SONIC interface exposes {len(isaac_names)} joints; "
          f"SONIC drives 29 and maps {len(extra_names)} additional finger joints through the Revo2 controller: {extra_names}",
          flush=True)
    torso_matches = [i for i, n in enumerate(robot.body_names) if n == "torso_link"]
    if len(torso_matches) != 1:
        raise RuntimeError("Expected exactly one torso_link body; got " + repr(robot.body_names))
    torso_idx = torso_matches[0]
    torso_data = robot.parts[0].data
    if args.ego_camera:
        stage = omni.usd.get_context().get_stage()
        torso_prim_path = f"{cfg.prim_path}/torso_link"
        ego_camera_info = setup_ego_camera(
            stage, torso_prim_path, width=args.ego_camera_width,
            height=args.ego_camera_height, mount_position_m=args.ego_camera_mount,
            pitch_down_deg=args.ego_camera_pitch_down_deg,
            focal_length_mm=args.ego_camera_focal_length_mm, intrinsics=camera_profile['intrinsics'],
            extrinsics=camera_profile['extrinsics'],
            clipping_range_m=camera_profile['capture_contract']['render']['clipping_range_m'])
        ego_camera_info.update(capture_contract=camera_profile['capture_contract'],
            runtime_renderer=upstream_renderer,
            native_fps=args.ego_camera_fps,
            sampling=('pre_action; source_state_index = floor(control_index / 2) * 2' if upstream_sampling else args.ego_render_schedule),
            profile=camera_profile['profile'],
            profile_config={k: camera_profile[k] for k in ('config_path', 'config_sha256', 'override_fields')})
        print(f"[EGO] {ego_camera_info['profile']}: mount={ego_camera_info['mount_position_m']} m, "
              f"pitch_down={ego_camera_info['pitch_down_deg']} deg, focal={ego_camera_info['focal_length_mm']} mm",
              flush=True)
        if camera_profile['intrinsics']:
            print(f"[EGO] rectified K={camera_profile['intrinsics']['K_px']}; raw distortion disabled; robot extrinsics unverified", flush=True)
        if camera_profile['extrinsics']:
            print(f"[EGO] rigid torso mount: {camera_profile['extrinsics']['profile']}; {camera_profile['extrinsics']['status']}",flush=True)
        update_ego_camera_transform(stage, ego_camera_info)
        # Flush the session-layer camera prim into Hydra before creating the
        # Replicator product; without one Kit update a newly-authored camera
        # can silently fall back to the previous viewport camera.
        simulation_app.update()
        # The headset has its own offscreen render product. Keep the desktop
        # overview available for observing the complete robot and scene.
        ego_capture = EgoCameraCapture(
            ego_camera_info["camera_path"], width=args.ego_camera_width,
            height=args.ego_camera_height, fps=args.ego_camera_fps,
            record_path=args.ego_camera_record,
            zmq_port=args.ego_camera_zmq_port,
            rtp_host=args.ego_camera_rtp_host,
            rtp_port=args.ego_camera_rtp_port,
            camera_info=ego_camera_info,
        )
        for _ in range(camera_profile['capture_contract']['render']['warmup_render_calls']):
            sim.render()
        if upstream_sampling:
            ego_camera_info['runtime_projection'] = ego_capture.verify_projection()
            print(f"[EGO] RTX output K verified; max error={ego_camera_info['runtime_projection']['max_K_error_px']:.6g} px", flush=True)
        print(f"[EGO] attached to {torso_prim_path}; view={ego_camera_info['camera_path']} "
              f"{args.ego_camera_width}x{args.ego_camera_height}@{args.ego_camera_fps}", flush=True)

    # Read effort limits from the 29 simulated joints. YAML motor_effort_limit_list
    # contains extra 14 hand entries interspersed; do NOT slice first 29!
    effort_limits = as_np(robot.data.joint_effort_limits[0])[motor_to_isaac]
    if source_model:
        actual_armature = as_np(robot.data.joint_armature[0])[motor_to_isaac]
        if (not np.allclose(effort_limits, source_model['g1_29dof']['effort_limit_nm'], atol=1e-6)
                or not np.allclose(actual_armature, source_model['g1_29dof']['joint_armature_kg_m2'], atol=1e-8)):
            raise RuntimeError("PhysX effort limits/armature differ from pinned upstream model")
    if args.local_token_replay:
        replay_meta = json.loads(args.local_token_replay.with_suffix('.json').read_text())
        expected_effort = replay_meta.get('physics_contract', {}).get('effort_limit_nm')
        if expected_effort is not None and not np.allclose(effort_limits, expected_effort, atol=1e-6):
            raise RuntimeError("Replay effort limits differ from the actual capture")
    if np.any(~np.isfinite(effort_limits)) or np.any(effort_limits <= 0):
        raise RuntimeError(f"Bad IsaacLab effort limits: {effort_limits}")
    # Position limits come from the actual loaded articulation, already indexed
    # by the verified SONIC joint-name map. The upstream YAML labels its index
    # table yaw-first while its limit arrays are pitch-first; reordering those
    # arrays by that table swaps hip pitch/yaw limits. Runtime USD limits are
    # therefore the safety authority for both the stock URDF and authored USD.
    runtime_position_limits = as_np(robot.data.joint_pos_limits[0])[motor_to_isaac]
    pos_low = np.asarray(runtime_position_limits[:, 0], dtype=np.float64)
    pos_high = np.asarray(runtime_position_limits[:, 1], dtype=np.float64)
    vel_limits = np.asarray(reorder_yaml_values(config["motor_vel_limit_list"], config), dtype=np.float64)
    hold_kp = np.asarray(reorder_yaml_values(config["MOTOR_KP"], config), dtype=np.float64) * args.hold_scale
    hold_kd = np.asarray(reorder_yaml_values(config["MOTOR_KD"], config), dtype=np.float64) * args.hold_scale
    robot.update(0.0)
    default_joint_pos = as_np(robot.data.default_joint_pos[0])[motor_to_isaac]
    startup_gate = (PolicyStartupGate(args.wait_for_policy_log, default_joint_pos,
                    require_measured_history=os.environ.get('SONIC_SIM_STARTUP_REFERENCE') == '1')
                    if args.wait_for_policy_log else None)
    startup_waiting = startup_gate is not None
    startup_release_wall_time = None
    if startup_waiting:
        print("[STARTUP] Free-base articulation; waiting without physics steps for SONIC policy takeover.", flush=True)

    replay_world_view = None
    replay_contact_view = None
    if args.local_token_replay:
        from sonic_physics_replay import LocalTokenReplay
        dds=LocalTokenReplay(args.local_token_replay,Path(args.record).parent)
        if args.wait_for_policy_log or args.mode!='drive' or args.fixed_base:
            raise ValueError('local token replay requires free-base drive without policy startup gate')
        root=torch.as_tensor(np.r_[dds.a['initial_root_pos'],dds.a['initial_root_quat_wxyz'],
            dds.a['initial_root_lin_vel_w'],dds.a['initial_root_ang_vel_w']],device=robot.device,dtype=torch.float32)[None]
        robot.write_root_link_state_to_sim(root)
        robot.write_joint_state_to_sim(torch.as_tensor(dds.a['initial_joint_pos'],device=robot.device,dtype=robot.data.joint_pos.dtype)[None],
            torch.as_tensor(dds.a['initial_joint_vel'],device=robot.device,dtype=robot.data.joint_vel.dtype)[None],joint_ids=body_isaac_indices)
        for left,indices in hand_full_joint_ids.items():
            side='left' if left else 'right'
            robot.write_joint_state_to_sim(torch.as_tensor(dds.a['initial_'+side+'_hand_joint_pos'],device=robot.device,dtype=robot.data.joint_pos.dtype)[None],
                torch.as_tensor(dds.a['initial_'+side+'_hand_joint_vel'],device=robot.device,dtype=robot.data.joint_vel.dtype)[None],joint_ids=indices)
        sim.forward();robot.update(0.)
        if dds.info.get('align_initial_feet'):
            replay_feet=foot_floor_alignment(stage,scene_profile,cfg.prim_path,initial_body_poses(), legacy_bounds=False)
            root[:,2]+=replay_feet['vertical_shift_m']
            robot.write_root_link_state_to_sim(root)
            sim.forward();robot.update(0.)
            atomic_json(Path(args.record).parent/'replay_initialization.json',{
                'floor_contact':replay_feet,'root_state':as_np(root[0]).tolist(),
                'assumptions':dds.info['initialization_assumptions'],
                'exact_original_world_reconstruction':False})
        print('[LOCAL REPLAY] g1_dyn fresh states at 50Hz; PhysX 200Hz; DDS DISABLED',flush=True)
    else:
        ChannelFactoryInitialize(0, "lo")
        dds = DdsBridge(pos_low, pos_high, vel_limits, dummy_hands=(not args.no_hand_state),
                        position_policy=args.target_policy)
    # Preserve the sampling phase explicitly for both new and historical readers.
    world_recorder = Recorder() if args.record and scene_mode and not args.local_token_replay else None
    world_path = Path(args.record).with_name('world_trace.npz') if args.record else None
    world_metadata = {'schema_version': 1, 'sampling_phase': 'pre_action' if upstream_sampling else 'post_action',
                      'clock': ('pre-action simulation seconds and same-host monotonic ns' if upstream_sampling else
                                                               'post-step simulation seconds and same-host monotonic ns'),
                      'source': 'actual PhysX world state; no task-success inference', 'paths': 'world_paths.json'}
    if args.local_token_replay or world_recorder is not None:
        import omni.physics.tensors as physics_tensors
        import carb.settings
        # IsaacLab disables CPU contact reporting by default. This changes
        # reporting only, not gravity, collision geometry or contact solving.
        carb.settings.get_settings().set_bool('/physics/disableContactProcessing',False)
        replay_sim_view=physics_tensors.create_simulation_view('numpy')
        replay_sim_view.set_subspace_roots('/')
        stage=omni.usd.get_context().get_stage()
        from pxr import UsdPhysics
        replay_paths=[str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath(scene_profile['environment_root']))
                      if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        replay_world_view=replay_sim_view.create_rigid_body_view(replay_paths) if replay_paths else None
        world_dynamic_indices = np.asarray([i for i, path in enumerate(replay_world_view.prim_paths)
            if not UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(path)).GetKinematicEnabledAttr().Get()],
            dtype=np.uint32) if replay_world_view else np.empty(0, dtype=np.uint32)
        if replay_world_view is not None and authored_world_poses and not args.local_token_replay:
            targets = np.asarray([authored_world_poses[path] for path in replay_world_view.prim_paths], dtype=np.float32)
            from scene_support import restore_world_transforms
            initial_velocities = (np.asarray([authored_world_velocities[path] for path in replay_world_view.prim_paths], dtype=np.float32))
            restore_world_transforms(replay_world_view, targets, world_dynamic_indices, initial_velocities)
            sim.forward()
            error = float(np.max(np.abs(replay_world_view.get_transforms()-targets)))
            if error > 1e-5:
                raise RuntimeError('Authored task world restoration failed: ' + str(error))
            startup_alignment['world_restoration'] = {'body_count': len(targets), 'max_transform_error': error,
                'dynamic_velocity_reset_count': len(world_dynamic_indices),
                'method': 'restore authored object poses and zero velocities after PhysX bootstrap; no recorded physics step'}
            print(f'[STARTUP ALIGN] restored {len(targets)} authored task bodies after PhysX bootstrap; error={error:.3g}', flush=True)
        replay_contact_view=replay_sim_view.create_rigid_contact_view(
            [cfg.prim_path+'/'+side+'_ankle_roll_link' for side in ('left','right')])
        world_paths_metadata = {'body_paths':list(replay_world_view.prim_paths) if replay_world_view else [],
            'robot_body_names':list(robot.body_names),
            'foot_contact_paths':list(replay_contact_view.sensor_paths) if hasattr(replay_contact_view,'sensor_paths') else []}
        atomic_json(Path(args.record).parent/'world_paths.json', world_paths_metadata)
        print(f'[WORLD] Recording {len(replay_paths)} scene bodies, robot links and foot contact forces.',flush=True)
    if args.batch_control_file:
        if not scene_mode or not args.record or args.local_token_replay or args.fixed_base:
            raise ValueError('batch requires an authored scene, recording, and free-base live DDS')
        # Capture the actual aligned initial state once, before any physics.
        batch_initial_root = torch.cat((robot.data.root_link_pos_w, robot.data.root_link_quat_w,
                                      robot.data.root_link_lin_vel_w, robot.data.root_link_ang_vel_w), dim=-1).clone()
        batch_initial_joints = robot.data.joint_pos.clone()
        batch_initial_world = replay_world_view.get_transforms().copy() if replay_world_view else None
        batch_initial_world_velocities = replay_world_view.get_velocities().copy() if replay_world_view else None
    batch_paused = False
    recorder = Recorder()
    print(f"[INFO] started: mode={args.mode}, fixed_base={args.fixed_base}, "
          f"duration={args.duration}, hand_state={'measured' if hand_articulations else 'dummy' if not args.no_hand_state else 'disabled'}")
    if args.mode == "listen":
        print("[INFO] listen mode is observation-only; received control will NOT actuate robot.")
    else:
        print(f"[INFO] drive mode source={'local g1_dyn (DDS disabled)' if args.local_token_replay else 'local rt/lowcmd'}; PD backend={args.pd_backend}.")
        print(f"[INFO] gains: kp-scale={args.kp_scale}, kd-scale={args.kd_scale}; target-policy={args.target_policy}; extra-waist-damping={args.extra_waist_damping}")
        if args.kp_scale < .5 or args.kd_scale < .5:
            print("[WARN] Feedback below 50% can let arms/torso sag before the policy starts.")
    print('[INFO] No DDS publishers/subscribers opened.' if args.local_token_replay else
          '[INFO] All communication stays on 127.0.0.1/lo. Never use this with a physical robot.')

    torque_motor = np.zeros(29, dtype=np.float64)
    prev_lin_w = None
    step = 0
    cycle = 0
    t0 = time.monotonic()
    last_heartbeat = t0
    mode_text = "hold"
    sim_time = 0.0
    armed = False
    fault_reason = ""
    fault_time = None
    fault_hold_target = default_joint_pos.copy()
    last_seen_command_count = 0
    last_accepted_target = None
    applied_pd_target = default_joint_pos.copy()
    target_slew_steps = 0
    last_drive_gains = None
    last_drive_limits = None
    last_checkpoint = t0
    performance = LoopPerformance(SIM_DT, args.profile_timing)
    pacer = PhysicsPacer(SIM_DT)
    performance_ready = False
    posture_readiness = StartupPostureReadiness()
    stop_requested = threading.Event()
    simulation_stop = SimulationStopGate(args.stop_request_file) if args.stop_request_file else None
    stop_reason = ""
    recording_finalized = False
    recording_error = ""
    # Kit installs its own SIGINT handler. Override it after startup so Ctrl+C
    # leaves the control loop normally and our final recording is not skipped.
    def request_stop(signum, frame):
        nonlocal stop_reason
        stop_reason = f"signal {signal.Signals(signum).name}; ending physics"
        stop_requested.set()
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    def latch_fault(reason, current_q):
        nonlocal armed, fault_reason, fault_hold_target, fault_time
        if not fault_reason:
            fault_reason = reason
            fault_time = {"sim_seconds": sim_time, "wall_seconds": time.monotonic() - t0}
            fault_hold_target = current_q.copy()
            response = "ending free-base physics"
            print(f"[SAFETY] LATCHED: {reason}; {response}. Restart Bridge to clear.",
                  flush=True)
        armed = False

    def record_metadata():
        _, command_age, command_count, reject_count, last_reject = dds.latest()
        _, _, _, left_hand_valid, left_hand_count = dds.latest_hand(True)
        _, _, _, right_hand_valid, right_hand_count = dds.latest_hand(False)
        rejection_summary = dds.rejection_summary()
        rejected_by_named_motor = {
            f"{index}:{motor_names[int(index)]}": count
            for index, count in rejection_summary["motors"].items()
        }
        return {
            "robot": ("Unitree G1 SONIC 29 DOF stable URDF proxy" +
                      (" with mounted Revo2 hands" if hand_articulations else "") +
                      (f" in authored {scene_profile['label']} scene")),
            "simulator": "Isaac Sim 5.0 / Isaac Lab 2.3.2",
            "cpu_runtime": CPU_RUNTIME,
            "appearance": appearance,
            "local_token_replay": str(args.local_token_replay) if args.local_token_replay else None,
            "startup_alignment": startup_alignment,
            "scene_render": scene_render,
            "ego_camera": ego_camera_info,
            "ego_camera_record": args.ego_camera_record or None,
            "ego_camera_zmq_port": args.ego_camera_zmq_port or None,
            "ego_camera_rtp": ({"host": args.ego_camera_rtp_host,
                                "port": args.ego_camera_rtp_port}
                               if args.ego_camera_rtp_port else None),
            "ego_camera_frames": ego_capture.frame_count if ego_capture is not None else 0,
            "scene": str(args.scene.resolve()),
            "scene_profile": scene_profile,
            "scene_robot_root": args.scene_robot_root,
            "scene_robot_mode": "native_upstream_usd",
            "scene_proxy_prim_path": None,
            "authored_scene_robot_disabled": False,
            "scene_articulation_joint_count": len(isaac_names),
            "render_every_physics_steps": args.render_every,
            "sim_dt": SIM_DT, "record_every": args.record_every,
            "started_monotonic": t0,
            "motor_joint_names": motor_names, "isaac_joint_names": isaac_names,
            "motor_to_isaac_joint_index": motor_to_isaac.tolist(),
            "position_limits_motor_order": {"low": pos_low.tolist(), "high": pos_high.tolist()},
            "position_limit_source": "loaded Isaac articulation joint limits, mapped by joint name",
            "velocity_limits_motor_order": vel_limits.tolist(),
            "effort_limits_motor_order": effort_limits.tolist(),
            "fall_guard_min_height_m": 0.38,
            "mode": args.mode, "fixed_base": args.fixed_base,
            "startup_policy_log": str(args.wait_for_policy_log) if args.wait_for_policy_log else None,
            "startup_physics_withheld": startup_gate is not None,
            "startup_release_wall_seconds": startup_release_wall_time,
            "simulation_startup_reference": os.environ.get('SONIC_SIM_STARTUP_REFERENCE') == '1',
            "startup_takeover_error_rad": startup_gate.release_error_rad if startup_gate else None,
            "max_base_tilt_deg": args.max_base_tilt_deg,
            "ground_material": {"source": "authored upstream object materials", "scene_default_override": False},
            "target_policy": args.target_policy,
            "pd_backend": args.pd_backend,
            "physics_contract": {
                "profile": args.physics_profile, "commit": CONFIG_COMMIT if source_model else None,
                "model": "configs/control/sonic/g1_revo2_model.json" if source_model else str(CFG),
                "pd_backend": args.pd_backend, "simulation_dt_s": SIM_DT, "control_frequency_hz": 50,
                "joint_order": motor_names, "effort_limit_nm": effort_limits.tolist(),
                "armature_kg_m2": as_np(robot.data.joint_armature[0])[motor_to_isaac].tolist(),
                "full_model_equivalence": bool(full_model_audit and full_model_audit['passed']),
                "equivalence_scope": "native robot physical asset and initialization; excludes global simulation settings and dynamic behavior",
                "checked_scene_configuration_equality": scene_physics_audit['passed'],
                "scene_physics_audit": scene_physics_audit,
                "equivalence_reference": "pinned upstream USD plus canonical initialization builder",
                "native_runtime_audit": full_model_audit,
                "world_initialization": native_world_initialization,
                "remaining_differences": [],
            },
            "device": str(robot.device),
            "extra_waist_damping": args.extra_waist_damping,
            "pd_target_slew": args.pd_target_slew,
            "render_every": args.render_every,
            "kp_scale": args.kp_scale, "kd_scale": args.kd_scale,
            "torque_scale": args.torque_scale, "hold_scale": args.hold_scale,
            "control_formula": "torque_scale*tau_ff + kp_scale*kp*position_error + kd_scale*kd*velocity_error",
            "feedback_location": "external explicit torque",
            "torque_recording_semantics": "clipped external PD effort command",
            "real_time_factor": sim_time / max(time.monotonic() - t0, 1e-9),
            "physics_runtime": performance.summary(),
            "physics_pacing": {"max_catchup_s": pacer.max_catchup_s, "rebases": pacer.rebases,
                               "discarded_startup_or_stall_debt_s": pacer.discarded_debt_s},
            "ego_render_schedule": args.ego_render_schedule,
            "checkpoint_mode": "background compression and atomic writes",
            "max_initial_error": args.max_initial_error,
            "max_command_step": args.max_command_step,
            "target_slew_steps": target_slew_steps,
            "watchdog": args.watchdog, "fault_reason": fault_reason,
            "fault_time": fault_time,
            "stop_reason": stop_reason,
            "recording_finalized": recording_finalized,
            "recording_error": recording_error,
            "recorded_samples": len(recorder.samples),
            "world_trace_required": replay_contact_view is not None,
            "world_trace": str(world_path) if replay_contact_view is not None else None,
            "stop_request_file": str(args.stop_request_file) if args.stop_request_file else None,
            "dds_lowcmd_count": command_count, "dds_rejected_count": reject_count,
            "dds_last_rejection": last_reject,
            "dds_rejections_by_category": rejection_summary["categories"],
            "dds_rejections_by_motor": rejected_by_named_motor,
            "dds_pd_targets_outside_physical_limits": rejection_summary["outside_physical_limit_count"],
            "dds_command_age_seconds": command_age if np.isfinite(command_age) else None,
            "dds_domain": 0, "dds_interface": "lo",
            "imu_acceleration": "approximate finite-difference specific force",
            "joint_torque": "commanded and clipped torque, not a physical torque sensor",
            "hands": ("native upstream Revo2; independent articulations, six driven joints and five native mimic joints per hand"),
            "hand_setup": hand_info,
            "hand_joint_names": {side: hand_info[side]["active_joint_names"] + hand_info[side]["mimic_joint_names"]
                                 for side in ("left", "right")}
                                if hand_articulations else None,
            "hand_position_limits": {("left" if left else "right"): {
                "low": as_np(hand.data.joint_pos_limits[0])[hand_full_joint_ids[left], 0].tolist(),
                "high": as_np(hand.data.joint_pos_limits[0])[hand_full_joint_ids[left], 1].tolist()}
                for left, hand in hand_articulations.items()},
            "dds_hand_command_counts": {"left": left_hand_count, "right": right_hand_count},
            "dds_hand_command_valid": {"left": left_hand_valid, "right": right_hand_valid},
        }

    def finish_episode_recording():
        nonlocal recording_finalized, recording_error
        if recording_finalized:
            return
        failures = [recording_error] if recording_error else []
        if ego_capture is not None:
            try:
                ego_capture.finish_recording()
            except BaseException as exc:
                failures.append(f'ego video: {type(exc).__name__}: {exc}')
        if world_recorder is not None:
            try:
                world_recorder.save(world_path, {**world_metadata, 'recording_finalized': not failures,
                                                'samples': len(world_recorder.samples),
                                                'recording_error': '; '.join(failures)})
            except BaseException as exc:
                failures.append(f'world trace: {type(exc).__name__}: {exc}')
        recording_finalized = not failures
        recording_error = '; '.join(failures)
        try:
            recorder.save(args.record, record_metadata())
            if args.record and not recorder.samples:
                # Failed startup still retains the physics audit and reason;
                # no empty/fabricated trajectory is presented as a recording.
                atomic_json(Path(args.record).with_suffix('.json'), record_metadata())
        except BaseException as exc:
            failures.append(f'Bridge: {type(exc).__name__}: {exc}')
            recording_finalized = False
            recording_error = '; '.join(failures)
            # If the first Bridge write got far enough to create an archive,
            # give it one last attempt with an explicit incomplete marker.
            try:
                recorder.save(args.record, record_metadata())
            except BaseException:
                pass
        recording_finalized = not failures
        recording_error = '; '.join(failures)
        if failures:
            raise RuntimeError('recording finalization incomplete: ' + recording_error)
        if args.record:
            atomic_json(Path(args.record).with_name('bridge_saved.json'), {
                'recording_finalized': recording_finalized, 'samples': len(recorder.samples),
                'camera_frames': ego_capture.frame_count if ego_capture else 0,
                'simulation_seconds': sim_time, 'host_monotonic_s': time.monotonic(),
                'stop_reason': stop_reason, 'fault_reason': fault_reason,
                'recording_error': recording_error})

    try:
        while simulation_app.is_running() and not stop_requested.is_set():
            if batch_paused:
                from batch_runtime import read_prepare, acknowledge_reset
                data = read_prepare(args.batch_control_file, Path(args.record).parent)
                if data is not None:
                    next_run = Path(data['run'])
                    # The coordinator waits for the previous Manager and SONIC
                    # to exit before issuing this token. Physics is still paused.
                    dds.reset_episode_buffers()
                    robot.write_root_link_state_to_sim(batch_initial_root.clone())
                    robot.write_joint_state_to_sim(batch_initial_joints.clone(), torch.zeros_like(batch_initial_joints))
                    robot.reset()
                    robot.set_joint_position_target(batch_initial_joints.clone())
                    robot.set_joint_velocity_target(torch.zeros_like(batch_initial_joints))
                    robot.set_joint_effort_target(torch.zeros_like(batch_initial_joints))
                    robot.write_data_to_sim()
                    if replay_world_view is not None:
                        from scene_support import restore_world_transforms
                        restore_world_transforms(replay_world_view, batch_initial_world, world_dynamic_indices, batch_initial_world_velocities)
                    sim.forward()
                    robot.update(0.)
                    root_error = float(np.max(np.abs(as_np(robot.data.root_link_pos_w) - as_np(batch_initial_root[:, :3]))))
                    joint_error = float(np.max(np.abs(as_np(robot.data.joint_pos) - as_np(batch_initial_joints))))
                    world_error = (float(np.max(np.abs(replay_world_view.get_transforms() - batch_initial_world)))
                                   if replay_world_view is not None else 0.)
                    velocity_error = max(float(np.max(np.abs(as_np(robot.data.joint_vel)))),
                                         float(np.max(np.abs(as_np(robot.data.root_link_lin_vel_w)))),
                                         float(np.max(np.abs(as_np(robot.data.root_link_ang_vel_w)))),
                                         float(np.max(np.abs(replay_world_view.get_velocities() - batch_initial_world_velocities))) if replay_world_view else 0.)
                    if max(root_error, joint_error, world_error, velocity_error) > 1e-4:
                        raise RuntimeError(f'batch reset verification failed: root={root_error}, joints={joint_error}, world={world_error}, velocities={velocity_error}')
                    args.record = str(next_run / 'bridge.npz')
                    args.ego_camera_record = str(next_run / 'ego_view.mp4')
                    args.wait_for_policy_log = next_run / 'sonic.log'
                    args.stop_request_file = next_run / 'simulation_stop.json'
                    recorder = Recorder()
                    world_recorder = Recorder() if replay_contact_view is not None else None
                    world_path = next_run / 'world_trace.npz'
                    if world_recorder is not None:
                        atomic_json(next_run / 'world_paths.json', world_paths_metadata)
                    if ego_capture is not None:
                        ego_capture.start_recording(args.ego_camera_record)
                    startup_gate = PolicyStartupGate(args.wait_for_policy_log, default_joint_pos,
                        require_measured_history=os.environ.get('SONIC_SIM_STARTUP_REFERENCE') == '1')
                    startup_waiting = True
                    startup_release_wall_time = None
                    simulation_stop = SimulationStopGate(args.stop_request_file)
                    recording_finalized = False
                    recording_error = stop_reason = fault_reason = ''
                    fault_time = None
                    prev_lin_w = None
                    torque_motor = np.zeros(29)
                    cycle = step = 0
                    sim_time = 0.
                    t0 = last_checkpoint = last_heartbeat = time.monotonic()
                    performance = LoopPerformance(SIM_DT, args.profile_timing)
                    pacer = PhysicsPacer(SIM_DT)
                    performance_ready = False
                    posture_readiness = StartupPostureReadiness()
                    armed = False
                    last_seen_command_count = 0
                    last_accepted_target = None
                    applied_pd_target = default_joint_pos.copy()
                    target_slew_steps = 0
                    last_drive_gains = last_drive_limits = None
                    batch_paused = False
                    acknowledge_reset(args.batch_control_file, data, {
                        'root_max_error_m': root_error, 'joint_max_error_rad': joint_error,
                        'scene_transform_max_error': world_error, 'scene_body_count': len(batch_initial_world)
                        if batch_initial_world is not None else 0,
                        'velocity_max': velocity_error})
                    print(f'[BATCH] reset verified; next episode {next_run.name}; no physics steps during reset.', flush=True)
            if not batch_paused and simulation_stop is not None and simulation_stop.requested():
                stop_reason = "local Manager requested simulation end before policy shutdown"
                print("[STOP] Manager handoff received; ending physics before policy shutdown.", flush=True)
                if args.batch_control_file:
                    simulation_stop.acknowledge(sim_time)
                    finish_episode_recording()
                    batch_paused = True
                    atomic_json(Path(args.record).parent / 'bridge_segment_closed.json', {
                        'recording_finalized': True, 'bridge_pid': os.getpid(), 'simulation_seconds': sim_time,
                        'physics_paused': True, 'runtime_retained': True})
                    print('[BATCH] episode files closed; physics paused; Isaac and camera stream retained.', flush=True)
                else:
                    break
            if args.duration and sim_time >= args.duration:
                break
            performance.begin()
            begin = time.monotonic()
            state_monotonic_ns = time.monotonic_ns()
            state_unix_ns = time.time_ns()
            # Physical observations in ISAAC joint order -> DDS motor order.
            joint_positions = as_np(robot.data.joint_pos[0])
            joint_velocities = as_np(robot.data.joint_vel[0])
            q = joint_positions[motor_to_isaac]
            dq = joint_velocities[motor_to_isaac]
            ddq = as_np(robot.data.joint_acc[0])[motor_to_isaac]
            root_pos = as_np(robot.data.root_link_pos_w[0])
            root_quat = as_np(robot.data.root_link_quat_w[0])
            root_lin_w = as_np(robot.data.root_link_lin_vel_w[0])
            root_ang_w = as_np(robot.data.root_link_ang_vel_w[0])
            base_tilt = base_tilt_radians(root_quat)
            posture_ready = posture_readiness.update(sim_time, base_tilt, root_lin_w, root_ang_w)
            # IMU gyroscope is in the robot frame, not world coordinates.
            root_inverse = inverse_quaternion_matrix(root_quat)
            root_ang_b = root_inverse @ root_ang_w
            torso_quat = as_np(torso_data.body_link_quat_w[0, torso_idx])
            torso_pos = as_np(torso_data.body_link_pos_w[0, torso_idx])
            torso_inverse = inverse_quaternion_matrix(torso_quat)
            torso_gyro_b = torso_inverse @ as_np(torso_data.body_link_ang_vel_w[0, torso_idx])
            # Approximate specific force (not a simulated IMU sensor).
            if prev_lin_w is None:
                root_acc_w = np.array([0.0, 0.0, 9.81])
            else:
                root_acc_w = (root_lin_w - prev_lin_w) / SIM_DT + np.array([0.0, 0.0, 9.81])
            prev_lin_w = root_lin_w.copy()
            root_acc_b = root_inverse @ root_acc_w
            # SONIC consumes secondary_imu accelerometer in the torso frame.
            # Publishing the default zero vector removes gravity/specific-force
            # information from the policy observation and can destabilize pose
            # tracking even when the joint mapping is correct.
            torso_acc_b = torso_inverse @ root_acc_w
            hand_states = {}
            hand_record = {}
            physical_sample_step = step
            record_this_cycle = args.record and ((step if upstream_sampling else cycle + 1) % args.record_every == 0)
            if hand_articulations:
                for is_left, hand in hand_articulations.items():
                    hand_states[is_left] = (
                        revo2_to_dex3(joint_positions[hand_joint_ids[is_left]], is_left),
                        revo2_to_dex3(joint_velocities[hand_joint_ids[is_left]], is_left, velocity=True))
                    if not record_this_cycle:
                        continue
                    side = "left" if is_left else "right"
                    wire_q, _, _, wire_valid, wire_count, wire_mono, wire_unix = dds.latest_hand(is_left, with_clock=True)
                    hand_record[f"{side}_hand_joint_pos"] = joint_positions[hand_full_joint_ids[is_left]].astype(np.float32)
                    hand_record[f"{side}_hand_joint_vel"] = joint_velocities[hand_full_joint_ids[is_left]].astype(np.float32)
                    hand_record[f"{side}_hand_dds_target"] = wire_q.astype(np.float32)
                    hand_record[f"{side}_hand_command_valid"] = np.int8(wire_valid)
                    hand_record[f"{side}_hand_command_count"] = np.int64(wire_count)
                    hand_record[f"{side}_hand_command_monotonic_s"] = np.float64(wire_mono)
                    hand_record[f"{side}_hand_command_unix_ns"] = np.int64(wire_unix)
            dds.publish(sim_time, q, dq, ddq, torque_motor, root_pos, root_quat,
                        root_lin_w, root_ang_w, root_ang_b, root_acc_b,
                        torso_quat, torso_gyro_b, torso_acc_b, hand_states)

            performance.mark("state_and_publish")
            if batch_paused:
                # Keep genuine frozen-state feedback and headset video alive;
                # do not actuate old commands or append to closed recordings.
                if ego_capture is not None and ego_capture.capture_due():
                    update_ego_camera_transform(omni.usd.get_context().get_stage(), ego_camera_info,
                                                world_pos=torso_pos, world_quat_wxyz=torso_quat)
                    sim.render()
                    ego_capture.maybe_capture(sim_time)
                time.sleep(.005)
                continue

            cmd, age, nmsg, nreject, last_reject = dds.latest()
            if nmsg != last_seen_command_count and cmd is not None:
                last_seen_command_count = nmsg
                if args.mode == "drive" and not fault_reason:
                    if not armed:
                        initial_error = first_command_error(cmd, q)
                        if initial_error > args.max_initial_error:
                            latch_fault(
                                f"first LowCmd target/current error {initial_error:.3f} rad "
                                f"> {args.max_initial_error:.3f}; align PICO/Sim pose before A+X",
                                q,
                            )
                        else:
                            armed = True
                            last_accepted_target = cmd.q.copy()
                            print(f"[SAFETY] armed on aligned first LowCmd (max error {initial_error:.3f} rad)",
                                  flush=True)
                    elif last_accepted_target is not None:
                        target_step = float(np.max(np.abs(cmd.q - last_accepted_target)))
                        if target_step > args.max_command_step and args.target_policy == "strict":
                            latch_fault(
                                f"LowCmd target jump {target_step:.3f} rad > "
                                f"{args.max_command_step:.3f}", q)
                        else:
                            last_accepted_target = cmd.q.copy()
            if (args.mode == "drive" and armed and not fault_reason
                    and cmd is not None and age > args.watchdog):
                latch_fault(f"LowCmd watchdog age {age:.3f}s > {args.watchdog:.3f}s", q)
            if args.mode == "drive" and not args.fixed_base and fault_reason:
                # A local joint hold does not balance an unsupported pelvis.
                # On loss of policy control, end physics before another step;
                # do not turn a communications failure into a recorded fall.
                stop_reason = "free-base safety fault ended physics: " + fault_reason
                break

            use_external = bool(args.mode == "drive" and armed and not fault_reason
                                and cmd is not None and age <= args.watchdog)
            if startup_waiting:
                try:
                    takeover_ready = startup_gate.ready(cmd, age, args.watchdog,
                        current_q=q, max_initial_error=args.max_initial_error)
                except ValueError as exc:
                    latch_fault(str(exc), q)
                    break
                if takeover_ready and use_external:
                    startup_waiting = False
                    startup_release_wall_time = time.monotonic()-t0
                    print(f"[STARTUP] Released to genuine free-base physics at wall={startup_release_wall_time:.3f}s; no further base support.", flush=True)
                elif time.monotonic()-t0 > args.startup_timeout:
                    latch_fault("free-base startup timeout: no fresh SONIC policy takeover", q)
                    break
            if upstream_sampling and startup_waiting:
                record_this_cycle = False
            # Apply a local pose hold whenever the controller is silent or safety is latched.
            q_target = fault_hold_target if fault_reason else default_joint_pos
            dq_target = np.zeros(29, dtype=np.float64)
            drive_kp = np.zeros(29)
            drive_kd = np.zeros(29)
            feedforward = np.zeros(29)
            if use_external:
                control_cmd = cmd
                if args.target_policy == "sonic-pd" and args.pd_target_slew == "joint-speed":
                    control_cmd = slew_pd_command(cmd, applied_pd_target, vel_limits,
                                                  SIM_DT, args.max_command_step)
                    target_slew_steps += int(np.any(np.abs(control_cmd.q - cmd.q) > 1e-6))
                applied_pd_target = control_cmd.q.copy()
                q_target = control_cmd.q.copy()
                dq_target = cmd.dq.copy()
                torque_motor = compute_torque(
                    control_cmd, q, dq, args.kp_scale, args.kd_scale, args.torque_scale,
                    effort_limits,
                )
                drive_kp = np.clip(control_cmd.kp * args.kp_scale, 0, 450.)
                drive_kd = np.clip(control_cmd.kd * args.kd_scale, 0, 25.)
                feedforward = np.clip(args.torque_scale * control_cmd.tau, -effort_limits, effort_limits)
                # Explicitly scoped to fixed-base tuning; do not change the
                # policy target or echo it as measured state.
                drive_kd[12:15] += args.extra_waist_damping
                torque_motor[12:15] -= args.extra_waist_damping * dq[12:15]
                torque_motor = np.clip(torque_motor, -effort_limits, effort_limits)
                mode_text = "SONIC"
            else:
                if args.mode == "listen":
                    # Listen mode is observation-only.  Holding the pose here
                    # silently actuated the articulation despite the CLI text.
                    torque_motor = np.zeros(29, dtype=np.float64)
                    mode_text = "listen"
                else:
                    torque_motor = hold_torque(q_target, q, dq, hold_kp, hold_kd, effort_limits)
                    drive_kp, drive_kd = hold_kp.copy(), hold_kd.copy()
                    mode_text = ("fault-hold" if fault_reason else
                                 ("hold" if cmd is None else "timeout-hold"))
            if not np.isfinite(torque_motor).all():
                raise RuntimeError("Non-finite computed torque; bridge stopped")
            torque_motor = guard_joint_effort(torque_motor, q, dq, pos_low, pos_high, vel_limits)
            pass
            if hand_articulations and args.mode == "drive":
                for is_left, hand in hand_articulations.items():
                    # Record the exact command snapshot used below, including
                    # clocks/counter under the same callback lock.
                    hand_q, _, _, hand_valid, count, received, unix_ns = dds.latest_hand(is_left, with_clock=True)
                    if not hand_valid:
                        hand_q = controller_hand_command(0., 0., is_left)
                    indices = hand_full_joint_ids[is_left]
                    limits = as_np(hand.data.joint_pos_limits[0])[indices]
                    active_target = dex3_to_revo2(hand_q, is_left)
                    measured_active = joint_positions[hand_joint_ids[is_left]]
                    hand_target = np.clip(revo2_full_target(active_target, measured_active), limits[:, 0], limits[:, 1])
                    if record_this_cycle:
                        side = "left" if is_left else "right"
                        hand_record[f"{side}_hand_dds_target"] = hand_q.astype(np.float32)
                        hand_record[f"{side}_hand_command_valid"] = np.int8(hand_valid)
                        hand_record[f"{side}_hand_command_count"] = np.int64(count)
                        hand_record[f"{side}_hand_command_monotonic_s"] = np.float64(received)
                        hand_record[f"{side}_hand_command_unix_ns"] = np.int64(unix_ns)
                        hand_record[f"{side}_hand_target_joint_pos"] = hand_target.astype(np.float32)
                    indices = hand_joint_ids[is_left]
                    command_target = hand_target[:6]
                    hand.set_joint_position_target(
                        torch.as_tensor(command_target, device=hand.device, dtype=torch.float32)[None, :],
                        joint_ids=indices)
                    hand.set_joint_velocity_target(
                        torch.zeros((1, len(indices)), device=hand.device, dtype=torch.float32), joint_ids=indices)
            performance.mark("control_prepare")
            if upstream_sampling:
                from upstream_camera import native_frame_due
                # Cameras share the 200 Hz physical state grid. During READY/startup
                # pauses use wall deadlines to keep the headset image responsive.
                due = native_frame_due(step) if not startup_waiting else ego_capture.capture_due()
                if due:
                    update_ego_camera_transform(omni.usd.get_context().get_stage(), ego_camera_info,
                                                world_pos=torso_pos, world_quat_wxyz=torso_quat)
                    performance.mark("camera_pose")
                    sim.render()
                    performance.mark("rtx_frame")
                    captured = ego_capture.maybe_capture(sim_time, force=True,
                                              sample_monotonic_ns=state_monotonic_ns,
                                              sample_unix_ns=state_unix_ns,
                                              source_physics_step=step if not startup_waiting else -1)
                    performance.mark("camera_output")
                    if not captured and not startup_waiting:
                        raise RuntimeError('Upstream native camera frame missing at physics step ' + str(step))
                if replay_contact_view is not None and not startup_waiting and record_this_cycle:
                    world_sample = {'time': sim_time, 'host_monotonic_ns': np.int64(state_monotonic_ns),
                        'object_transform_xyzw': replay_world_view.get_transforms().copy() if replay_world_view else np.empty((0,7),dtype=np.float32),
                        'foot_net_contact_force_w': replay_contact_view.get_net_contact_forces(SIM_DT).copy(),
                        'body_link_pos_w': as_np(robot.data.body_link_pos_w[0]).copy(),
                        'body_link_quat_wxyz': as_np(robot.data.body_link_quat_w[0]).copy()}
                    if args.local_token_replay:
                        dds.world.append(world_sample)
                    else:
                        world_recorder.add(**world_sample)
                performance.mark("world_snapshot")
            torques_isaac = np.zeros(len(isaac_names), dtype=np.float32)
            torques_isaac[motor_to_isaac] = (torque_motor).astype(np.float32)
            robot.set_joint_effort_target(torch.as_tensor(torques_isaac, device=robot.device)[None, :])
            robot.write_data_to_sim()
            action_monotonic_ns = time.monotonic_ns()
            sample_time = sim_time
            performance.mark("control_write")
            advancing = not startup_waiting
            if advancing:
                pacer.begin(begin)
            if upstream_sampling:
                should_render = False  # already rendered the pre-action state
            elif ego_capture is not None and args.ego_render_schedule == "wall":
                should_render = ego_capture.capture_due()
            else:
                render_step = step if advancing else cycle
                should_render = (not args.headless or ego_capture is not None) and render_step % args.render_every == 0
            if advancing:
                sim.step(render=should_render and ego_capture is None)
                robot.update(SIM_DT)
                sim_time += SIM_DT
                step += 1
            performance.mark("physics")
            if should_render and (ego_capture is not None or not advancing):
                if ego_capture is not None:
                    # Update USD only for an actual render, using post-step pose.
                    update_ego_camera_transform(
                        omni.usd.get_context().get_stage(), ego_camera_info,
                        world_pos=as_np(torso_data.body_link_pos_w[0, torso_idx]),
                        world_quat_wxyz=as_np(torso_data.body_link_quat_w[0, torso_idx]))
                render_start = time.monotonic() if args.profile_timing else 0.
                sim.render()
                if args.profile_timing:
                    performance.pending.append(('rtx_frame', time.monotonic() - render_start))
                if ego_capture is not None:
                    camera_start = time.monotonic() if args.profile_timing else 0.
                    ego_capture.maybe_capture(sim_time)
                    if args.profile_timing:
                        performance.pending.append(('camera_output', time.monotonic() - camera_start))
            performance.mark("render_and_camera")
            cycle += 1
            if not upstream_sampling and replay_contact_view is not None and not startup_waiting and cycle % args.record_every == 0:
                world_sample = {'time':sim_time, 'host_monotonic_ns':np.int64(time.monotonic_ns()),
                                'object_transform_xyzw':replay_world_view.get_transforms().copy() if replay_world_view else np.empty((0,7),dtype=np.float32),
                                'foot_net_contact_force_w':replay_contact_view.get_net_contact_forces(SIM_DT).copy(),
                                'body_link_pos_w':as_np(robot.data.body_link_pos_w[0]).copy(),
                                'body_link_quat_wxyz':as_np(robot.data.body_link_quat_w[0]).copy()}
                if args.local_token_replay:
                    dds.world.append(world_sample)
                else:
                    world_recorder.add(**world_sample)
            if record_this_cycle:
                raw_q, raw_valid = dds.raw_snapshot()
                recorder.add(time=sample_time, joint_pos=q.astype(np.float32), joint_vel=dq.astype(np.float32),
                             state_monotonic_ns=np.int64(state_monotonic_ns), state_unix_ns=np.int64(state_unix_ns),
                             action_monotonic_ns=np.int64(action_monotonic_ns), control_cycle_index=np.int64(physical_sample_step if upstream_sampling else cycle),
                             root_acc_b=root_acc_b.astype(np.float32), root_ang_vel_b=root_ang_b.astype(np.float32),
                             joint_acc=ddq.astype(np.float32), applied_torque_cmd=torque_motor.astype(np.float32),
                             external_effort_cmd=(torque_motor).astype(np.float32),
                             drive_kp=drive_kp.astype(np.float32), drive_kd=drive_kd.astype(np.float32),
                             target_joint_pos=q_target.astype(np.float32), target_joint_vel=dq_target.astype(np.float32),
                             root_pos=root_pos.astype(np.float32), root_quat_wxyz=root_quat.astype(np.float32),
                             root_lin_vel_w=root_lin_w.astype(np.float32), root_ang_vel_w=root_ang_w.astype(np.float32),
                             controller_active=np.int8(use_external),
                             wall_time=np.float64(time.monotonic() - t0),
                             command_age=np.float64(age),
                             dds_accepted_count=np.int64(nmsg), dds_rejected_count=np.int64(nreject),
                             raw_target_joint_pos=raw_q.astype(np.float32),
                             raw_command_valid=np.int8(raw_valid), fault_latched=np.int8(bool(fault_reason)),
                             startup_waiting=np.int8(startup_waiting), base_tilt_rad=np.float32(base_tilt),
                             **hand_record)
            if (args.record and args.checkpoint_seconds > 0
                    and time.monotonic() - last_checkpoint >= args.checkpoint_seconds):
                recorder.checkpoint(args.record, record_metadata())
                if world_recorder is not None:
                    world_recorder.checkpoint(world_path, world_metadata)
                last_checkpoint = time.monotonic()
            if (args.mode == "drive" and not args.fixed_base and not startup_waiting
                    and (root_pos[2] < 0.38 or base_tilt > math.radians(args.max_base_tilt_deg))):
                latch_fault(f"free-base fall guard: z={root_pos[2]:.3f}m, tilt={math.degrees(base_tilt):.1f}deg", q)
                print("[STOP] Free-base fall guard; ending physics and saving observations.", flush=True)
                break
            performance.mark("record_and_checkpoint")
            performance.finish(advancing)
            if performance.steps == 1 and advancing:
                print('[PERF] WARMING: hold a neutral pose; wait for [PERF] READY before recording. This is a timing notice, not a save gate.', flush=True)
            if time.monotonic() - last_heartbeat >= 1.0:
                _, _, _, left_hand_valid, left_hand_count = dds.latest_hand(True)
                _, _, _, right_hand_valid, right_hand_count = dds.latest_hand(False)
                physics_hz = performance.summary().get('recent_physics_steps_per_wall_second', 0.)
                ready = performance.steps >= 1000 and 190. <= physics_hz <= 210. and posture_ready
                if ready and not performance_ready:
                    print(f'[PERF] READY: recent physics={physics_hz:.1f}/200 Hz; neutral posture steady for 2 simulation seconds; SONIC target=50 Hz. Left grip+A starts/stops recording.', flush=True)
                elif performance_ready and not ready:
                    print(f'[PERF] NOT_READY: physics={physics_hz:.1f}/200 Hz; tilt={math.degrees(base_tilt):.1f} deg; neutral_posture_ready={posture_ready}. Review timing and balance before recording.', flush=True)
                performance_ready = ready
                print(f"[BRIDGE] sim={sim_time:6.2f}s wall={time.monotonic()-t0:6.2f}s "
                      f"dds_lowcmd={nmsg} rejected={nreject} age={age:.2f}s state={mode_text} "
                      f"z={root_pos[2]:.3f}m max_tau={np.max(abs(torque_motor)):.1f}Nm "
                      f"hands={left_hand_count}/{right_hand_count} "
                      f"physics_hz={physics_hz:.1f}", flush=True)
                last_heartbeat = time.monotonic()
            if not args.no_pace:
                # Calibration/policy waits never create physics catch-up debt.
                remain = pacer.remaining(time.monotonic()) if advancing else SIM_DT
                if remain > 0:
                    time.sleep(remain)
    except KeyboardInterrupt:
        print("[STOP] Ctrl+C received")
    except BaseException as exc:
        recording_error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if not recording_finalized and simulation_stop is not None and simulation_stop.request is not None:
            # The control loop has ended: after this acknowledgement there are
            # no further sim.step calls, even while final recording is written.
            simulation_stop.acknowledge(sim_time)
        # Finish encoding independently of Replicator/Kit teardown. A physics
        # ACK alone never means that recording files have been finalized.
        try:
            finish_episode_recording()
        except BaseException as exc:
            recording_error = f"{type(exc).__name__}: {exc}"
            print(f"[STOP] Recording incomplete; raw diagnostics preserved: {recording_error}", flush=True)
        try:
            dds.close()
        finally:
            # A failed MP4/recorder write must never strand Kit or hide the
            # incomplete status. Keep every other cleanup in its own finally.
            signal.signal(signal.SIGINT, signal.default_int_handler)
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            try:
                print("[STOP] Closing DDS and camera", flush=True)
                if ego_capture is not None:
                    ego_capture.close()
            except BaseException as exc:
                if not recording_error:
                    recording_error = f"camera shutdown: {type(exc).__name__}: {exc}"
                print(f"[STOP] Camera shutdown reported an error: {exc}", flush=True)
            finally:
                print("[STOP] Closing Isaac Sim", flush=True)
                simulation_app.close(wait_for_replicator=False)


if __name__ == "__main__":
    main()
