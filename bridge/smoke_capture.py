from project_paths import SONIC_ROOT
#!/usr/bin/env python3
"""Short, local, free-base SONIC POSE capture with Revo2 and an actual ego camera.

Uses prerecorded human POSE and explicitly diagnostic hand inputs, never PICO
hardware or a physical-robot interface. Does not change either latest-session
pointer or the live video publisher. All three components use the production supervisor.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = SONIC_ROOT
DEPLOY = PROJECT / "gear_sonic_deploy"
DEFAULT_SOURCE = ROOT / "outputs/20261025_173708_pico_lSfOT5/pico_replay_archive.npz"


def driver(run, source, pose_seconds=5., port=15556, warmup=5., replay_hands=False, hand_cycle=False, manual_recording=False):
    sys.path.insert(0, str(PROJECT))
    from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_command_message, build_planner_message, pack_pose_message
    from gear_sonic.utils.teleop.packet_recorder import PacketRecorder
    from gear_sonic.utils.teleop.revo2_hand_control import controller_hand_command
    from gear_sonic.utils.teleop.simulation_stop import request_simulation_stop
    from gear_sonic.utils.teleop.capture_control import RecordingControl, RecordingChords
    from pose_replay import PoseReplay
    from capture_utils import atomic_json
    import zmq
    replay = PoseReplay(source)
    if replay_hands and (len(replay.hands) != 2 or len(replay.controllers) != 4):
        raise ValueError("recorded hand replay requires both hands and all four controller inputs")
    recorder = PacketRecorder(run / "pico_input")
    recording = RecordingControl(run/'recording_control.json') if manual_recording else None
    chords = RecordingChords()
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.LINGER, 1000)
    sock.bind(f"tcp://127.0.0.1:{port}")
    time.sleep(.4)
    start = time.monotonic()
    period = .02
    deadline = start
    sent = 0
    stopped = False
    try:
        while time.monotonic() - start < warmup + pose_seconds + 1.:
            now = time.monotonic()
            elapsed = now - start
            if elapsed < warmup:
                hands = {f"{side}_hand_position": controller_hand_command(0, 0, side == "left") for side in ("left", "right")}
                sock.send(build_planner_message(0, [0., 0., 0.], [1., 0., 0.], **hands))
                sock.send(build_command_message(start=True, stop=False, planner=True))
            else:
                ptime = elapsed - warmup
                a = False
                if recording:
                    # Exercise production gesture detection, including held
                    # buttons, rather than manufacture a closed gate.
                    a = 1. <= ptime < 1.3 or pose_seconds - .5 <= ptime < pose_seconds
                    event = chords.update(a, False, False, False, .8 if a else 0.)
                    if event and recording.handle(event, pose_ready=sent > 0) == 'stopped':
                        break
                payload = replay.payload(min(ptime, pose_seconds), now)
                if not replay_hands:
                    trigger = .45 if 1. <= ptime < min(3., pose_seconds - .5) else 0.
                    grip = 0.
                    if hand_cycle:
                        grip = .55 if 1. <= ptime < 3. or 7. <= ptime < 8.5 else 0.
                        trigger = .65 if 4. <= ptime < 6. or 7. <= ptime < 8.5 else 0.
                    for side in ("left", "right"):
                        # Include the real diagnostic recording chord's side
                        # grip in both raw input and the grasp mapping.
                        side_grip = max(grip, .8 if side == 'left' and a else 0.)
                        payload[f"{side}_hand_joints"] = controller_hand_command(trigger, side_grip, side == "left")
                        payload[f"{side}_trigger"] = np.array([trigger], dtype=np.float32)
                        payload[f"{side}_grip"] = np.array([side_grip], dtype=np.float32)
                sock.send(build_command_message(start=True, stop=False, planner=False))
                sock.send(pack_pose_message(payload))
                packet = dict(payload)
                packet["timestamp_send_monotonic"] = np.array([time.monotonic()])
                recorder.submit(sent, packet)
                sent += 1
            deadline += period
            time.sleep(max(0., deadline - time.monotonic()))
    except KeyboardInterrupt:
        # Match the real Manager's normal interrupt/finally shutdown semantics.
        print('[SMOKE] normal interrupted episode shutdown', flush=True)
    finally:
        if recording:
            recording.handle('finish', pose_ready=False, reason='smoke driver exit')
        # Exactly the production stop order: physics ACK, policy stop, drain.
        stopped = request_simulation_stop(run / "simulation_stop.json", "recorded POSE capture/replay complete", timeout=10.)
        if stopped:
            sock.send(build_command_message(start=False, stop=True, planner=False))
            time.sleep(.15)
        recorder.close()
        sock.close()
        ctx.term()
        atomic_json(run / "smoke_driver.json", {"source": str(source), "sent_packets": sent,
            "physics_stopped": stopped, "warmup_s": warmup, "pose_seconds": pose_seconds,
            "wall_duration_s": time.monotonic() - start,
            "replay_hands": replay_hands, "diagnostic_hand_cycle": hand_cycle,
            "input_kind": ("recorded human POSE and recorded hands/controllers" if replay_hands else
                           "recorded human POSE + synthetic main-trigger full-hand grasp and side-grip recording inputs; no new human capture")})
    if not stopped:
        raise RuntimeError("Bridge did not acknowledge stop; policy not disconnected")


def main():
    from runtime_cpu import configure_cpu_affinity
    configure_cpu_affinity()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", type=Path)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--pose-seconds", type=float, default=5.)
    parser.add_argument("--port", type=int, default=15556)
    parser.add_argument('--ego-zmq-port', type=int, default=0,
                        help='Optional isolated video publisher for end-to-end performance tests; 0 disables')
    parser.add_argument("--replay-hands", action="store_true", help="preserve recorded hand/controller inputs instead of diagnostic grip")
    parser.add_argument("--hand-cycle", action="store_true", help="diagnostic open/side-grip-only/open/main-trigger/open/combined/open sequence; use 10..12 seconds")
    parser.add_argument('--manual-recording', action='store_true', help='exercise production left-grip+A start/stop boundaries')
    parser.add_argument("--full-recording", action="store_true", help="explicitly replay the entire source; default stays a 2..12s smoke test")
    parser.add_argument("--scene-profile", choices=["supermarket"], default="supermarket")
    parser.add_argument('--task', help='diagnostic task instruction for testing propagation, never a success label')
    args = parser.parse_args()
    if not 0 <= args.ego_zmq_port <= 65535:
        parser.error('--ego-zmq-port must be 0..65535')
    if args.hand_cycle and (args.replay_hands or not 10 <= args.pose_seconds <= 12):
        parser.error('--hand-cycle requires synthetic hands and 10..12 seconds')
    if args.driver:
        driver(args.driver.resolve(), args.source.resolve(), args.pose_seconds, args.port, replay_hands=args.replay_hands, hand_cycle=args.hand_cycle, manual_recording=args.manual_recording)
        return
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            arguments = (proc / "cmdline").read_bytes().split(b"\0")
            executable = Path(os.fsdecode(arguments[0])).name
            if not (executable.startswith("python") or executable == "g1_deploy_onnx_ref"):
                continue
            names = {Path(os.fsdecode(argument)).name for argument in arguments if argument}
            if names & {"sonic_isaacsim_bridge.py", "g1_deploy_onnx_ref", "pico_manager_thread_server.py"}:
                raise RuntimeError(f"Existing control process PID {proc.name}; smoke test will not disturb it")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    if not args.source.is_file():
        parser.error("a recorded source file is required")
    if args.full_recording:
        from pose_replay import PoseReplay
        args.pose_seconds = PoseReplay(args.source).duration
    elif not 2 <= args.pose_seconds <= 12:
        parser.error("use a recorded source and 2..12 seconds of POSE")
    from capture_session import initialize, finalize
    from capture_manifest import capture
    from capture_utils import atomic_json
    run = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("%Y%m%d_%H%M%S") + "_capture_smoke_", dir=ROOT / "outputs"))
    scene_path = ROOT/"assets/scenes/pico_supermarket_task1_7e5fd9d.usda"
    initialize(run, profile=f"bridge-{args.scene_profile}", input_kind="recorded_pose_smoke", manual_recording=args.manual_recording, task=args.task)
    for component in ("bridge", "sonic", "manager"):
        capture(run, component, pose_file=args.source,
                parameters={"smoke": True, "source": str(args.source), "pose_seconds": args.pose_seconds,
                            "replay_hands": args.replay_hands, "diagnostic_hand_cycle": args.hand_cycle},
                scene_file=scene_path if component == "bridge" else None)
    atomic_json(run / "smoke_config.json", {"source": str(args.source), "pose_seconds": args.pose_seconds,
        "scene": str(scene_path), "scene_profile": args.scene_profile,
        "private_manager_port": args.port, "ego_zmq_port": args.ego_zmq_port, "diagnostic_hand_cycle": args.hand_cycle})
    print(f"OUTPUT {run}", flush=True)
    env = os.environ.copy()
    env.update(OMNI_KIT_ACCEPT_EULA="YES", PYTHONUNBUFFERED="1")
    env["LD_LIBRARY_PATH"] = ":".join([str(Path.home() / "TensorRT/lib"), "/usr/local/cuda-13.0/lib64", "/opt/onnxruntime/lib", env.get("LD_LIBRARY_PATH", "")])
    children = {}
    def launch(component, command, cwd):
        with (run / f"supervisor_{component}.log").open("xb") as log:
            p = subprocess.Popen([sys.executable, str(ROOT / "bridge/capture_session.py"), "run-component",
                "--run", str(run), "--component", component, "--", *command], cwd=cwd, env=env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        children[component] = p
        return p
    def wait_text(component, text, timeout=240):
        path = run / f"{component}.log"
        until = time.monotonic() + timeout
        while time.monotonic() < until:
            if path.exists() and text in path.read_text(errors="replace"):
                return
            if children[component].poll() is not None:
                raise RuntimeError(f"{component} exited before ready; see {run}")
            time.sleep(.3)
        raise TimeoutError(f"{component} startup timed out; see {run}")
    try:
        launch("bridge", [sys.executable, str(ROOT / "bridge/sonic_isaacsim_bridge.py"), "--headless", "--device", "cpu",
            "--scene", str(scene_path),
            "--dexterous-hands", "--ego-camera", "--ego-camera-height", "360",
            '--ego-camera-zmq-port', str(args.ego_zmq_port),
            "--ego-camera-record", str(run / "ego_view.mp4"), "--render-every", "8",
            "--mode", "drive", "--target-policy", "sonic-pd", "--pd-backend", "external", "--pd-target-slew", "none",
            "--wait-for-policy-log", str(run / "sonic.log"), "--startup-timeout", "240", "--duration", "0",
            "--stop-request-file", str(run / "simulation_stop.json"), "--record", str(run / "bridge.npz"),
            "--checkpoint-seconds", "5"], ROOT / ".runtime/IsaacLab-2.3.2-sparse")
        wait_text("bridge", "[INFO] started:")
        print("BRIDGE_READY (free base, Revo2, 640x360 camera)", flush=True)
        launch("sonic", [sys.executable, str(ROOT / "bridge/run_sonic_with_debug.py"),
            "--run", str(run), "--port", "5557", "--",
            str(DEPLOY / "target/release/g1_deploy_onnx_ref"), "lo", "policy/release/model_decoder.onnx", "reference/example/",
            "--obs-config", "policy/release/observation_config.yaml", "--encoder-file", "policy/release/model_encoder.onnx",
            "--planner-file", "planner/target_vel/V2/planner_sonic.onnx", "--disable-crc-check", "--input-type", "zmq_manager",
            "--zmq-port", str(args.port), "--output-type", "zmq", "--zmq-out-port", "5557",
            "--target-motion-logfile", str(run / "sonic_target.csv"),
            "--planner-motion-logfile", str(run / "sonic_planner.csv"), "--enable-csv-logs", "--logs-dir", str(run / "sonic_state")], DEPLOY)
        wait_text("sonic", "Init Done")
        print("SONIC_READY; running short POSE reference", flush=True)
        manager = launch("manager", [sys.executable, str(Path(__file__).resolve()), "--driver", str(run),
            "--source", str(args.source.resolve()), "--pose-seconds", str(args.pose_seconds), "--port", str(args.port),
            *(["--replay-hands"] if args.replay_hands else []),
            *(["--hand-cycle"] if args.hand_cycle else []),
            *(["--manual-recording"] if args.manual_recording else [])], ROOT)
        manager.wait(timeout=args.pose_seconds + 120)
        for component in ("sonic", "bridge"):
            children[component].wait(timeout=120)
        result = finalize(run)
        print(json.dumps({"run": str(run), "status": result["status"]}, indent=2), flush=True)
        if result["status"] != "complete":
            raise SystemExit(2)
    finally:
        # Never match/kill global names. Only children created above are owned.
        for component in ("manager", "bridge", "sonic"):
            child = children.get(component)
            if child is not None and child.poll() is None:
                os.kill(child.pid, signal.SIGINT)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    print(f"CLEANUP_PENDING {component} supervisor_pid={child.pid}; see {run}", flush=True)


if __name__ == "__main__":
    main()
