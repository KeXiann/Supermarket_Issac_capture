#!/usr/bin/env python3
"""Local PICO session lifecycle, durable completion, offline integrity checks.

The physical-stop ACK and bridge_saved.json deliberately have different meanings.
Only finalized, checked files constitute a complete capture. No legacy capture is
modified or retroactively certified by this tool.
"""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid

import numpy as np

from capture_utils import (HUMAN_REFERENCE_MAX_AGE_S, MAX_HUMAN_PACKET_GAP_S, human_packet_gap_summary,
                           atomic_json, human_reference_valid, sync_directory, sync_tree)

ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = ("bridge", "sonic", "manager")
FINALIZATION_JOURNAL = "finalization_status.json"


def capture_notice(message):
    # Closing the terminal must not turn an otherwise successful export into
    # a BrokenPipeError. The durable journal remains the progress authority.
    try:
        print(message, flush=True)
    except (OSError, ValueError):
        # Prevent Python's exit-time stdout flush from turning successful
        # export into exit 120 after the terminal's read end disappears.
        try:
            null_fd = os.open(os.devnull, os.O_WRONLY)
            try:
                os.dup2(null_fd, sys.stdout.fileno())
            finally:
                os.close(null_fd)
        except (AttributeError, OSError, ValueError):
            pass


class FinalizationProgress:
    """Durable phases and deferred terminal signals during closed-file export.

    Construct only while holding the session's finalizer lock. This operational
    journal is mutable and deliberately excluded from artifact checksums.
    SIGKILL/native crashes cannot be intercepted: a running journal with a dead
    PID then identifies the interrupted phase for recovery.
    """
    def __init__(self, run):
        self.run = Path(run)
        self.path = self.run / FINALIZATION_JOURNAL
        self.state = {"schema_version": 1, "attempt_id": uuid.uuid4().hex,
                      "status": "running", "stage": "starting", "pid": os.getpid(),
                      "started_utc": utc_now(), "deferred_signals": []}
        self.previous = {}

    def persist(self):
        self.state["updated_utc"] = utc_now()
        atomic_json(self.path, self.state)

    def stage(self, name):
        self.state["stage"] = name
        self.persist()
        capture_notice(f"[CAPTURE] exporting: {name} ({self.run.name})")

    def defer(self, signum, frame):
        name = signal.Signals(signum).name
        self.state["deferred_signals"].append({"signal": name, "utc": utc_now()})
        # A signal can interrupt print/JSON/fsync themselves. Keep this handler
        # free of reentrant I/O; the next phase checkpoint persists the signal.

    def __enter__(self):
        try:
            if threading.current_thread() is threading.main_thread():
                for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                    self.previous[sig] = signal.signal(sig, self.defer)
            self.persist()
        except BaseException:
            self.restore_handlers()
            raise
        return self

    def restore_handlers(self):
        for sig, handler in self.previous.items():
            signal.signal(sig, handler)

    def finish(self, result):
        mentor = result.get("mentor_submission", {})
        self.state.update(status=("complete" if result.get("status") == "complete"
                                 and mentor.get("status") != "incomplete" else "incomplete"),
                          last_stage=self.state["stage"], stage="finished", ended_utc=utc_now())
        failed = [check for check in result.get("checks", []) if not check.get("passed")]
        if failed:
            self.state["failed_checks"] = failed
        if mentor:
            self.state["mentor_submission"] = mentor
        self.persist()
        return result

    def __exit__(self, typ, error, tb):
        try:
            # Successful Mentor-only publication removes this temporary run
            # while the lock and signal guard are still held.
            if not self.run.exists():
                return False
            if error is not None:
                self.state.update(status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                                  error=f"{type(error).__name__}: {error}",
                                  traceback="".join(traceback.format_exception(typ, error, tb)), ended_utc=utc_now())
                self.persist()
            elif self.state["deferred_signals"]:
                # A signal can arrive in the small window after the final
                # checkpoint; persist it before restoring normal handlers.
                self.persist()
        finally:
            self.restore_handlers()
        return False


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def initialize(run, profile="bridge-supermarket", input_kind="live_pico", manual_recording=False,
               task=None, mentor_only_output=None, publication_root=None):
    run = Path(run).resolve()
    run.mkdir(parents=True, exist_ok=True)
    if (run / "session.json").exists() or any(run.iterdir()):
        raise FileExistsError(f"Session requires an empty, new directory: {run}")
    if task is not None and (not isinstance(task, str) or not task.strip()):
        raise ValueError("task must be a non-empty task instruction")
    mentor_only_output = Path(mentor_only_output).resolve() if mentor_only_output else None
    publication_root = Path(publication_root).resolve() if publication_root else None
    if mentor_only_output is not None:
        if profile not in ("bridge-livingroom", "bridge-supermarket") or input_kind != "live_pico":
            raise ValueError("Mentor-only publishing is restricted to live PICO scene captures")
        if publication_root is None or mentor_only_output.parent != publication_root:
            raise ValueError("Mentor-only output must be a direct child of the publication root")
        if run.parent != publication_root / ".capture_work":
            raise ValueError("Mentor-only capture work must be isolated under outputs/.capture_work/")
        if mentor_only_output == run.parent:
            raise ValueError("Mentor-only output must not replace the capture work root")
        if os.path.lexists(mentor_only_output):
            raise FileExistsError(f"refusing to overwrite Mentor output: {mentor_only_output}")
    data = {"schema_version": 1, "session_id": uuid.uuid4().hex,
            "created_utc": utc_now(), "host_monotonic_s": time.monotonic(),
            "host_boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "stage": "development", "formal_collection_approved": False,
            "profile": profile, "manager_profile": "manager", "control_mode": "SONIC_POSE",
            "input_kind": input_kind, "manual_recording": bool(manual_recording),
            "task": task.strip() if task is not None else "PICO SONIC POSE development capture",
            "task_source": "operator_before_capture" if task is not None else "development_default",
            "task_outcome": "unreviewed",
            "required_components": list(COMPONENTS),
            "require_hands": profile in ("bridge-livingroom", "bridge-supermarket"),
            "require_camera": profile in ("bridge-livingroom", "bridge-supermarket"),
            "require_sonic_debug": True,
            "publish_mentor_sample": profile in ("bridge-livingroom", "bridge-supermarket"),
            "clock": "same-host CLOCK_MONOTONIC; simulation clock separate; UTC for provenance only"}
    if mentor_only_output is not None:
        data["mentor_only_output_directory"] = str(mentor_only_output)
        data["publication_root"] = str(publication_root)
    atomic_json(run / "session.json", data)
    return data


def component_state(run, component):
    return Path(run) / "components" / f"{component}.json"


def preflight(run, component):
    """Bind additional launch terminals to this live session before any writes."""
    run = Path(run).resolve()
    session = read_json(run / "session.json")
    if (run / "completion.json").exists() or component_state(run, component).exists() or (run / f"capture_{component}.json").exists():
        raise FileExistsError(f"refusing to restart/overwrite {component} in {run}; start a new Bridge session")
    required = ("bridge", "sonic") if component == "manager" else ("bridge",)
    for dependency in required:
        state = read_json(component_state(run, dependency))
        if state.get("status") != "running" or state["session_id"] != session["session_id"]:
            raise RuntimeError(f"{dependency} is not running in this session")
        command = Path(f"/proc/{state['pid']}/cmdline").read_bytes().split(b"\0")
        wanted = "sonic_isaacsim_bridge.py" if dependency == "bridge" else "g1_deploy_onnx_ref"
        names = {Path(os.fsdecode(arg)).name for arg in command if arg}
        binding = (str(Path(session['batch_runtime_directory']) / 'control.json')
                   if dependency == 'bridge' and session.get('batch_bridge_segment') else str(run))
        if wanted not in names or not any(binding.encode() in arg for arg in command):
            raise RuntimeError(f"{dependency} PID no longer belongs to this session")
    if component == "manager" and "Init Done" not in (run / "sonic.log").read_text(errors="replace"):
        raise RuntimeError("wait for SONIC Init Done before starting Manager")
    return {"session_id": session["session_id"], "component": component, "ready": True}


def supervise(run, component, command, cleanup_timeout=20.):
    """Own exactly one child process group; never signal another user's run."""
    run = Path(run).resolve()
    session = read_json(run / "session.json")
    directory = run / "components"
    directory.mkdir(exist_ok=True)
    state_path = component_state(run, component)
    # Lifetime lock prevents racing terminals before the state is written.
    with (directory / f"{component}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if state_path.exists():
            raise FileExistsError(f"Component already started in this session: {state_path}")
        state = {"session_id": session["session_id"], "component": component,
                 "status": "starting", "started_utc": utc_now(),
                 "started_monotonic": time.monotonic(), "supervisor_pid": os.getpid(),
                 "command": command, "cwd": str(Path.cwd())}
        atomic_json(state_path, state)
        child = None
        cleanup = "normal"
        def forward(signum, frame):
            if child is not None and child.poll() is None:
                # The SDK manager uses KeyboardInterrupt/finally for shutdown.
                try:
                    os.killpg(child.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
        previous = {sig: signal.signal(sig, forward) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            with (run / f"{component}.log").open("xb") as log:
                child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         start_new_session=True, bufsize=0)
                state.update(status="running", pid=child.pid)
                atomic_json(state_path, state)
                def copy_output():
                    while True:
                        block = child.stdout.read(65536)
                        if not block:
                            break
                        log.write(block)
                        log.flush()
                        try:
                            sys.stdout.buffer.write(block)
                            sys.stdout.buffer.flush()
                        except (BrokenPipeError, OSError):
                            pass
                reader = threading.Thread(target=copy_output, daemon=True)
                reader.start()
                saved_at = None
                while child.poll() is None:
                    if component == "bridge" and (run / "bridge_saved.json").exists():
                        saved_at = saved_at or time.monotonic()
                        if time.monotonic() - saved_at > cleanup_timeout:
                            cleanup = "native_cleanup_timeout_after_durable_save"
                            os.killpg(child.pid, signal.SIGTERM)
                            try:
                                child.wait(timeout=3)
                            except subprocess.TimeoutExpired:
                                os.killpg(child.pid, signal.SIGKILL)
                            break
                    time.sleep(.1)
                code = child.wait()
                reader.join(timeout=5)
                if reader.is_alive():
                    raise RuntimeError("child log pipe did not close")
                log.flush()
                os.fsync(log.fileno())
            state.update(status="exited", exit_code=code, cleanup=cleanup,
                         ended_utc=utc_now(), ended_monotonic=time.monotonic())
            atomic_json(state_path, state)
        except BaseException as exc:
            state.update(status="failed", error=f"{type(exc).__name__}: {exc}", ended_utc=utc_now())
            atomic_json(state_path, state)
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGINT)
            raise
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    # Last component to finish builds the reviewable package automatically.
    if session.get('finalization_owner') == 'batch_coordinator':
        return 0 if cleanup != 'normal' else code
    package = finalize(run, if_ready=True)
    if (package["status"] == "incomplete"
            or package.get("mentor_submission", {}).get("status") == "incomplete"):
        return 2
    return 0 if cleanup != "normal" else code


def inspect(run, verify_hashes=True):
    """Fail closed on missing, truncated, dropped, mismatched or unfinalized data."""
    from prepare_trajectory import load_robot, load_human, mode_samples
    from capture_manifest import fingerprint
    run = Path(run).resolve()
    checks = []
    detail = {}
    def check(name, fn):
        try:
            value = fn()
            checks.append({"name": name, "passed": True, "detail": value})
            return value
        except Exception as exc:
            checks.append({"name": name, "passed": False,
                           "detail": f"{type(exc).__name__}: {exc}"})
    def require(condition, message):
        if not condition:
            raise ValueError(message)
        return message
    session = check("session_metadata", lambda: read_json(run / "session.json"))
    if session and session.get('manual_recording'):
        from recording_window import recording_window
        def explicit_window():
            control = read_json(run / 'recording_control.json')
            if control.get('status') == 'discarded':
                return {'status': 'discarded', 'raw_diagnostics_retained': True}
            return recording_window(run)
        check('explicit_operator_recording_window', explicit_window)
    def components():
        for name in COMPONENTS:
            state = read_json(component_state(run, name))
            expected_status = 'segment_closed' if name == 'bridge' and session.get('batch_bridge_segment') else 'exited'
            require(state["status"] == expected_status, f"{name} did not close cleanly ({expected_status})")
            require(state["session_id"] == session["session_id"], f"{name} session mismatch")
            if expected_status == 'segment_closed':
                segment = read_json(run / 'bridge_segment_closed.json')
                require(state.get('segment_close_code') == 0 and segment.get('physics_paused') is True
                        and segment.get('recording_finalized') is True and segment.get('bridge_pid') == state['pid'],
                        'persistent Bridge segment is not durably closed with physics paused')
            else:
                require(state["exit_code"] == 0 or (name == "bridge" and state.get("cleanup") ==
                        "native_cleanup_timeout_after_durable_save"), f"{name} exit code {state['exit_code']}")
        return "all component logs closed and exit states recorded"
    check("components_closed", components)
    def provenance():
        for name in COMPONENTS:
            info = read_json(run / f"capture_{name}.json")
            require(info["host_boot_id"] == session["host_boot_id"], "different host boot")
            for entry in info["files"]:
                if "snapshot_relative_to_run" in entry:
                    snap = run / entry["snapshot_relative_to_run"]
                    require(fingerprint(snap)["sha256"] == entry["sha256"], f"snapshot mismatch: {snap}")
        return "component host clocks and captured source snapshots match"
    check("provenance", provenance)
    loaded = check("robot_arrays", lambda: _robot_summary(load_robot(run), detail))
    host = None
    if loaded:
        meta, robot, indices, _ = detail["robot"]
        host = (robot["state_monotonic_ns"] / 1e9 if "state_monotonic_ns" in robot
                else meta["started_monotonic"] + robot["wall_time"])
        def bridge_end():
            saved = read_json(run / "bridge_saved.json")
            require(meta.get("recording_finalized") is True and saved["recording_finalized"] is True,
                    "Bridge contains only a checkpoint")
            require(saved["samples"] == meta["recorded_samples"] == len(indices) + detail["startup_samples"],
                    "final sample count mismatch")
            require(not saved.get("recording_error") and not meta.get("recording_error"), "recording error")
            require(not meta.get("fault_reason"), f"control fault: {meta.get('fault_reason')}")
            require(np.all(robot["controller_active"]) and not np.any(robot["fault_latched"]), "inactive physical samples")
            require(meta["dds_rejected_count"] == 0, "DDS rejected commands")
            expected = meta["sim_dt"] * meta["record_every"]
            require(np.all(np.abs(np.diff(robot["time"]) - expected) <= expected * .5), "robot recording gaps")
            require(saved["simulation_seconds"] - robot["time"][-1] <= expected + meta["sim_dt"] + 1e-6,
                    "missing final physical samples")
            request = read_json(run / "simulation_stop.json")
            ack = read_json(run / "simulation_stop.json.ack.json")
            require(ack["token"] == request["token"] and ack["physics_stopped"] is True, "stop ACK mismatch")
            require(abs(ack["simulation_seconds"] - saved["simulation_seconds"]) < 1e-8, "physics advanced after ACK")
            return {"samples": len(indices), "simulation_seconds": saved["simulation_seconds"]}
        check("bridge_final_save_and_stop", bridge_end)
        if meta.get('world_trace_required'):
            def physical_world():
                paths = read_json(run / 'world_paths.json')
                status = read_json(run / 'world_trace.json')
                require(status.get('recording_finalized') is True, 'world state has only a checkpoint')
                with np.load(run / 'world_trace.npz') as world:
                    n = len(world['time'])
                    require(n == len(indices) == status['samples'], 'world/body sample count mismatch')
                    require(world['object_transform_xyzw'].shape == (n, len(paths['body_paths']), 7), 'world object path/shape mismatch')
                    require(world['body_link_pos_w'].shape == (n, len(paths['robot_body_names']), 3), 'robot world shape mismatch')
                    require(world['foot_net_contact_force_w'].shape == (n, 2, 3), 'foot force shape mismatch')
                    require(all(np.isfinite(world[k]).all() for k in world.files), 'nonfinite world sample')
                    phase = status.get('sampling_phase', 'post_action')
                    expected_phase = 'pre_action' if meta.get('ego_render_schedule') == 'upstream' else 'post_action'
                    require(phase == expected_phase, 'world/robot sampling phase mismatch')
                    offset = 0. if phase == 'pre_action' else meta['sim_dt']
                    require(np.allclose(world['time'], robot['time'] + offset, rtol=0, atol=1e-6), 'world/robot clock mismatch')
                    require(np.all(np.diff(world['host_monotonic_ns']) > 0), 'world host clock not increasing')
                    return {'samples': n, 'scene_bodies': len(paths['body_paths']),
                            'robot_links': len(paths['robot_body_names']), 'sampling_phase': phase, 'world_clock_verified': True,
                            'max_foot_force_N': float(np.linalg.norm(world['foot_net_contact_force_w'], axis=2).max())}
            check('actual_physics_world_and_contacts', physical_world)
        if session and session["require_hands"]:
            check("revo2_state_and_command", lambda: require(bool(meta.get("hand_joint_names")) and all(
                np.all(robot[f"{side}_hand_command_valid"])
                and meta["dds_hand_command_counts"][side] > 0 for side in ("left", "right")),
                "11 physical joints and 7-slot DDS commands recorded for both Revo2 hands"))
    if (session and session.get("require_sonic_debug")) or (run / "sonic_debug.msgpack").exists():
        def sonic_debug():
            from sonic_debug_recorder import validate_recording
            summary, _ = validate_recording(run / "sonic_debug.msgpack")
            require(host is not None, "robot clock unavailable; fix robot_arrays first")
            require(summary["first_state_monotonic_ns"] / 1e9 <= host[0] + .1
                    and summary["last_state_monotonic_ns"] / 1e9 >= host[-1] - .1,
                    "SONIC state/action stream does not cover physical capture")
            return summary
        check("sonic_protocol_v2_durable_cycles", sonic_debug)
    def packets():
        require(host is not None, "robot clock unavailable; fix robot_arrays first")
        status = read_json(run / "pico_input/recording_status.json")
        require(status.get("closed") is True and not status["writer_still_running"], "packet writer not closed")
        require(status["dropped"] == 0 and not status["error"], "packet drops or writer error")
        files = sorted((run / "pico_input").glob("pose_*.npz"))
        require(len(files) == status["written"] == status["submitted"] and len(files) >= 2,
                "packet submitted/written/file counts differ")
        require([p.name for p in files] == [f"pose_{i:06d}.npz" for i in range(len(files))], "missing packet indices")
        human, _ = load_human(run)
        packet_times = human['packet_time']
        if session.get('manual_recording'):
            from recording_window import recording_window
            window = recording_window(run)
            packet_times = packet_times[(packet_times >= window['start_monotonic_ns'] / 1e9)
                                        & (packet_times < window['end_monotonic_ns'] / 1e9)]
            require(len(packet_times) >= 2, 'recording interval has fewer than two human packets')
        gap_summary = human_packet_gap_summary(packet_times)
        max_human_gap = gap_summary["max_packet_gap_s"]
        require(gap_summary["capture_gap_gate_passed"],
                f"human clock gap {max_human_gap * 1000:.1f}ms exceeds "
                f"{MAX_HUMAN_PACKET_GAP_S * 1000:.0f}ms capture gate")
        for field in ("left_hand_joints", "right_hand_joints", "left_trigger", "right_trigger", "left_grip", "right_grip"):
            require(field in human and np.isfinite(human[field]).all(), f"missing controller field: {field}")
        if session["input_kind"] == "live_pico":
            require("body_poses_raw" in human and "headset_pose" in human, "raw PICO tracking missing")
        modes = mode_samples(run / "sonic_state/encoder_mode.csv", host)
        pose = modes == 2
        require(pose.any(), "no actual SONIC SMPL/POSE mode=2")
        require(np.count_nonzero(pose & (host >= human["packet_time"][0]) &
                                 (host <= human["packet_time"][-1])) >= 2, "no shared human/robot POSE interval")
        detail["human"] = human
        detail["modes"] = modes
        return {"packets": len(files), "duration_s": float(np.ptp(human["packet_time"])),
                "pose_samples": int(pose.sum()), **gap_summary}
    check("human_control_and_pose_coverage", packets)
    if session and session["require_camera"]:
        def camera():
            require(host is not None, "robot clock unavailable; fix robot_arrays first")
            import cv2
            status = read_json(run / "ego_view.recording.json")
            require(status["closed"] is True, "camera writer not finalized")
            with (run / "ego_view.timestamps.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            require(len(rows) == status["frames"] >= 2, "video/timestamp count mismatch")
            require([int(row["frame"]) for row in rows] == list(range(len(rows))), "camera frame sequence gap")
            times = np.array([float(row["host_monotonic_s"]) for row in rows])
            sim = np.array([float(row["sim_time_s"]) for row in rows])
            require(np.isfinite([times, sim]).all() and np.all(np.diff(times) > 0)
                    and np.all(np.diff(sim) >= 0), "invalid camera clocks")
            cap = cv2.VideoCapture(str(run / "ego_view.mp4"))
            count = 0
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    require(frame.shape[:2] == (status["height"], status["width"]), "video dimensions changed")
                    count += 1
            finally:
                cap.release()
            require(count == len(rows), f"decoded {count} frames, expected {len(rows)}")
            require(times[0] <= host[0] + .25 and times[-1] >= host[-1] - .25, "video does not cover physical capture")
            active = times[(times >= host[0]) & (times <= host[-1])]
            require(len(active) >= 2 and np.max(np.diff(active)) <= .5, "camera gap >500ms during physics")
            detail["camera_time"] = times
            detail["camera_sim_time"] = sim
            detail['camera_rows'] = rows
            if meta.get('ego_render_schedule') == 'upstream':
                steps = np.array([int(row.get('source_physics_step', -1)) for row in rows])
                active_steps = steps[steps >= 0]
                require(len(active_steps) >= 2 and active_steps[0] == 0
                        and np.all(np.diff(active_steps) == 8), 'upstream native camera cadence mismatch')
                require(np.allclose(sim[steps >= 0], active_steps * meta['sim_dt'], atol=1e-6, rtol=0),
                        'camera source physics time mismatch')
            return {"decoded_frames": count, "capture_span_s": float(np.ptp(times)),
                    "raw_mp4_duration_s": count / status["nominal_fps"],
                    "timing": "timestamp sidecar is authoritative; MP4 nominal FPS is not capture time"}
        check("video_decode_timestamps_and_coverage", camera)
    temporary = list(run.rglob("*.tmp"))
    check("no_partial_files", lambda: require(not temporary, f"temporary files: {temporary[:5]}"))
    if verify_hashes and (run / "checksums.json").exists():
        def checksums():
            entries = read_json(run / "checksums.json")["files"]
            for entry in entries:
                require(fingerprint(run / entry["relative_path"])["sha256"] == entry["sha256"],
                        f"checksum mismatch: {entry['relative_path']}")
            return {"verified_files": len(entries)}
        check("artifact_checksums", checksums)
    if (run / "meta/info.json").exists():
        from export_capture_dataset import validate
        check("reference_2609_parquet_metadata_video", lambda: validate(run))
    if verify_hashes:
        def completed_package():
            require(read_json(run / "completion.json")["status"] == "complete", "session not finalized")
            require((run / "checksums.json").is_file(), "missing artifact checksums")
            if session and session["require_camera"]:
                require(read_json(run / "meta/export_status.json")["complete"] is True, "dataset export incomplete")
            return "completion and export markers present"
        check("completed_package", completed_package)
    return {"schema_version": 1, "checked_utc": utc_now(),
            "status": "complete" if all(c["passed"] for c in checks) else "incomplete",
            "stage": "development", "formal_collection_approved": False,
            "checks": checks}, detail


def _robot_summary(loaded, detail):
    detail["robot"] = loaded
    detail["startup_samples"] = loaded[3]
    return {"physical_samples": len(loaded[2]), "startup_samples": loaded[3]}


def publish_mentor_submission(run, report):
    """Build the user-facing sample only after the raw capture is complete."""
    if report["status"] != "complete":
        return report
    if not read_json(Path(run) / "session.json").get("publish_mentor_sample", False):
        return report
    try:
        from prepare_mentor_sample import ensure_submission
        session = read_json(Path(run) / "session.json")
        configured_destination = session.get("mentor_only_output_directory")
        if configured_destination:
            submission = ensure_submission(run, update_latest=False,
                                           destination=configured_destination,
                                           write_validation=False, atomic_publish=True)
        elif session.get("input_kind", "live_pico") == "live_pico":
            submission = ensure_submission(run)
        else:
            submission = ensure_submission(run, update_latest=False)
        capture_notice(f"[MENTOR] complete: {submission['directory']}")
    except Exception as exc:
        submission = {"status": "incomplete", "error": f"{type(exc).__name__}: {exc}"}
        capture_notice(f"[MENTOR] incomplete; original capture remains saved: {submission['error']}")
    return {**report, "mentor_submission": submission}


def finalize(run, if_ready=False):
    run = Path(run).resolve()
    if (run / "meta/info.json").is_file() and not (run / "session.json").exists():
        from prepare_mentor_sample import validate_submission
        return {**validate_submission(run), "directory": str(run)}
    try:
        session = read_json(run / "session.json")
    except FileNotFoundError:
        if if_ready and not run.exists():
            # Another component's finalizer has already promoted this run.
            return {"status": "already_finalized"}
        raise
    try:
        lock = (run / ".finalize.lock").open("a")
    except FileNotFoundError:
        return _revalidate_promoted_output(session)
    with lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (run / "session.json").exists():
            return _revalidate_promoted_output(session)
        if if_ready and not all(component_state(run, c).exists() and
            read_json(component_state(run, c))["status"] in ("exited", "failed", "segment_closed") for c in COMPONENTS):
            return {"status": "waiting_for_components"}
        with FinalizationProgress(run) as progress:
            result = progress.finish(_finalize_artifacts(run, progress))
            result = _publish_mentor_only_output(run, result)
            if progress.state['deferred_signals']:
                result['deferred_terminal_signals'] = progress.state['deferred_signals'][:]
            return result


def _revalidate_promoted_output(session):
    from prepare_mentor_sample import validate_submission
    sample = Path(session["mentor_only_output_directory"])
    validation = validate_submission(sample)
    origin = read_json(sample / "meta/info.json")["script_config"]["submission"]
    if origin["source_session_id"] != session["session_id"]:
        raise RuntimeError("promoted output belongs to a different session")
    return {**validation, "directory": str(sample)}


def _write_text_pointer(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(str(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_mentor_only_output(run, result):
    """After a complete live scene capture, retain only the Mentor delivery."""
    run = Path(run).resolve()
    session_path = run / "session.json"
    if not session_path.is_file():
        return result
    session = read_json(session_path)
    output_value = session.get("mentor_only_output_directory")
    if not output_value:
        return result
    if (result.get("status") != "complete"
            or result.get("mentor_submission", {}).get("status") != "complete"):
        return result

    output = Path(output_value).resolve()
    publication_root = Path(session["publication_root"]).resolve()
    work_root = publication_root / ".capture_work"
    if run.parent != work_root or output.parent != publication_root or not output.is_dir():
        raise RuntimeError("Mentor-only cleanup path does not match the isolated capture work layout")
    from prepare_mentor_sample import validate_submission
    validation = validate_submission(Path(result["mentor_submission"]["directory"]))
    if validation.get("status") != "complete" or not validation.get("mentor_file_layout_match"):
        raise RuntimeError("refusing to remove capture work before the minimal Mentor sample validates")
    sample = Path(result["mentor_submission"]["directory"]).resolve()
    if sample != output:
        raise RuntimeError(f"published Mentor directory differs from configured output: {sample}")
    origin = read_json(sample / "meta/info.json")["script_config"]["submission"]
    if origin["source_session_id"] != session["session_id"]:
        raise RuntimeError("refusing to remove capture work: Mentor output belongs to a different session")
    if (run / 'world_trace.npz').is_file():
        from submission_replay import validate as validate_original_replay
        if validate_original_replay(sample) is None:
            raise RuntimeError('refusing to remove original world trajectory before its retained replay validates')

    pointer = publication_root / "latest_pico_run.txt"
    # Do not steal a newer Bridge launch's active pointer while finishing this one.
    if not session.get('diagnostic_preserve_latest') and (not pointer.exists() or Path(pointer.read_text().strip()) in (run, sample)):
        _write_text_pointer(pointer, sample)
    if not session.get('diagnostic_preserve_latest'):
        _write_text_pointer(publication_root / "latest_mentor_sample.txt", sample)
    shutil.rmtree(run)
    sync_directory(work_root)
    try:
        work_root.rmdir()
    except OSError:
        pass
    else:
        sync_directory(publication_root)
    capture_notice(f"[CAPTURE] Mentor-only output retained: {sample}")
    return result


def _finalize_artifacts(run, progress):
    from audit_pico_response import collect
    from capture_manifest import fingerprint
    from prepare_trajectory import prepare
    run = Path(run).resolve()
    # Caller owns the lock and signal guard for this complete transaction.
    if (run / "completion.json").exists():
        # Idempotence does not exempt an existing package from validation.
        progress.stage("revalidating_completed_capture")
        return publish_mentor_submission(run, inspect(run)[0])
    progress.stage("checking_closed_raw_capture")
    report, detail = inspect(run, verify_hashes=False)
    control = run / 'recording_control.json'
    if (read_json(run / 'session.json').get('manual_recording') and control.exists()
            and read_json(control).get('status') == 'discarded'):
        report['status'] = 'discarded'
        atomic_json(run / 'integrity_report.json', report)
        atomic_json(run / 'discarded.json', {'status': 'discarded', 'reason': 'operator left grip + B',
                    'raw_diagnostics_retained': True, 'training_sample_published': False})
        capture_notice(f'[CAPTURE] discarded: {run}; raw diagnostics retained; no training sample published')
        return report
    if report["status"] == "complete":
        try:
            if not (run / "pico_replay_archive.npz").exists():
                progress.stage("preparing_human_replay_archive")
                with (run / "pico_replay_archive.npz").open("xb") as stream:
                    np.savez_compressed(stream, **collect(run))
                    stream.flush()
                    os.fsync(stream.fileno())
            if not (run / "prepared").exists():
                progress.stage("preparing_robot_trajectory")
                quality = prepare(run, run, run / "prepared", preview=False)
            else:
                quality = read_json(run / "prepared/quality_report.json")
            if read_json(run / "session.json")["require_camera"]:
                progress.stage("exporting_standard_data_metadata_video")
                from export_capture_dataset import export, validate
                dataset_check = (validate(run) if (run / "meta/info.json").exists() else
                                 export(run, task=read_json(run / "session.json").get(
                                     "task", "PICO SONIC POSE development capture")))
                report["checks"].append({"name": "reference_2609_export", "passed": True, "detail": dataset_check})
            human = detail["human"]
            progress.stage("writing_time_alignment")
            if read_json(run / 'session.json').get('manual_recording'):
                from prepare_trajectory import load_robot
                meta, robot, _, _ = load_robot(run, apply_recording_window=True)
            else:
                meta, robot, _, _ = detail["robot"]
            host = (robot["state_monotonic_ns"] / 1e9 if "state_monotonic_ns" in robot
                    else meta["started_monotonic"] + robot["wall_time"])
            alignment = {"host_monotonic_s": host, "sim_time_s": robot["time"],
                "human_reference_row": np.searchsorted(human["packet_time"], host, side="right") - 1}
            if "camera_time" in detail:
                from upstream_camera import select_camera_frames
                alignment["video_frame"] = select_camera_frames(detail['camera_rows'], host, robot['time'],
                                                                meta.get('ego_render_schedule'))
            # -1 or a stale row is explicitly invalid; never fill across gaps.
            rows = alignment["human_reference_row"]
            alignment["human_age_s"] = host - human["packet_time"][np.maximum(rows, 0)]
            from prepare_trajectory import mode_samples
            alignment_modes = mode_samples(run / 'sonic_state/encoder_mode.csv', host)
            alignment["human_valid"] = human_reference_valid(rows, alignment["human_age_s"], alignment_modes)
            with (run / "prepared/alignment.npz").open("wb") as stream:
                np.savez_compressed(stream, **alignment)
                stream.flush()
                os.fsync(stream.fileno())
            report["trajectory_quality"] = quality["status"]
            report["checks"].append({"name": "trajectory_quality", "passed": quality["status"] == "development_checks_pass",
                                     "detail": "prepared/quality_report.json"})
            report["status"] = "complete" if all(c["passed"] for c in report["checks"]) else "incomplete"
            if report['status'] == 'complete' and (run / 'training/manifest.json').is_file():
                progress.stage('assessing_training_readiness')
                from assess_training_readiness import ensure_capture_assessment
                assessment = ensure_capture_assessment(run, run / 'training/readiness', integrity_status='complete')
                report['training_assessment'] = {key: assessment[key] for key in
                    ('status', 'machine_quality_passed', 'technical_quality_passed', 'camera_calibration_excluded', 'metrics')}
                info_path = run / 'meta/info.json'
                info = read_json(info_path)
                info['script_config']['training_assessment'] = {
                    **report['training_assessment'], 'formal_collection_approved': False,
                    'task_outcome': 'unreviewed', 'brainco_equivalent': False,
                    'unavailable_fields_exclude_from_training': assessment['unavailable_fields_exclude_from_training']}
                atomic_json(info_path, info)
                capture_notice(f"[TRAINING] {assessment['status']}; see {run}/training/readiness/report.json")
        except Exception as exc:
            report["status"] = "incomplete"
            report["checks"].append({"name": "replay_export", "passed": False, "detail": f"{type(exc).__name__}: {exc}"})
    atomic_json(run / "integrity_report.json", report)
    if report["status"] == "complete":
        progress.stage("syncing_artifacts_and_checksums")
        # Hashes verify contents; they do not make buffered artifacts durable.
        sync_tree(run)
        entries = []
        for path in sorted(run.rglob("*")):
            # Outer diagnostic stdout can still receive this finalizer's
            # message; primary component logs have already closed.
            if (path.is_file() and path.name not in ("checksums.json", "completion.json", FINALIZATION_JOURNAL)
                    and not path.name.startswith("supervisor_") and path.suffix != ".lock"):
                entries.append({"relative_path": str(path.relative_to(run)), **fingerprint(path)})
        atomic_json(run / "checksums.json", {"files": entries, "algorithm": "sha256"})
        atomic_json(run / "completion.json", {"status": "complete", "completed_utc": utc_now(),
                    "stage": "development", "formal_collection_approved": False,
                    "integrity_report": "integrity_report.json", "checksums": "checksums.json"})
    capture_notice(f"[CAPTURE] {report['status']}: {run}/integrity_report.json")
    if report["status"] == "complete":
        progress.stage("publishing_mentor_sample")
    return publish_mentor_submission(run, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    init = sub.add_parser("init")
    init.add_argument("--run", type=Path, required=True)
    init.add_argument("--profile", default="bridge-supermarket")
    init.add_argument("--input-kind", default="live_pico")
    init.add_argument("--manual-recording", action="store_true")
    init.add_argument("--task", help="actual task instruction, recorded before collection; does not certify success")
    init.add_argument("--mentor-only-output", type=Path,
                      help="publish this live scene capture directly as the only retained dataset")
    init.add_argument("--publication-root", type=Path,
                      help="outputs directory that owns the Mentor-only output and latest pointers")
    runner = sub.add_parser("run-component")
    runner.add_argument("--run", type=Path, required=True)
    runner.add_argument("--component", choices=COMPONENTS, required=True)
    runner.add_argument("command", nargs=argparse.REMAINDER)
    launch_check = sub.add_parser("preflight")
    launch_check.add_argument("--run", type=Path, required=True)
    launch_check.add_argument("--component", choices=("sonic", "manager"), required=True)
    for action in ("finalize", "check", "stop"):
        cmd = sub.add_parser(action)
        cmd.add_argument("run", type=Path)
    args = parser.parse_args()
    if args.action == "init":
        initialize(args.run, args.profile, args.input_kind, args.manual_recording, args.task,
                   args.mentor_only_output, args.publication_root)
    elif args.action == "preflight":
        print(json.dumps(preflight(args.run, args.component)))
    elif args.action == "run-component":
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        if not command:
            parser.error("a child command is required")
        raise SystemExit(supervise(args.run, args.component, command))
    elif args.action == "stop":
        session = read_json(args.run / 'session.json')
        if session.get('finalization_owner') == 'batch_coordinator':
            from batch_runtime import request_batch_stop
            pid = request_batch_stop(session['batch_runtime_directory'])
            print(f'Requested normal batch stop: coordinator PID {pid}')
            return
        state = read_json(component_state(args.run, "manager"))
        pid = state["supervisor_pid"]
        command = Path(f"/proc/{pid}/cmdline").read_bytes()
        if b"capture_session.py" not in command or str(args.run.resolve()).encode() not in command:
            raise RuntimeError("Manager supervisor PID no longer belongs to this session")
        os.kill(pid, signal.SIGINT)
        print(f"Requested normal Manager stop: {args.run}")
    else:
        if args.action == "check" and args.run.is_dir() and not (args.run / "session.json").exists():
            from prepare_mentor_sample import validate_submission
            result = validate_submission(args.run)
        else:
            result = finalize(args.run) if args.action == "finalize" else inspect(args.run)[0]
        capture_notice(json.dumps(result, indent=2))
        success = (result["status"] in ("complete", "discarded")
                   and result.get("mentor_submission", {}).get("status") != "incomplete")
        raise SystemExit(0 if success else 2)


if __name__ == "__main__":
    main()
