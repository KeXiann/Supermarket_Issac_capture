"""Recording contract, actual LeRobot roundtrip, and video shutdown regression."""

import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, MagicMock, patch

import av
import numpy as np
import pyarrow.parquet as pq

from gear_sonic.data.body_alignment import BodyAlignmentError, aligned_body_frame, configure_body_alignment_schema
from gear_sonic.data.brainco_collection import BraincoRecordingState, configure_brainco_schema, brainco_metadata
from gear_sonic.utils.teleop.brainco_poses import load_poses, pose_targets
from gear_sonic.data.exporter import Gr00tDataExporter
from gear_sonic.data.features_sonic_vla import (
    get_g1_robot_model, get_features_sonic_vla, get_modality_config_sonic_vla,
)
from gear_sonic.data.video_writer import VideoWriter
from gear_sonic.scripts.run_data_exporter import GrootDataCollector, TimingThresholdMonitor
from gear_sonic.scripts.launch_data_collection import DataCollectionLaunchConfig, build_commands
from gear_sonic.utils.data_collection.episode_state import EpisodeState


def message(now=None, left=.8, right=.1, command=(0, 1)):
    now = time.time_ns() if now is None else now
    return {
        "version": 3, "timestamp_ns": now,
        "pose_config": load_poses(), "q_command": pose_targets(load_poses(), command),
        "monotonic_ns": time.monotonic_ns(), "command_session_id": 1,
        "command_history": [{"sequence": 1, "monotonic_ns": time.monotonic_ns() - 20_000_000,
                             "timestamp_ns": now - 20_000_000, "binary": list(command),
                             "q_command": pose_targets(load_poses(), command)}],
        "feedback": {side: {"q": [q] * 6, "dq": [.02] * 6, "valid": True,
                            "received_timestamp_ns": now - 10_000_000}
                     for side, q in (("left", left), ("right", right))},
        "binary_command": list(command), "command_timestamp_ns": now - 20_000_000,
        "pico_timestamp_ns": 777, "command_valid": True,
        "command_enabled": True, "dry_run": False,
    }


def update(state, payload):
    state.update(b"brainco_state" + json.dumps(payload).encode())


class ContractTests(unittest.TestCase):
    def test_measured_state_is_not_command_and_stale_is_invalid(self):
        state = BraincoRecordingState()
        now = time.time_ns()
        update(state, message(now))
        frame, ready = state.frame(now)
        self.assertTrue(ready)
        np.testing.assert_array_equal(frame["observation.hand_binary_state"], [1, 0])
        np.testing.assert_array_equal(frame["action.hand_binary"], [0, 1])
        np.testing.assert_allclose(frame["observation.hand_joint_position"][:6], .8)
        np.testing.assert_array_equal(frame["teleop.left_hand_joints"], [0, 1, 0, 0, 0, 0])
        self.assertFalse(state.frame(now + 500_000_000)[1])
        for change in ({"dry_run": True}, {"command_valid": False}, {"feedback": {}}):
            update(state, {**message(now), **change})
            self.assertFalse(state.frame(now)[1])

    def test_custom_actual_actions_feedback_and_history(self):
        poses = load_poses()
        poses["left"]["open"] = [0, .3, 0, 0, 0, 0]
        poses["right"]["closed"] = [.8, .4, .9, .8, .7, .6]
        with tempfile.NamedTemporaryFile(mode="w+") as config:
            json.dump(poses, config)
            config.flush()
            state = BraincoRecordingState(pose_config=config.name)
            payload = message()
            payload["pose_config"] = poses
            expected = pose_targets(poses, [0, 1])
            payload["q_command"] = expected
            payload["command_history"][0]["q_command"] = expected
            update(state, payload)
            frame, ready = state.frame(payload["timestamp_ns"])
            self.assertTrue(ready, state.error)
            np.testing.assert_allclose(frame["action.hand_joint_position"], expected["left"]+expected["right"])
            np.testing.assert_allclose(frame["observation.hand_joint_position"][:6], [.8]*6)
            aligned = state.aligned_commands(payload["monotonic_ns"], payload["monotonic_ns"])
            np.testing.assert_allclose(aligned["action.aligned_hand_joint_position"], frame["action.hand_joint_position"])
            # A new action must not leak into the earlier body-state sample.
            earlier = state.aligned_commands(1, 1)
            self.assertFalse(earlier["action.aligned_hand_valid"][0])
            for change in ({"q_command": None}, {"pose_config": load_poses()}, {"version": 2}):
                update(state, {**payload, **change})
                self.assertIsNone(state.message)

    def test_malformed_message_cannot_reuse_previous_good_state(self):
        state = BraincoRecordingState()
        update(state, message())
        state.update(b'brainco_state{"version":1,"binary_command":[2,0]}')
        self.assertIsNone(state.message)
        self.assertFalse(state.frame()[1])
        state.update(b'brainco_state{invalid')
        self.assertFalse(state.frame()[1])

    def test_launcher_quotes_prompt_and_enables_both_ends(self):
        config = DataCollectionLaunchConfig(brainco_interface="eno1", ego_view_height=360,
                    camera_host="192.168.123.164", dataset_name="trial", task_prompt="cup's $(touch /tmp/should_not_exist)")
        commands = build_commands(config)
        self.assertIn("--brainco-interface eno1", commands["teleop"])
        self.assertIn("--brainco-hands", commands["exporter"])
        self.assertIn("--brainco-pose-config", commands["teleop"])
        self.assertIn("--brainco-pose-config", commands["exporter"])
        self.assertIn("--ego-view-height 360", commands["exporter"])
        self.assertIn("--output-type zmq", commands["deploy"])
        import shlex
        args = shlex.split(commands["exporter"].split(" && ", 1)[1])
        self.assertEqual(args[args.index("--task-prompt") + 1], config.task_prompt)


class RoundtripTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = get_g1_robot_model()

    def schema(self):
        features = get_features_sonic_vla(self.model)
        configure_body_alignment_schema(features)
        features["observation.images.ego_view"]["shape"] = [360, 640, 3]
        modality = get_modality_config_sonic_vla(self.model)
        configure_brainco_schema(features, modality, self.model)
        return features, modality

    def collector(self, exporter):
        # Real frame builder and FK, with no hardware/network objects created.
        c = GrootDataCollector.__new__(GrootDataCollector)
        c.frequency = 50
        c.robot_model = self.model
        c.data_exporter = exporter
        c.text_to_speech = None
        c._brainco = BraincoRecordingState()
        c._body_indices = self.model.get_joint_group_indices("body")
        c._episode_state = EpisodeState()
        c._episode_state.change_state()
        c._initial_yaw = None
        c._last_latency_log_time = time.time()
        c.sonic_timing_monitor = TimingThresholdMonitor()
        c.current_stream_mode = 0
        c.latest_sonic_msg = c.latest_planner_msg = None
        c.latest_proprio_msg = {"body_q": np.arange(29, dtype=np.float64) * .001,
                              "last_action": np.full(29, -99.),
                              "body_action": np.arange(29, dtype=np.float32) * .002,
                              "body_action_alignment_version": 2, "index": 0, "body_action_index": 0,
                              "state_timestamp_ns": time.time_ns(),
                              "state_monotonic_ns": 100, "body_action_monotonic_ns": 200,
                              "body_dq": np.arange(29) * .01, "base_quat": [1.,0,0,0],
                              "base_ang_vel": [.1,.2,.3], "base_accel": [0.,0.,9.81],
                              "lowstate_receive_monotonic_ns": 50, "lowstate_tick": 1,
                              "previous_body_action": np.zeros(29), "previous_body_action_index": -1,
                              "previous_body_action_monotonic_ns": 0, "previous_body_action_valid": False}
        c._proprio_received_monotonic = time.monotonic()
        c._image_subscriber = Mock(_last_new_message_time=time.time())
        c.latest_image_msg = {"images": {"ego_view": np.full((360, 640, 3), [210, 30, 15], dtype=np.uint8)},
                              "timestamps": {"ego_view": time.time()}}
        update(c._brainco, message())
        return c

    def test_default_bottle_pose_joint_state_and_action_parquet(self):
        features, modality = self.schema()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "default_bottle"
            exporter = Gr00tDataExporter.create(root, 50, features, modality, "pick up the cola",
                script_config={"hand_recording": brainco_metadata()})
            expected_rows = []
            try:
                c = self.collector(exporter)
                for cycle, binary in enumerate(([0, 0], [1, 0], [0, 1], [1, 1])):
                    payload = message(command=binary)
                    # Distinct feedback in every channel: detect swaps and commands
                    # incorrectly substituted for observed state.
                    measured = np.arange(12, dtype=np.float32) * .05 + cycle * .01
                    velocity = np.arange(12, dtype=np.float32) * .01 - .05
                    for i, side in enumerate(("left", "right")):
                        payload["feedback"][side]["q"] = measured[i*6:(i+1)*6].tolist()
                        payload["feedback"][side]["dq"] = velocity[i*6:(i+1)*6].tolist()
                    cutoff = payload["monotonic_ns"]
                    c.latest_proprio_msg.update(index=cycle, body_action_index=cycle,
                        state_monotonic_ns=cutoff-2, body_action_monotonic_ns=cutoff-1)
                    update(c._brainco, payload)
                    c._add_data_frame_sonic(time.monotonic())
                    target = []
                    for goal in binary:
                        target.extend([1]*6 if goal else [0, 1, 0, 0, 0, 0])
                    expected_rows.append((measured, velocity, target))
                exporter.save_episode()
                table = pq.read_table(next(root.rglob("*.parquet"))).to_pydict()
                self.assertEqual(len(table["action.hand_joint_position"]), 4)
                for row, (measured, velocity, target) in enumerate(expected_rows):
                    np.testing.assert_allclose(table["observation.hand_joint_position"][row], measured)
                    np.testing.assert_allclose(table["observation.hand_joint_velocity"][row], velocity)
                    for key in ("action.hand_joint_position", "action.aligned_hand_joint_position",
                                "observation.previous_hand_action_joint_position"):
                        np.testing.assert_array_equal(table[key][row], target)
                    np.testing.assert_array_equal(table["teleop.left_hand_joints"][row], target[:6])
                    np.testing.assert_array_equal(table["teleop.right_hand_joints"][row], target[6:])
                    self.assertEqual(table["observation.hand_valid"][row], [1, 1])
                    self.assertEqual(table["action.hand_valid"][row], 1)
                    self.assertEqual(table["action.aligned_hand_valid"][row], 1)
                saved = json.loads((root / "meta/info.json").read_text())
                self.assertEqual(saved["script_config"]["hand_recording"], brainco_metadata())
            finally:
                exporter.stop_video_writers()

    def test_actual_parquet_mp4_two_episodes_and_resume_guard(self):
        features, modality = self.schema()
        self.assertEqual(features["observation.state"]["shape"], (29,))
        self.assertEqual(modality["state"]["right_arm"], {"start": 22, "end": 29})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            poses = load_poses()
            poses["left"]["closed"][1] = .35
            pose_path = Path(directory) / "poses.json"
            pose_path.write_text(json.dumps(poses))
            metadata = {"hand_recording": brainco_metadata(str(pose_path))}
            exporter = Gr00tDataExporter.create(root, 50, features, modality, "test only", script_config=metadata)
            try:
                c = self.collector(exporter)
                c._brainco = BraincoRecordingState(pose_config=str(pose_path))
                for episode in range(2):
                    for cycle in [0, 1, 1, 3, 4, 4, 7, 9]:
                        c.latest_proprio_msg.update(index=cycle, body_action_index=cycle,
                            body_q=np.arange(29) * .001 + cycle * .01,
                            body_action=np.asarray(np.arange(29) * .002 + cycle * .02, dtype=np.float32))
                        payload = message(command=(episode, 1 - episode))
                        payload["pose_config"] = poses
                        payload["q_command"] = pose_targets(poses, payload["binary_command"])
                        payload["command_history"][0]["q_command"] = payload["q_command"]
                        update(c._brainco, payload)
                        c._add_data_frame_sonic(time.monotonic())
                    exporter.save_episode()
                tables = sorted(root.rglob("*.parquet"))
                self.assertEqual(len(tables), 2)
                for index, path in enumerate(tables):
                    table = pq.read_table(path).to_pydict()
                    self.assertEqual(len(table["observation.state"]), 8)
                    self.assertEqual(len(table["observation.state"][0]), 29)
                    for row, cycle in enumerate([0, 1, 1, 3, 4, 4, 7, 9]):
                        self.assertEqual(table["observation.control_cycle_index"][row], cycle)
                        np.testing.assert_allclose(table["observation.state"][row], np.arange(29)*.001 + cycle*.01)
                        np.testing.assert_array_equal(table["action.wbc"][row],
                            np.asarray(np.arange(29)*.002 + cycle*.02, dtype=np.float32).astype(np.float64))
                        self.assertEqual(table["action.body_monotonic_ns"][row], 200)
                    np.testing.assert_allclose(table["observation.hand_joint_position"][0][:6], .8)
                    np.testing.assert_array_equal(table["action.hand_binary"][0], [index, 1-index])
                    self.assertEqual(table["observation.hand_binary_state"][0], [1, 0])
                    expected = pose_targets(poses, [index, 1-index])
                    np.testing.assert_allclose(table["action.hand_joint_position"][0], expected["left"]+expected["right"])
                    np.testing.assert_allclose(table["teleop.left_hand_joints"][0], expected["left"])
                videos = sorted(root.rglob("*.mp4"))
                nonempty = 0
                for path in videos:
                    if path.stat().st_size < 100:
                        continue
                    with av.open(str(path)) as video:
                        frames = list(video.decode(video=0))
                    self.assertEqual(len(frames), 8)
                    rgb = frames[0].to_ndarray(format="rgb24")
                    self.assertEqual(rgb.shape, (360, 640, 3))
                    self.assertGreater(rgb[..., 0].mean(), rgb[..., 2].mean() + 100)
                    nonempty += 1
                self.assertEqual(nonempty, 2)
                self.assertEqual(json.loads((root / "meta/info.json").read_text())["script_config"]["hand_recording"]["pose_config"], poses)
                with self.assertRaisesRegex(ValueError, "hand configuration differs"):
                    Gr00tDataExporter.create(root, 50, features, modality, "wrong mapping",
                                             script_config={"hand_recording": brainco_metadata()})
                wrong_features, _ = self.schema()
                wrong_features["observation.images.ego_view"]["shape"] = [480, 640, 3]
                with self.assertRaisesRegex(ValueError, "differs"):
                    Gr00tDataExporter.create(root, 50, wrong_features, modality, "wrong")
            finally:
                exporter.stop_video_writers()

    def test_v2_capture_merge_and_semanticdpc_conversion(self):
        import subprocess
        features, modality = self.schema()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            exporter = Gr00tDataExporter.create(source, 50, features, modality, 'synthetic',
                                                script_config={'body_action_alignment_version': 2})
            try:
                c = self.collector(exporter)
                start = time.monotonic_ns() - 60_000_000
                for i in range(3):
                    stamp = start + i * 20_000_000
                    c.latest_proprio_msg.update(index=10+i, body_action_index=10+i,
                        state_monotonic_ns=stamp, body_action_monotonic_ns=stamp+2_000_000,
                        lowstate_receive_monotonic_ns=stamp-1_000_000, lowstate_tick=100+i*20,
                        body_action=np.full(29, i*.01, np.float32),
                        previous_body_action=np.full(29, (i-1)*.01, np.float32),
                        previous_body_action_index=9+i, previous_body_action_monotonic_ns=stamp-18_000_000,
                        previous_body_action_valid=True)
                    telemetry = message()
                    telemetry['monotonic_ns'] = stamp + 3_000_000
                    telemetry['command_history'][0]['monotonic_ns'] = start - 5_000_000
                    update(c._brainco, telemetry)
                    # A newer, valid body message is not yet covered by hand telemetry.
                    newer = {**c.latest_proprio_msg, 'index':11+i, 'body_action_index':11+i,
                             'state_monotonic_ns':stamp+20_000_000,
                             'body_action_monotonic_ns':stamp+22_000_000,
                             'previous_body_action_index':10+i,
                             'previous_body_action_monotonic_ns':stamp+2_000_000,
                             'previous_body_action':c.latest_proprio_msg['body_action']}
                    c._proprio_history = [dict(c.latest_proprio_msg), newer]
                    # Deterministic acquisition latency; do not wait on real hardware clocks.
                    with patch('gear_sonic.scripts.run_data_exporter.time.monotonic_ns', return_value=stamp+5_000_000):
                        c._add_data_frame_sonic(time.monotonic())
                exporter.save_episode()
            finally:
                exporter.stop_video_writers()
            selection = root / 'selection.txt'
            selection.write_text(str(source)+'\nkeep: 0\n')
            scripts = Path(__file__).resolve().parents[2] / 'scripts'
            commands = [
                ['bash', str(scripts/'process_reviewed_datasets.sh'), '--selection-file', str(selection), '--output-path', str(root/'merged')],
                ['bash', str(scripts/'convert_semanticdpc_dataset.sh'), '--dataset-path', str(root/'merged'), '--output-path', str(root/'semantic')],
            ]
            for command in commands:
                result = subprocess.run(command, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            with np.load(root/'semantic/episode_000000.npz') as output:
                self.assertTrue(output['sample_valid'].all())
                np.testing.assert_array_equal(output['source_control_cycle_index'], [10,11,12])
                np.testing.assert_array_equal(output['history_mask'].sum(axis=1), [1,2,3])
                np.testing.assert_allclose(output['history10'][:, -1, :3], np.tile([.1,.2,.3], (3,1)))

    def test_alignment_rejects_legacy_invalid_and_mismatched_frames(self):
        c = self.collector(Mock())
        valid = dict(c.latest_proprio_msg)
        changes = [{"body_action_alignment_version": 0}, {"body_action_index": 1},
                   {"body_action": []}, {"body_action": [float("nan")] * 29},
                   {"state_monotonic_ns": 300}, {"index": 0.5}, {"state_timestamp_ns": -1}]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                aligned_body_frame({**valid, **change})
        for field in ("body_action", "body_action_alignment_version", "body_action_index"):
            payload = dict(valid)
            del payload[field]
            with self.subTest(missing=field), self.assertRaises(ValueError):
                aligned_body_frame(payload)

    def test_protocol_failure_discards_interrupted_episode(self):
        exporter = Mock(episode_buffer={"size": 1})
        c = self.collector(exporter)
        c.telemetry = MagicMock()
        c._poll_state_zmq = Mock(side_effect=BodyAlignmentError("bad pair"))
        c.save_and_cleanup = Mock()
        with self.assertRaises(BodyAlignmentError):
            c.run()
        exporter.save_episode_as_discarded.assert_called_once()
        c.save_and_cleanup.assert_called_once()

    def test_start_gating_and_disconnect_marks_episode_discarded(self):
        exporter = Mock(episode_buffer={"size": 1, "episode_index": 0})
        c = self.collector(exporter)
        c._brainco.message = None
        c._add_data_frame()
        exporter.save_episode_as_discarded.assert_called_once()
        self.assertEqual(c._episode_state.get_state(), c._episode_state.IDLE)
        c._keyboard_listener = Mock()
        c._keyboard_listener.read_msg.return_value = "c"
        c._manager_toggle_da = c._manager_toggle_dc = False
        c._check_recording_commands()
        self.assertEqual(c._episode_state.get_state(), c._episode_state.IDLE)

    def test_video_immediate_stop_drains_last_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "video.mp4")
            writer = VideoWriter(path, 64, 48, 30, buffer_size=2)
            for _ in range(13):
                writer.add_frame(np.full((48, 64, 3), 180, dtype=np.uint8))
            writer.stop()
            writer.stop()
            with av.open(path) as video:
                self.assertEqual(len(list(video.decode(video=0))), 13)
            self.assertFalse(writer._thread.is_alive())


if __name__ == "__main__":
    unittest.main()
