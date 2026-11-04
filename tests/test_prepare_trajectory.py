import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from prepare_trajectory import (ROBOT_FIELDS, align_reference, forward_kinematics,
                                validate_fk_manifest, file_info,
                                evaluate_quality, load_human, load_robot, mode_samples, segment_ids)
from sonic_contract import SONIC_CPP_MOTOR_ORDER


class TrajectoryExportTests(unittest.TestCase):
    def test_body_fk_identity_is_unambiguous_with_reference_and_hand_urdfs(self):
        with tempfile.TemporaryDirectory() as tmp:
            body=Path(tmp)/'captured_body.urdf';body.write_text('<robot name="body"/>')
            entry=dict(path='/repo/robot_description/urdf/g1/main.urdf',sha256=file_info(body)['sha256'])
            hands=[dict(path='/reference/revo2_'+s+'.urdf',sha256='different') for s in ('left','right')]
            for role in (None,'body_fk_urdf'):
                deployed=dict(entry)
                if role:
                    deployed.update(path='/snapshot/deployed_body.urdf',role=role)
                self.assertIn('matches',validate_fk_manifest({'files':hands+[deployed]},body))
                deployed['sha256']='tampered'
                with self.assertRaisesRegex(ValueError,'FK asset differs'):
                    validate_fk_manifest({'files':hands+[deployed]},body)
            with self.assertRaisesRegex(ValueError,'FK asset differs'):
                validate_fk_manifest({'files':hands},body)

    def robot(self, root):
        n = 6
        metadata = dict(motor_joint_names=list(SONIC_CPP_MOTOR_ORDER),
                        isaac_joint_names=list(SONIC_CPP_MOTOR_ORDER),
                        motor_to_isaac_joint_index=list(range(29)),
                        velocity_limits_motor_order=[30.] * 29,
                        position_limits_motor_order={"low": [-3.] * 29, "high": [3.] * 29})
        data = {k: np.zeros((n, width)) for k, (_, width) in ROBOT_FIELDS.items()}
        data.update(time=np.array([0., 0., 0., .02, .04, .06]), wall_time=np.arange(n) * .02,
                    startup_waiting=np.array([1, 1, 0, 0, 0, 0]),
                    controller_active=np.array([0, 0, 1, 1, 0, 1]),
                    raw_command_valid=np.ones(n), fault_latched=np.array([0, 0, 0, 0, 1, 0]),
                    command_age=np.array([np.inf, np.inf, .001, .001, .31, .001]),
                    base_tilt_rad=np.zeros(n), dds_accepted_count=np.arange(n), dds_rejected_count=np.zeros(n))
        data["root_quat_wxyz"][:, 0] = 1
        self.save_robot(root, metadata, data)
        return metadata, data

    def save_robot(self, root, metadata, data):
        (root / "bridge.json").write_text(json.dumps(metadata))
        np.savez(root / "bridge.npz", **data)

    def test_excludes_only_frozen_startup_and_preserves_fault_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.robot(root)
            _, data, indices, excluded = load_robot(root)
            self.assertEqual(excluded, 2)
            np.testing.assert_array_equal(indices, [2, 3, 4, 5])
            np.testing.assert_array_equal(data["fault_latched"], [0, 0, 1, 0])
            np.testing.assert_allclose(data["time"], [0., .02, .04, .06])

    def test_rejects_nonfinite_state_after_startup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata, data = self.robot(root)
            data["joint_pos"][3, 0] = np.nan
            self.save_robot(root, metadata, data)
            with self.assertRaisesRegex(ValueError, "nonfinite physical"):
                load_robot(root)

    def test_rejects_ambiguous_motor_permutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata, data = self.robot(root)
            metadata["motor_to_isaac_joint_index"][:2] = [1, 0]
            self.save_robot(root, metadata, data)
            with self.assertRaisesRegex(ValueError, "permutation"):
                load_robot(root)

    def test_shared_revo2_articulation_preserves_exact_body_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata, data = self.robot(root)
            # Interleave additional joints to exercise a nontrivial mapping.
            names = list(SONIC_CPP_MOTOR_ORDER)
            metadata["isaac_joint_names"] = [f"hand_{i}" for i in range(11)] + names + [f"hand_{i}" for i in range(11, 22)]
            metadata["motor_to_isaac_joint_index"] = list(range(11, 40))
            self.save_robot(root, metadata, data)
            self.assertEqual(len(load_robot(root)[0]["isaac_joint_names"]), 51)
            for bad in ([11] * 29, list(range(29)), list(range(11, 39)) + [51]):
                metadata["motor_to_isaac_joint_index"] = bad
                self.save_robot(root, metadata, data)
                with self.assertRaisesRegex(ValueError, "permutation"):
                    load_robot(root)

    def test_rejects_repeated_physical_time_and_return_to_startup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata, data = self.robot(root)
            data["time"][4] = .02
            self.save_robot(root, metadata, data)
            with self.assertRaisesRegex(ValueError, "strictly increase"):
                load_robot(root)
            data["time"][4] = .04
            data["startup_waiting"][4] = 1
            self.save_robot(root, metadata, data)
            with self.assertRaisesRegex(ValueError, "resumed"):
                load_robot(root)

    def test_segments_split_at_faults_and_recording_gaps(self):
        valid = np.array([1, 1, 0, 1, 1, 1], dtype=bool)
        times = np.array([0., .02, .04, .06, .20, .22])
        np.testing.assert_array_equal(segment_ids(valid, times, .02), [0, 0, -1, 1, 2, 2])

    def test_replay_aligns_distinct_capture_and_publication_clocks(self):
        human = dict(packet_time=np.array([100., 100.5, 101.]),
                     time_from_first_packet_s=np.array([0., .5, 1.]))
        summary = dict(test="pose", publication_start_monotonic=1000., publication_end_monotonic=1007.,
                       pose_start_seconds=0., planner_warmup_seconds=5., source_duration_seconds=1.)
        phase, ref, rows, _ = align_reference(np.array([1004., 1005., 1005.6, 1006.5, 1008.]),
                                             summary, human, np.full(5, 2))
        np.testing.assert_array_equal(phase, [0, 1, 1, 2, 3])
        np.testing.assert_array_equal(rows, [-1, 0, 1, 2, -1])
        self.assertTrue(np.isnan(ref[0]))
        self.assertAlmostEqual(ref[2], .6)
        summary["source_duration_seconds"] = 2.
        with self.assertRaisesRegex(ValueError, "source duration"):
            align_reference(np.array([1005.]), summary, human, np.array([2]))

    def test_live_does_not_extend_pose_outside_reference_clock(self):
        human = dict(packet_time=np.array([100., 101.]), time_from_first_packet_s=np.array([0., 1.]))
        phase, _, rows, _ = align_reference(np.array([99., 100., 100.5, 102.]), {}, human, np.array([0, 2, 2, 2]))
        np.testing.assert_array_equal(phase, [0, 1, 1, 3])
        np.testing.assert_array_equal(rows, [-1, 0, 0, -1])

    def test_wrong_mode_after_entry_window_is_not_hidden(self):
        human = dict(packet_time=np.array([100., 101.]), time_from_first_packet_s=np.array([0., 1.]))
        summary = dict(test="pose", publication_start_monotonic=1000., publication_end_monotonic=1007.,
                       pose_start_seconds=0., planner_warmup_seconds=5., source_duration_seconds=1.)
        phase, _, rows, _ = align_reference(np.array([1005.02, 1005.08, 1005.20, 1005.40]),
                                           summary, human, np.array([0, 2, 0, -1]))
        np.testing.assert_array_equal(phase, [4, 1, 1, 1])
        np.testing.assert_array_equal(rows, [0, 0, 0, 0])

    def test_mode_coverage_never_extrapolates_across_logging_gaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mode.csv"
            path.write_text("index,time_monotonic_ms,encoder_mode_0\n0,100000,0\n1,100020,2\n2,100400,0\n")
            modes = mode_samples(path, np.array([99., 100.01, 100.03, 100.3, 100.41]))
            np.testing.assert_array_equal(modes, [-1, 0, 2, -1, 0])

    def test_human_raw_join_refuses_mismatched_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "pico_input"
            raw.mkdir()
            np.savez(root / "pico_replay_archive.npz", packet_time=[100., 100.02], source_frame=[0, 1],
                     smpl_pose=np.zeros((2, 21, 3)), smpl_joints=np.zeros((2, 24, 3)),
                     body_quat_w=np.tile([1., 0., 0., 0.], (2, 1)), joint_pos=np.zeros((2, 29)))
            for i in range(2):
                np.savez(raw / f"pose_{i:06d}.npz", timestamp_monotonic=[100. + .02 * i], frame_index=[i + 5])
            with self.assertRaisesRegex(ValueError, "frame mismatch"):
                load_human(root)

    def test_fk_uses_recorded_world_translation_and_wxyz_rotation(self):
        from prepare_trajectory import DEFAULT_URDF
        q = np.zeros((2, 29))
        roots = np.array([[0., 0., 0.], [1., 2., 3.]])
        quat = np.array([[1., 0., 0., 0.], [np.sqrt(.5), 0., 0., np.sqrt(.5)]])
        result, _ = forward_kinematics(q, roots, quat, DEFAULT_URDF)
        local = result["ankle_pos_root_m"][1]
        expected = np.column_stack([-local[:, 1], local[:, 0], local[:, 2]]) + roots[1]
        np.testing.assert_allclose(result["ankle_pos_w_m"][1], expected, atol=1e-8)

    def test_fault_and_wrong_pose_mode_cannot_pass_quality_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.robot(root)
            metadata, data, _, excluded = load_robot(root)
            metadata.update(sim_dt=.005, record_every=4, fixed_base=False, target_policy="sonic-pd",
                            pd_backend="implicit", extra_waist_damping=0., pd_target_slew="none",
                            fault_reason="", dds_rejected_count=0, max_base_tilt_deg=60.,
                            stop_reason="local Manager requested simulation end before policy shutdown")
            result = dict(time_s=data["time"], q_rad=data["joint_pos"], dq_rad_s=data["joint_vel"],
                          root_quat_wxyz=data["root_quat_wxyz"], root_pos_w_m=np.tile([0., 0., .8], (4, 1)),
                          sample_valid=np.ones(4, dtype=bool), phase_id=np.ones(4), encoder_mode=np.full(4, 2),
                          pose_valid=np.ones(4, dtype=bool), pose_segment_id=np.zeros(4))
            human = dict(packet_time=np.array([100., 100.02, 100.04, 100.06]),
                         time_from_first_packet_s=np.array([0., .02, .04, .06]))
            quality = evaluate_quality(metadata, data, result, human, {}, excluded)
            self.assertEqual(quality["status"], "development_checks_pass")
            # Bounded stalls may be retained without accepting stale labels.
            # Outages over two seconds still fail the continuity gate.
            for gap, passed in ((.168, True), (.249, True), (1.025415796, True), (1.999, True), (2.001, False)):
                with self.subTest(human_gap=gap):
                    source = dict(packet_time=np.array([100., 100.02, 100.02 + gap, 100.04 + gap]),
                                  time_from_first_packet_s=np.array([0., .02, .02 + gap, .04 + gap]))
                    report = evaluate_quality(metadata, data, result, source, {}, excluded)
                    check = next(c for c in report["checks"] if c["name"] == "human_reference_gaps")
                    self.assertEqual(check["passed"], passed)
                    self.assertEqual(check["detail"]["capture_gap_gate_s"], 2.)
                    self.assertEqual(check["detail"]["gaps_over_100ms"], 1)
                    self.assertEqual(check["detail"]["gaps_over_gate"], 0 if passed else 1)
            result["encoder_mode"][2] = 0
            quality = evaluate_quality(metadata, data, result, human, {}, excluded)
            self.assertFalse(next(c for c in quality["checks"] if c["name"] == "pose_mode_coverage")["passed"])
            self.assertEqual(quality["status"], "development_review_required")
            result["encoder_mode"][2] = 2
            metadata["fault_reason"] = "watchdog"
            result["sample_valid"][2] = False
            result["pose_valid"][2] = False
            quality = evaluate_quality(metadata, data, result, human, {}, excluded)
            self.assertEqual(quality["status"], "development_review_required")


if __name__ == "__main__":
    unittest.main()
