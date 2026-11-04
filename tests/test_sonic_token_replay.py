import unittest

import numpy as np

from sonic_motion_replay import rotation6d_matrix, unpack_motion_output
from validate_sonic_tokens import gravity_cpp, reconstruct_observations, restore_policy_states


class SonicTokenReplayTests(unittest.TestCase):
    def test_motion_layout_restores_all_positions_before_all_velocities(self):
        # Official command_multi_future: cat(q.flatten(), dq.flatten()), then [10,58].
        q = np.arange(290, dtype=np.float32).reshape(10, 29)
        dq = q+1000
        command = np.r_[q.ravel(), dq.ravel()].reshape(10, 58)
        rot = np.tile([1., 0., 0., 1., 0., 0.], (10, 1))
        output = np.c_[command, rot].reshape(1, 640)
        order = np.arange(29)[::-1]
        restored_q, restored_dq, restored_rot = unpack_motion_output(output, order)
        np.testing.assert_array_equal(restored_q[0], q[:, order])
        np.testing.assert_array_equal(restored_dq[0], dq[:, order])
        np.testing.assert_array_equal(restored_rot[0], rot)
        # A naive per-frame q29/dq29 split must be detectably wrong.
        self.assertFalse(np.array_equal(command[:, :29], q))

    def test_rotation_uses_row_major_first_two_columns(self):
        desired = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        np.testing.assert_allclose(rotation6d_matrix(desired[:, :2].reshape(1, 6))[0], desired)
        with self.assertRaises(ValueError):
            rotation6d_matrix(np.zeros((1, 6)))

    def test_wire_state_inverts_joint_permutation_and_action_scaling(self):
        order = np.roll(np.arange(29), 7)
        default = np.linspace(-.2, .2, 29)
        scale = np.linspace(.1, .9, 29)
        q = np.linspace(-1., 1., 29)
        dq = np.linspace(-2., 2., 29)
        action = np.linspace(-.5, .5, 29)
        wire = {'body_q': q[order]+default, 'body_dq': dq[order],
                'last_action': action[order]*scale+default}
        restored = restore_policy_states([wire], default, scale, order)[0]
        for key, expected in [('body_q', q), ('body_dq', dq), ('last_action', action)]:
            np.testing.assert_allclose(restored[key], expected, atol=1e-15)

    def test_history_is_oldest_first_and_padding_matches_cpp_gravity(self):
        states = [{'base_ang_vel': [i+1., 0., 0.], 'body_q': [0.]*29,
                   'body_dq': [0.]*29, 'last_action': [0.]*29,
                   'base_quat': [1., 0., 0., 0.], 'token_state': [i*.0625]*64}
                  for i in range(3)]
        config = {'observations': [{'name': 'token_state'},
                  {'name': 'his_base_angular_velocity_3frame_step1'},
                  {'name': 'his_gravity_dir_3frame_step1'}]}
        obs, layout = reconstruct_observations(states, config)
        np.testing.assert_array_equal(obs[0, 64:73], [0, 0, 0, 0, 0, 0, 1, 0, 0])
        np.testing.assert_array_equal(obs[2, 64:73], [1, 0, 0, 2, 0, 0, 3, 0, 0])
        np.testing.assert_array_equal(obs[0, 73:], [0, 0, 1, 0, 0, 1, 0, 0, -1])
        self.assertEqual([x['offset'] for x in layout], [0, 64, 73])
        np.testing.assert_array_equal(gravity_cpp(np.array([1., 0., 0., 0.])), [0, 0, -1])

    def test_motion_invalid_layout_rejected(self):
        with self.assertRaises(ValueError):
            unpack_motion_output(np.zeros((2, 64)), np.arange(29))
        invalid = np.zeros((1, 640)); invalid[0, 0] = np.nan
        with self.assertRaises(ValueError):
            unpack_motion_output(invalid, np.arange(29))


if __name__ == '__main__':
    unittest.main()
