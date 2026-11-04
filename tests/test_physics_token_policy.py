import unittest
import numpy as np
from sonic_physics_replay import PolicyHistory


class PhysicsPolicyHistoryTests(unittest.TestCase):
    def test_history_uses_new_states_and_previous_generated_actions(self):
        cfg={'observations':[{'name':'token_state'}, {'name':'his_body_joint_positions_3frame_step1'},
                             {'name':'his_last_actions_3frame_step1'}]}
        h=PolicyHistory(cfg)
        for i in range(5):
            obs=h.append(np.full(29,i),np.zeros(29),np.zeros(3),[1,0,0,0],np.full(29,-i),np.full(64,.25))
        np.testing.assert_array_equal(obs[0,:64],np.full(64,.25))
        np.testing.assert_array_equal(obs[0,64:151],np.repeat([2,3,4],29))
        np.testing.assert_array_equal(obs[0,151:],np.repeat([-2,-3,-4],29))
        changed=h.append(np.full(29,99),np.zeros(29),np.zeros(3),[1,0,0,0],np.ones(29),np.full(64,.25))
        self.assertTrue((changed[0,122:151]==99).all())

    def test_zero_history_padding_preserves_cpp_gravity_convention(self):
        h=PolicyHistory({'observations':[{'name':'token_state'},{'name':'his_gravity_dir_3frame_step1'}]})
        obs=h.append(np.zeros(29),np.zeros(29),np.zeros(3),[1,0,0,0],np.zeros(29),np.zeros(64))
        np.testing.assert_array_equal(obs[0,64:],[0,0,1,0,0,1,0,0,-1])


if __name__=='__main__':unittest.main()
