from project_paths import SONIC_ROOT
import unittest
import json
from pathlib import Path

import yaml

from sonic_contract import (CPP_HEADER, SONIC_CPP_MOTOR_ORDER,
                            policy_parameters, reorder_yaml_values, validate_cpp_header,
                            yaml_names_by_index)


CFG = SONIC_ROOT / "gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml"


class SonicContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with CFG.open(encoding="utf-8") as stream:
            cls.config = yaml.safe_load(stream)

    def test_cpp_header_and_order_are_explicit(self):
        validate_cpp_header(CPP_HEADER)
        self.assertEqual(len(SONIC_CPP_MOTOR_ORDER), 29)
        self.assertEqual(len(set(SONIC_CPP_MOTOR_ORDER)), 29)

    def test_yaml_named_arrays_reorder_pitch_before_yaw(self):
        yaml_names = yaml_names_by_index(self.config)
        self.assertEqual(yaml_names[:3], (
            "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint"))
        values = list(range(29))
        converted = reorder_yaml_values(values, self.config)
        self.assertEqual(converted[:3], (2, 1, 0))
        self.assertEqual(converted[6:9], (8, 7, 6))

    def test_current_model_joint_order_matches_cpp_contract(self):
        from upstream_contract import model_contract
        self.assertEqual(tuple(model_contract()['joint_order']), SONIC_CPP_MOTOR_ORDER)

    def test_nominal_diagnostic_gains_are_read_from_cpp_source(self):
        parameters = policy_parameters()
        for values in parameters.values():
            self.assertEqual(len(values), 29)
        self.assertAlmostEqual(parameters["default_angles"][4], -.363)
        self.assertAlmostEqual(parameters["kps"][4], 2 * parameters["kps"][15])
        self.assertGreater(parameters["kds"][0], parameters["kds"][15])


if __name__ == "__main__":
    unittest.main()
