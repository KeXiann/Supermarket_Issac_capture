"""Check CPU bridge IMU rotations against independent rotation composition."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from bridge_core import inverse_quaternion_matrix


class RotationTests(unittest.TestCase):
    def test_random_imu_vectors_match_independent_rotation_and_roundtrip(self):
        random = np.random.default_rng(20261101)
        for _ in range(250):
            rotation = Rotation.random(random_state=random)
            xyzw = rotation.as_quat()
            q = xyzw[[3, 0, 1, 2]]
            world = random.normal(size=(4, 3)) * 20.
            inverse = inverse_quaternion_matrix(q)
            np.testing.assert_allclose(world @ inverse.T, rotation.inv().apply(world), atol=1e-12)
            np.testing.assert_allclose(inverse @ inverse.T, np.eye(3), atol=2e-15)

    def test_stationary_yawed_torso_keeps_gravity_up_and_correct_gyro(self):
        rotation = Rotation.from_euler('z', 90, degrees=True)
        q = rotation.as_quat()[[3, 0, 1, 2]]
        inverse = inverse_quaternion_matrix(q)
        np.testing.assert_allclose(inverse @ [0, 0, 9.81], [0, 0, 9.81], atol=1e-12)
        np.testing.assert_allclose(inverse @ [1, 0, 0], [0, -1, 0], atol=1e-12)
