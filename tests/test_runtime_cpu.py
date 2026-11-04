import unittest
from runtime_cpu import fast_cpu_set


class RuntimeCpuTests(unittest.TestCase):
    def test_hybrid_selects_performance_tier_within_existing_affinity(self):
        self.assertEqual(fast_cpu_set({0, 1, 2, 3}, {0: 5500, 1: 5400, 2: 4700, 3: 4700}), {0, 1})
        self.assertEqual(fast_cpu_set({2, 3}, {0: 5500, 1: 5400, 2: 4700, 3: 4700}), {2, 3})

    def test_homogeneous_missing_data_and_single_fast_core_stay_unchanged(self):
        self.assertEqual(fast_cpu_set({0, 1}, {0: 5500, 1: 5400}), {0, 1})
        self.assertEqual(fast_cpu_set({0, 1}, {0: 5500}), {0, 1})
        self.assertEqual(fast_cpu_set({0, 1}, {0: 5500, 1: 4700}), {0, 1})


if __name__ == '__main__':
    unittest.main()
