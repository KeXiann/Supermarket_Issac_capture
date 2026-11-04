"""Saving tolerates short gaps; fresh human labels remain strictly masked."""
import unittest

import numpy as np

from capture_utils import (HUMAN_REFERENCE_MAX_AGE_S, MAX_HUMAN_PACKET_GAP_S,
                           human_packet_gap_summary, human_reference_valid)
from prepare_trajectory import segment_ids


class HumanFreshnessTests(unittest.TestCase):
    def test_capture_tolerance_does_not_relax_per_row_freshness(self):
        self.assertEqual(MAX_HUMAN_PACKET_GAP_S, 2.)
        self.assertEqual(HUMAN_REFERENCE_MAX_AGE_S, .1)
        rows = np.array([0, 0, 0, 0, 1, -1, 1, 1])
        ages = np.array([0., .08, .10, .12, .0, .0, -.01, .02])
        modes = np.array([2, 2, 2, 2, 2, 2, 2, 0])
        valid = human_reference_valid(rows, ages, modes)
        np.testing.assert_array_equal(valid, [True, True, True, False, True, False, False, False])
        # A saved 160ms human gap must not join stale-labelled trajectory rows
        # into the same training/replay segment.
        np.testing.assert_array_equal(segment_ids(valid, np.arange(8) * .02, .02),
                                      [0, 0, 0, -1, 1, -1, -1, -1])

    def test_missing_or_nonfinite_reference_cannot_be_fresh(self):
        valid = human_reference_valid(np.array([-1, 0, 0]),
                                      np.array([0., np.nan, np.inf]), np.full(3, 2))
        self.assertFalse(valid.any())

    def test_observed_one_second_stall_is_saved_but_stale_rows_split(self):
        packets = np.array([0., 1.0254157960880548, 1.0454157960880548])
        report = human_packet_gap_summary(packets)
        self.assertTrue(report['capture_gap_gate_passed'])
        self.assertEqual(report['packet_gaps_over_100ms'], 1)
        self.assertGreater(report['gap_intervals'][0]['uncovered_fresh_reference_s'], .9)
        clock = np.arange(0, 1.08, .02)
        rows = np.searchsorted(packets, clock, side='right') - 1
        ages = clock - packets[rows]
        valid = human_reference_valid(rows, ages, np.full(len(clock), 2))
        self.assertFalse(valid[(clock > .1) & (clock < packets[1])].any())
        segments = segment_ids(valid, clock, .02)
        self.assertGreater(segments[-1], segments[0])
        self.assertTrue(np.all(segments[~valid] == -1))

    def test_long_interruptions_and_invalid_clocks_still_fail(self):
        self.assertFalse(human_packet_gap_summary([0., 2.01])['capture_gap_gate_passed'])
        self.assertTrue(human_packet_gap_summary([0., 2.])['capture_gap_gate_passed'])
        for times in ([0.], [0., 0.], [1., 0.], [0., float('nan')]):
            with self.subTest(times=times), self.assertRaises(ValueError):
                human_packet_gap_summary(times)


if __name__ == "__main__":
    unittest.main()
