"""Tests for metric conventions and assignment behavior that affect reported quality."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from evaluation import evaluate
from group_policy import apply_policies, chosen_pairs


class MetricsTests(unittest.TestCase):
    def test_macro_includes_singletons_and_false_merges(self):
        truth = {'a': {'x', 'y'}, 'b': set(), 'c': set()}
        predictions = {'a': {'x', 'z'}, 'b': set(), 'c': {'w'}}
        result = evaluate(truth, predictions)
        self.assertAlmostEqual(result['macro_f05'], .5)
        self.assertAlmostEqual(result['link_precision'], 1 / 3)
        self.assertEqual(result['link_recall'], .5)
        self.assertEqual(result['singleton_accuracy'], .5)

    def test_no_predictions_and_empty_reference_sets(self):
        self.assertEqual(evaluate({'a': {'x'}}, {'a': set()})['macro_f05'], 0)
        result = evaluate({'a': set()}, {'a': set()})
        self.assertEqual(result['macro_f05'], 1)
        self.assertIsNone(result['link_precision'])
        self.assertIsNone(result['link_recall'])
        self.assertIsNone(evaluate({}, {})['macro_f05'])

    def test_missing_reference_is_an_error(self):
        with self.assertRaises(ValueError):
            evaluate({'a': set()}, {})

    def test_candidate_recall_and_country_slices(self):
        result = evaluate({'a': {'x', 'y'}, 'b': set()}, {'a': {'x'}, 'b': set()},
                          {'a': 'India', 'b': 'France'}, {'a': {'x'}, 'b': set()})
        self.assertEqual(result['candidate_recall'], .5)
        self.assertEqual(result['by_country']['France'], 1)
        self.assertAlmostEqual(result['by_country']['India'], 1.25 / 1.5)


class AssignmentTests(unittest.TestCase):
    def test_target_goes_to_highest_score_then_first_reference_on_tie(self):
        score = np.array([.8, .9, .8, .9])
        groups = np.array([0, 0, 1, 1])
        targets = np.array(['x', 'y', 'x', 'y'])
        result = chosen_pairs(score, .7, groups, 3, targets=targets)
        np.testing.assert_array_equal(result, [True, True, False, False])

    def test_rescue_only_fills_groups_without_threshold_matches(self):
        result = chosen_pairs(np.array([.9, .6, .6, .4]), .8,
                              np.array([0, 0, 1, 2]), 4, rescue=.5)
        np.testing.assert_array_equal(result, [True, False, True, False])

    def test_unknown_country_uses_global_policy(self):
        cfg = dict(threshold=.8, rescue_threshold=None,
                   country_policies={'india': dict(threshold=.6, rescue_threshold=None)})
        result = apply_policies(np.array([.7, .7]), np.array([0, 1]),
                                np.array(['india', 'unseen']), cfg)
        np.testing.assert_array_equal(result, [True, False])

    def test_empty_candidates(self):
        result = chosen_pairs(np.array([]), .7, np.array([], dtype=int), 2, rescue=.5)
        self.assertEqual(result.size, 0)


if __name__ == '__main__':
    unittest.main()
