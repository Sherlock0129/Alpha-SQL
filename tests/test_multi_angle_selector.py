import unittest

import numpy as np

from alphasql.runner.multi_angle_selector import _fit, _rule, _tournament, _features, _valid, VIEWS


class MultiAngleSelectorTest(unittest.TestCase):
    def test_raw_pool_can_be_collected_before_gold_labels(self):
        row = {"rounds": [{"candidates": [{"execution": {"valid": True}},
                                           {"execution": {"valid": False}}]}]}
        self.assertEqual(_valid(row), [0])

    def test_rule_rejects_missing_forward_requirement(self):
        strong = np.array([.8, .9, .9, .9, .9, .9, .9, .9])
        missing = np.array([.9, .9, .9, .1, .9, .9, .9, .9])
        self.assertGreater(_rule(strong), _rule(missing))

    def test_reversed_pair_disagreement_is_neutral(self):
        neutral = {"pairs": {"0_1": [.9, .9]}}
        scores = _tournament(neutral, [0, 1], {0: .5, 1: .5})
        self.assertAlmostEqual(scores[0], scores[1])

    def test_pairwise_fit_favors_correct_candidate(self):
        def candidate(label, raw):
            return {"execution_rechecked_valid": True, "execution_label": label,
                    "candidate_raw": raw}
        row = {"rounds": [{"candidates": [candidate(1, .8), candidate(0, .7)]}]}
        evidence = {"views": {"0": {v: .9 for v in VIEWS},
                              "1": {v: .2 for v in VIEWS}}}
        weights = _fit([(row, evidence)])
        self.assertGreater(_features(row, evidence, 0) @ weights,
                           _features(row, evidence, 1) @ weights)


if __name__ == "__main__":
    unittest.main()
