import unittest

from alphasql.runner.multi_angle_selector import VIEWS
from alphasql.runner.seven_view_aggregate import _fit_simplex, _score


def candidate(index, label, raw, view_scores):
    return {"index": index, "label": label, "raw": raw,
            "views": dict(zip(VIEWS, view_scores))}


class SevenViewAggregateTest(unittest.TestCase):
    def test_scores_stay_bounded_and_single_noisy_view_is_not_a_veto(self):
        item = candidate(0, 1, .95, [.05, .9, .9, .9, .9, .9, .9])
        self.assertLess(_score(item, "old_min", {}), _score(item, "trimmed_mean", {}))
        for method in ("raw_jev", "old_min", "mean", "trimmed_mean", "median", "softmin_02"):
            self.assertLessEqual(_score(item, method, {}), 1.0)
            self.assertGreaterEqual(_score(item, method, {}), 0.0)

    def test_simplex_uses_predictive_views_without_negative_weights(self):
        good = candidate(0, 1, .5, [.9] * 7)
        bad = candidate(1, 0, .5, [.1] * 7)
        pools = [{"items": [good, bad]} for _ in range(5)]
        for family in ("group_simplex", "seven_simplex"):
            model = _fit_simplex(pools, family, 0.0)
            self.assertAlmostEqual(sum(model["weights"]), 1.0)
            self.assertTrue(all(weight >= 0 for weight in model["weights"]))
            self.assertGreater(_score(good, family, {family: model}),
                               _score(bad, family, {family: model}))


if __name__ == "__main__":
    unittest.main()
