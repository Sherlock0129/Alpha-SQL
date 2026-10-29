import unittest

from alphasql.runner.multi_angle_selector import VIEWS
from alphasql.runner.group_game_aggregate import _candidate_scores, _groups, select


def item(index, raw, forward, reverse, critical):
    values = [forward] * 3 + [reverse] * 3 + [critical]
    return {"index": index, "raw": raw,
            "views": dict(zip(VIEWS, values)), "sql": f"SELECT {index}"}


class GroupGameAggregateTest(unittest.TestCase):
    def test_copeland_uses_group_majority_and_raw_only_for_final_ties(self):
        first = item(0, .4, .9, .9, .1)
        second = item(1, .8, .8, .8, .9)
        scores = _candidate_scores([first, second], "group_copeland")
        self.assertEqual(scores, {0: 1.0, 1: 0.0})
        self.assertEqual(select({"items": [first, second]}, "group_copeland")[0]["index"], 0)
        tied_first = item(0, .4, .5, .5, .5)
        tied_second = item(1, .8, .5, .5, .5)
        self.assertEqual(select({"items": [tied_first, tied_second]},
                                "group_copeland")[0]["index"], 1)

    def test_borda_uses_group_ranks_and_half_credit_for_ties(self):
        first = item(0, .4, .9, .9, .1)
        second = item(1, .8, .8, .8, .9)
        scores = _candidate_scores([first, second], "group_borda")
        self.assertAlmostEqual(scores[0], 2 / 3)
        self.assertAlmostEqual(scores[1], 1 / 3)
        tied_first = item(0, .4, .5, .5, .5)
        tied_second = item(1, .8, .5, .5, .5)
        self.assertEqual(_candidate_scores([tied_first, tied_second], "group_borda"),
                         {0: .5, 1: .5})
        self.assertEqual(select({"items": [tied_first, tied_second]},
                                "group_borda")[0]["index"], 1)

    def test_entropic_minimax_prefers_balanced_groups_at_equal_raw(self):
        uneven = item(0, .5, .9, .9, .1)
        balanced = item(1, .5, .6, .6, .6)
        self.assertEqual(_groups(uneven).tolist(), [.9, .9, .1])
        chosen, score = select({"items": [uneven, balanced]}, "group_entropic_minimax")
        self.assertEqual(chosen["index"], 1)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_empty_pool_has_no_selection(self):
        self.assertEqual(select({"items": []}, "group_copeland"), (None, None))


if __name__ == "__main__":
    unittest.main()
