import unittest
from types import SimpleNamespace

from alphasql.runner.llm_jev_phase1 import (
    _self_consistency,
    is_supported_slice,
    prepare_split,
)


class PhaseOneComparisonTest(unittest.TestCase):
    def test_capability_filter_accepts_only_current_vertical_slice(self):
        self.assertTrue(is_supported_slice("SELECT name FROM people WHERE city = 'Paris'"))
        self.assertTrue(is_supported_slice("SELECT COUNT(id) FROM people"))
        self.assertFalse(is_supported_slice(
            "SELECT name FROM people WHERE city = 'Paris' AND age > 20"
        ))
        self.assertFalse(is_supported_slice("SELECT name FROM people ORDER BY name"))
        self.assertFalse(is_supported_slice("SELECT name, age FROM people"))

    def test_database_split_is_disjoint(self):
        tasks = [
            SimpleNamespace(question_id=1, db_id="restaurant", sql="SELECT city FROM geo"),
            SimpleNamespace(question_id=2, db_id="other", sql="SELECT name FROM people"),
        ]
        split = prepare_split(tasks)
        self.assertEqual(split["calibration_ids"], [1])
        self.assertEqual(split["evaluation_ids"], [2])
        self.assertTrue(split["database_disjoint"])

    def test_self_consistency_prefers_nonempty_majority_and_allows_empty_fallback(self):
        a, b, empty = (("a",),), (("b",),), ()
        self.assertEqual(_self_consistency([a, b, a, empty], [1, 1, 1, 0]), 0)
        self.assertEqual(_self_consistency([empty, empty], [0, 0]), 0)
        self.assertIsNone(_self_consistency([None], [0]))


if __name__ == "__main__":
    unittest.main()
