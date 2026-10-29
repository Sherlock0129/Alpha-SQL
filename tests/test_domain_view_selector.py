import unittest
from pathlib import Path
from unittest.mock import patch

from alphasql.jev.models import ChoiceQuestion, ChoiceResult, JevResponse, NoulQuestion, NoulResult, Usage
from alphasql.runner.domain_view_selector import (
    CHECKS, _collect_one, _decisions, _self_consistency,
)


def _row():
    return {"db_id": "fixture", "question_id": 7,
            "question": "List patient names with cystitis, ordered by name",
            "evidence": "", "full_schema": {"tables": {"patients": {}}},
            "rounds": [{"candidates": [
                {"sql": "SELECT name FROM patients WHERE disease = 'cystitis' ORDER BY name",
                 "candidate_raw": .65, "execution": {"valid": True, "row_count": 2}},
                {"sql": "SELECT name FROM patients WHERE disease = 'cystitis'",
                 "candidate_raw": .60, "execution": {"valid": True, "row_count": 2}},
            ]}]}


def _execution():
    return {"candidates": [{"route": "direct", "result_hash": "a"},
                           {"route": "linked", "result_hash": "b"}],
            "clusters": [{"indices": [0], "result_hash": "a"},
                         {"indices": [1], "result_hash": "b"}]}


class FakeJev:
    def __init__(self):
        self.calls = []

    def ask(self, state, questions):
        self.calls.append((state, questions))
        answers = {}
        for key, question in questions.items():
            if isinstance(question, NoulQuestion):
                value = .2 if key == "order_set_missing_1" else .8
                answers[key] = NoulResult(value, {"true": value}, .6)
            elif isinstance(question, ChoiceQuestion):
                choice = next(k for k in question.criteria if k not in {"none", "missing"})
                answers[key] = ChoiceResult(choice, {choice: 1.0}, 1.0)
        return JevResponse(answers, Usage(10, 3), "fixture-model")


class DomainViewSelectorTest(unittest.TestCase):
    def test_eight_directed_checks_plus_overall_and_typed_locations(self):
        self.assertEqual(len(CHECKS), 9)
        row = _row()
        row["gold_sql"] = "SHOULD NEVER ENTER STATE"
        fake = FakeJev()
        with patch("alphasql.runner.domain_view_selector._save"):
            result = _collect_one(fake, row, Path("unused_fixture_pool.json"),
                                  _execution(), pairwise=True)
        self.assertEqual(len(fake.calls), 3)
        self.assertEqual(len(fake.calls[0][1]), 18)
        self.assertEqual(result["locations"]["1"]["order_set"]["sql_clause_id"][:1], "s")
        self.assertIn("question_fragment", result["locations"]["1"]["order_set"])
        self.assertTrue(result["pairwise_stage_done"])
        self.assertEqual(result["usage"]["checks"]["input_tokens"], 10)
        self.assertNotIn("gold_sql", str(fake.calls))
        choices = _decisions(row, result, _execution())
        self.assertEqual(choices["raw_jev"], 0)
        self.assertEqual(choices["self_consistency"], 0)

    def test_self_consistency_counts_results_and_deduplicates_routes(self):
        execution = {"candidates": [{"route": "same"}, {"route": "same"},
                                    {"route": "other"}, {"route": "third"}],
                     "clusters": [{"indices": [0, 1], "result_hash": "a"},
                                  {"indices": [2, 3], "result_hash": "b"}]}
        raw = {0: .8, 1: .7, 2: .6, 3: .5}
        self.assertEqual(_self_consistency([0, 1, 2, 3], execution, raw), 0)
        self.assertEqual(_self_consistency([0, 1, 2, 3], execution, raw, True), 2)


if __name__ == "__main__":
    unittest.main()
