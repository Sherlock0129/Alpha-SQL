import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from alphasql.algorithm.selection.content_verifier import ContentVerifier, content_features
from alphasql.algorithm.selection.dual_confidence import JevPoolJudge, PoolJudgement
from alphasql.algorithm.selection.selective_stop import StopPolicy, fit_stop_policy
from alphasql.algorithm.compiler.validation import CandidateExecution
from alphasql.database.database_manager import DatabaseManager
from alphasql.database.schema import ColumnSchema, DatabaseSchema, TableSchema
from alphasql.jev.models import Usage
from alphasql.jev.client import JevClient
from alphasql.jev.runtime import JevSettings
from alphasql.runner.new_selector import AdaptiveSQLSelector, SelectorSettings
from alphasql.runner.task import Task


def _row(db, qid, score, correct, pool):
    return {"db_id": db, "question_id": qid, "generator": {"version": "frozen"},
            "rounds": [{"jev": {"called": True, "model": "jev-test", "pool_covered": pool},
                        "candidates": [{"candidate_raw": score, "execution_label": correct,
                                        "execution_rechecked_valid": True}]}]}


class SelectiveStopTest(unittest.TestCase):
    def test_validation_curve_and_frozen_threshold(self):
        rows = [_row("a", 1, .95, 1, .9), _row("a", 2, .8, 0, .9),
                _row("b", 3, .7, 1, .2), _row("b", 4, .5, 0, .9)]
        policy, curve = fit_stop_policy(rows, score_key="candidate_raw",
                                        max_empirical_risk=0, pool_review_below=.8)
        self.assertEqual(policy.threshold, .875)
        self.assertEqual(policy.accepted_questions, 1)
        self.assertTrue(policy.accepts(.96, .9))
        self.assertFalse(policy.accepts(.96, .2))
        self.assertEqual(next(item for item in curve if item["threshold"] == .875)["errors"], 0)

    def test_low_pool_triggers_adjudication_without_regeneration(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "tmp") as dirname:
            root = Path(dirname)
            folder = root / "db"
            folder.mkdir()
            (folder / "db.sqlite").touch()
            DatabaseManager.CACHED_DATABASE_SCHEMA["db"] = DatabaseSchema(
                db_id="db", tables={"items": TableSchema(table_name="items", columns={
                    "name": ColumnSchema(original_column_name="name", column_type="TEXT")})})
            self.addCleanup(DatabaseManager.CACHED_DATABASE_SCHEMA.pop, "db", None)

            class Generator:
                calls = 0
                def generate(self, task, round_index, feedback=None):
                    self.calls += 1
                    return ["SELECT name FROM items"]

            class Judge:
                max_candidates = 8
                reviews = 0
                def assess(self, question, evidence, schema, candidates):
                    return PoolJudgement(.2, (.99,), Usage())
                def adjudicate(self, question, evidence, schema, candidates, scores):
                    self.reviews += 1
                    return {"recommended_index": 0, "scores": {"0": .8},
                            "usage": Usage(), "model": "jev-test", "called": True}

            generator, judge = Generator(), Judge()
            policy = StopPolicy("candidate_raw", .9, .1, 47, 20, ("val",))
            task = Task(question_id=1, db_id="db", question="Names", evidence="",
                        sql=None, table_schema_dict={})
            with patch("alphasql.runner.new_selector.validate_sql",
                       return_value=CandidateExecution(True, "success", ["name"], [["Ada"]])):
                result = AdaptiveSQLSelector(generator, judge,
                    SelectorSettings(max_rounds=3, max_candidates=8),
                    stop_policy=policy).run(task, str(root))
            self.assertEqual(generator.calls, 1)
            self.assertEqual(judge.reviews, 1)
            self.assertEqual(result["decision_action"], "adjudicate")
            self.assertFalse(result["accepted_stop"])
            self.assertTrue(result["uncertain"])
            self.assertEqual(result["jev_calls_total"], 2)

            capped_generator, capped_judge = Generator(), Judge()
            with patch("alphasql.runner.new_selector.validate_sql",
                       return_value=CandidateExecution(True, "success", ["name"], [["Ada"]])):
                capped = AdaptiveSQLSelector(capped_generator, capped_judge,
                    SelectorSettings(max_rounds=3, max_candidates=8, max_jev_calls=1),
                    stop_policy=policy).run(task, str(root))
            self.assertEqual(capped["decision_action"], "budget_exhausted")
            self.assertEqual(capped_judge.reviews, 0)

    def test_content_and_pairwise_training(self):
        features = content_features("How many items?", "", {"tables": {}},
                                    "SELECT COUNT(*) FROM items",
                                    {"valid": True, "row_count": 1,
                                     "columns": ["COUNT(*)"], "sample_rows": [[2]]})
        self.assertIn("hint:count:1:1", features)
        records = [
            {"question_id": "a", "label": 1, "near_miss": False,
             "features": {"bias": 1, "count_match": 1}},
            {"question_id": "a", "label": 0, "near_miss": True,
             "features": {"bias": 1, "count_mismatch": 1}},
            {"question_id": "b", "label": 1, "near_miss": False,
             "features": {"bias": 1, "count_match": 1}},
            {"question_id": "b", "label": 0, "near_miss": False,
             "features": {"bias": 1, "count_mismatch": 1}},
        ]
        model, report = ContentVerifier.fit(records)
        scores = model.predict(records)
        self.assertGreater(scores[0], scores[1])
        self.assertEqual(report["pairs"], 2)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "tmp") as dirname:
            path = Path(dirname) / "model.json"
            model.save(path)
            self.assertAlmostEqual(ContentVerifier.load(path).predict(records)[0], scores[0])

    def test_adjudication_only_reads_existing_candidates(self):
        observed = {}
        def transport(state, questions):
            observed.update(state=state, questions=questions)
            return {"answers": {"verify_0": {"noul": .8},
                                "verify_1": {"noul": .2},
                                "prefer_0_1": {"noul": .9}}}
        judge = JevPoolJudge(JevClient(JevSettings(), transport))
        candidates = [{"sql": "SELECT 1", "execution": {"valid": True}},
                      {"sql": "SELECT 2", "execution": {"valid": True}}]
        result = judge.adjudicate("one", "", {"tables": {}}, candidates, [.9, .8])
        self.assertEqual(result["recommended_index"], 0)
        self.assertEqual(len(observed["state"]["candidates"]), 2)
        self.assertNotIn("gold", str(observed))


if __name__ == "__main__":
    unittest.main()
