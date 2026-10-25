import json
import sqlite3
import tempfile
import unittest
from argparse import Namespace
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

from alphasql.algorithm.selection.dual_calibration import (
    DualCalibrator, LogisticHead, curriculum_fraction, fit_dual,
)
from alphasql.algorithm.selection.dual_confidence import JevPoolJudge, PoolJudgement
from alphasql.database.database_manager import DatabaseManager
from alphasql.database.schema import ColumnSchema, DatabaseSchema, TableSchema
from alphasql.jev.client import JevClient
from alphasql.jev.models import Usage
from alphasql.jev.runtime import JevSettings
from alphasql.runner.new_selector import (
    AdaptiveSQLSelector, SelectorSettings, _split, _train, label_result,
)
from alphasql.runner.task import Task


class ScriptedGenerator:
    def __init__(self, rounds):
        self.rounds = rounds
        self.calls = []

    def generate(self, task, round_index):
        assert task.sql is None
        self.calls.append(round_index)
        return self.rounds[round_index]


class ScriptedJudge:
    max_candidates = 8

    def __init__(self, coverages, scores):
        self.coverages, self.scores = coverages, scores
        self.calls = []

    def assess(self, question, evidence, schema, candidates):
        self.calls.append(tuple(item["sql"] for item in candidates))
        index = len(self.calls) - 1
        return PoolJudgement(self.coverages[index], tuple(self.scores[index]), Usage())


class NewSelectorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "tmp")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        db_dir = self.root / "testdb"
        db_dir.mkdir()
        self.db_path = db_dir / "testdb.sqlite"
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute("CREATE TABLE items (name TEXT)")
            connection.executemany("INSERT INTO items VALUES (?)", [("Ada",), ("Grace",)])
            connection.commit()
        DatabaseManager.CACHED_DATABASE_SCHEMA["testdb"] = DatabaseSchema(
            db_id="testdb", tables={"items": TableSchema(table_name="items", columns={
                "name": ColumnSchema(original_column_name="name", column_type="TEXT")})})
        self.addCleanup(DatabaseManager.CACHED_DATABASE_SCHEMA.pop, "testdb", None)
        self.task = Task(question_id=1, db_id="testdb", question="Name Ada",
                         evidence="", sql="SELECT secret_gold", table_schema_dict={})

    def test_low_coverage_expands_then_selects_from_valid_candidates(self):
        generator = ScriptedGenerator([
            ["SELECT name FROM items WHERE name = 'Grace'"],
            ["SELECT name FROM items WHERE name = 'Ada'"],
        ])
        judge = ScriptedJudge([0.2, 0.95], [[0.3], [0.1, 0.92]])
        result = AdaptiveSQLSelector(generator, judge,
            SelectorSettings(max_rounds=2, max_candidates=8)).run(self.task, str(self.root))
        self.assertEqual(generator.calls, [0, 1])
        self.assertEqual(result["selected_index"], 1)
        self.assertEqual(len(result["rounds"]), 2)
        self.assertFalse(result["uncertain"])
        self.assertEqual(result["confidence_kind"], "raw_jev_judgement")

    def test_invalid_and_empty_sql_never_get_nonzero_probability(self):
        generator = ScriptedGenerator([["SELECT missing FROM items"]])
        calls = []

        def transport(state, questions):
            calls.append(state)
            raise AssertionError("No Jev request expected for an invalid-only pool")

        judge = JevPoolJudge(JevClient(JevSettings(), transport), max_candidates=8)
        result = AdaptiveSQLSelector(generator, judge,
            SelectorSettings(max_rounds=1, max_candidates=8),
            DualCalibrator(LogisticHead(3, 1), LogisticHead(3, 1))).run(
                self.task, str(self.root))
        self.assertIsNone(result["selected_index"])
        self.assertEqual(result["coverage_probability"], 0.0)
        self.assertEqual(result["candidates"][0]["candidate_probability"], 0.0)
        self.assertFalse(calls)

    def test_jev_questions_are_separate_and_gold_is_not_in_state(self):
        captured = {}

        def transport(state, questions):
            captured["state"] = state
            captured["questions"] = questions
            return {"answers": {
                "pool_covered": {"noul": 0.6},
                "candidate_0": {"noul": 0.2},
                "candidate_1": {"noul": 0.8},
            }}

        judge = JevPoolJudge(JevClient(JevSettings(), transport))
        pool = [{"sql": "SELECT 1", "execution": {"valid": True}},
                {"sql": "SELECT 2", "execution": {"valid": True}}]
        answer = judge.assess("question", "hint", {"tables": {}}, pool)
        self.assertEqual(answer.coverage_raw, 0.6)
        self.assertEqual(answer.candidate_raw, (0.2, 0.8))
        self.assertEqual(set(captured["questions"]),
                         {"pool_covered", "candidate_0", "candidate_1"})
        self.assertNotIn("gold", json.dumps(captured["state"]))

    def test_offline_labels_use_execution_multiset_and_do_not_store_sql(self):
        result = {"question_id": 1, "db_id": "testdb", "coverage_raw": 0.7,
                  "candidates": [
                      {"sql": "SELECT name FROM items WHERE name = 'Ada'", "candidate_raw": 0.8},
                      {"sql": "SELECT name FROM items WHERE name = 'Grace'", "candidate_raw": 0.4},
                  ]}
        labelled = label_result(result, "SELECT name FROM items WHERE name = 'Ada'",
                                str(self.db_path))
        self.assertEqual(labelled["covered_label"], 1)
        self.assertEqual([item["label"] for item in labelled["candidates"]], [1, 0])
        self.assertNotIn("SELECT", json.dumps(labelled))

    def test_empty_result_is_valid_for_offline_equivalence(self):
        result = {"question_id": 1, "db_id": "testdb", "coverage_raw": 0.5,
                  "candidates": [{"sql": "SELECT name FROM items WHERE name = 'Nobody'",
                                  "candidate_raw": 0.4}]}
        labelled = label_result(result,
            "SELECT name FROM items WHERE name = 'Nobody'", str(self.db_path))
        self.assertEqual(labelled["candidates"][0],
                         {"valid": True, "raw": 0.4, "label": 1})

    def test_curriculum_and_split_boundary(self):
        self.assertLess(curriculum_fraction(0, 10), 1)
        self.assertEqual(curriculum_fraction(9, 10), 1)
        train = [{"question_id": "1", "db_id": "db_train", "coverage_raw": 0.9,
                  "covered_label": 1,
                  "candidates": [{"valid": True, "raw": 0.9, "label": 1}]}]
        validation = [{"question_id": "2", "db_id": "db_val", "coverage_raw": 0.1,
                       "covered_label": 0,
                       "candidates": [{"valid": True, "raw": 0.1, "label": 0}]}]
        model, report = fit_dual(train, validation, epochs=3)
        self.assertTrue(0 <= model.coverage.predict(0.5) <= 1)
        self.assertIn("validation_brier", report["candidate"])
        with self.assertRaisesRegex(ValueError, "disjoint by database"):
            fit_dual(train, [dict(validation[0], db_id="db_train")], epochs=3)

    def test_split_and_train_commands_write_reusable_model(self):
        rows = [
            {"question_id": str(i), "db_id": f"db_{i}",
             "coverage_raw": 0.8 if i % 2 else 0.2,
             "covered_label": i % 2,
             "candidates": [{"valid": True, "raw": 0.8 if i % 2 else 0.2,
                             "label": i % 2}]}
            for i in range(6)
        ]
        source = self.root / "labels.jsonl"
        source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        split_dir = self.root / "split"
        _split(Namespace(labels=str(source), output_dir=str(split_dir),
                         validation_fraction=0.34, seed=42))
        model_dir = self.root / "model"
        _train(Namespace(train_labels=str(split_dir / "train_labels.jsonl"),
                         validation_labels=str(split_dir / "validation_labels.jsonl"),
                         output_dir=str(model_dir), epochs=3, learning_rate=0.03))
        model = DualCalibrator.load(model_dir / "dual_calibrator.json")
        self.assertTrue(0 < model.coverage.predict(0.5) < 1)
        self.assertTrue((model_dir / "validation_report.json").is_file())


if __name__ == "__main__":
    unittest.main()
