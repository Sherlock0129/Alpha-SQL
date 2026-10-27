import json
import sqlite3
import tempfile
import unittest
from argparse import Namespace
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from alphasql.algorithm.selection.dual_calibration import curriculum_order
from alphasql.algorithm.selection.training_data import (
    ACTION_NAMES, generator_identity, label_action_arms, label_trace,
    make_action_manifest, pool_id,
    split_by_database,
)
from alphasql.runner.selector_action_probe import _adjudicate, _feedback, probe_action
from alphasql.runner.selector_data import _audit, _label, _prepare_actions
from alphasql.jev.models import Usage
from alphasql.jev.runtime import JevSettings


class SelectorTrainingDataTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "tmp")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        db_dir = self.root / "demo"
        db_dir.mkdir()
        self.db_path = db_dir / "demo.sqlite"
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("CREATE TABLE items (name TEXT)")
            conn.executemany("INSERT INTO items VALUES (?)", [("Ada",), ("Grace",)])
            conn.commit()
        self.gold = "SELECT name FROM items WHERE name = 'Ada'"
        self.wrong = "SELECT name FROM items WHERE name = 'Grace'"
        self.raw = self._raw_trace()

    def _raw_trace(self):
        candidate = {"sql": self.wrong, "canonical_sql": self.wrong,
                     "source_round": 0, "route": "skeleton_linked",
                     "selected_tables": ["items"], "selected_columns": ["items.name"],
                     "execution": {"valid": True, "status": "success", "row_count": 1},
                     "candidate_raw": 0.7, "candidate_probability": 0.7}
        first = {"round": 0, "new_candidates": 1, "pool_size": 1,
                 "new_candidate_indices": [0], "question": "Name Ada", "evidence": "",
                 "schema_hypotheses": [{"candidate_index": 0, "tables": ["items"],
                                        "columns": ["items.name"], "scope": "linked"}],
                 "candidates": [candidate], "generation_action": "initial",
                 "jev": {"called": True, "pool_covered": 0.2,
                         "candidate": [0.7], "gap_schema": 0.1,
                         "gap_projection": 0.1, "gap_filters": 0.8,
                         "gap_logic": 0.1,
                         "usage": {"input_tokens": 10, "output_tokens": 1}}}
        second = json.loads(json.dumps(first))
        second.update(round=1, new_candidates=1, pool_size=2,
                      new_candidate_indices=[1], generation_action="revise")
        second["candidates"].append(dict(candidate, sql=self.gold,
                                         canonical_sql=self.gold, source_round=1,
                                         route="revise", candidate_raw=0.9,
                                         candidate_probability=0.9))
        second["jev"].update(pool_covered=0.95, candidate=[0.1, 0.9])
        return {"schema_version": 2, "question_id": 1, "db_id": "demo",
                "question": "Name Ada", "evidence": "", "full_schema": {
                    "db_id": "demo", "tables": {"items": {"columns": {"name": {"type": "TEXT"}}}}},
                "generator": {"kind": "Scripted", "version": "frozen-v1"},
                "rounds": [first, second]}

    def test_rich_trace_labels_each_round_without_gold_sql(self):
        row = label_trace(self.raw, self.gold, str(self.db_path), near_miss_max_edits=10)
        self.assertEqual(row["rounds"][0]["covered_label"], 0)
        self.assertEqual(row["rounds"][0]["difficulty"]["stratum"], "generation_hard")
        self.assertEqual(row["rounds"][1]["covered_label"], 1)
        self.assertEqual(row["rounds"][1]["selected_correct_label"], 1)
        self.assertEqual(row["rounds"][1]["observed_action_outcome"]["delta_coverage"], 1)
        self.assertTrue(row["rounds"][0]["candidates"][0]["near_miss"])
        self.assertTrue(row["rounds"][0]["diagnosis_target"]["labels"]["filters"])
        self.assertNotIn("gold_sql", json.dumps(row))
        self.assertEqual(row["initial_pool_id"], row["rounds"][0]["pool_id"])

    def test_split_manifest_and_action_inputs_preserve_frozen_pool(self):
        labelled = label_trace(self.raw, self.gold, str(self.db_path))
        manifest = make_action_manifest([labelled], {"1": self.raw}, 1, 42,
                                        {"max_new_candidates": 1, "max_model_calls": 2})
        self.assertEqual(len(manifest), 1)
        self.assertEqual(set(manifest[0]["actions"]), set(ACTION_NAMES))
        self.assertNotIn("execution_label", json.dumps(manifest))
        self.assertNotIn("sampling_stratum", json.dumps(manifest))
        self.assertNotIn("covered_label", json.dumps(manifest))
        self.assertEqual(manifest[0]["initial_pool_id"], pool_id(
            "demo", 1, self.raw["rounds"][0]["candidates"],
            generator_identity(self.raw["generator"])))
        rows = [dict(labelled, question_id=i, db_id=f"db_{i}") for i in range(6)]
        parts = split_by_database(rows, 0.2, 0.2, 42)
        self.assertEqual(sum(map(len, parts.values())), 6)
        self.assertTrue(set(item["db_id"] for item in parts["train"]).isdisjoint(
            item["db_id"] for item in parts["validation"]))
        self.assertTrue(set(item["db_id"] for item in parts["test"]).isdisjoint(
            item["db_id"] for item in parts["train"]))

    def test_action_labels_require_same_state_and_budget(self):
        labelled = label_trace(self.raw, self.gold, str(self.db_path))
        state_id, initial_id = labelled["rounds"][0]["state_id"], labelled["initial_pool_id"]
        budget = {"max_new_candidates": 1, "max_model_calls": 2}
        arms = []
        for action, sql, tokens in (("relink", self.gold, 100),
                                     ("revise", self.gold, 40),
                                     ("diversify", self.wrong, 80),
                                     ("adjudicate", self.wrong, 20)):
            arms.append({"state_id": state_id, "initial_pool_id": initial_id,
                         "question_id": 1, "db_id": "demo", "action": action,
                         "budget": budget, "new_candidates": [] if action == "adjudicate"
                         else [sql], "selected_sql": sql,
                         "usage": {"model_calls": 1, "input_tokens": tokens,
                                   "output_tokens": 0}})
        labels = label_action_arms(arms, [labelled], {"1": self.gold}, str(self.root))
        self.assertEqual(labels[0]["best_observed_action"], "revise")
        self.assertTrue(labels[0]["complete"])
        self.assertEqual([row["delta_selected_correct"] for row in labels[0]["arms"]],
                         [1, 1, 0, 0])
        arms[0] = dict(arms[0], initial_pool_id="wrong")
        with self.assertRaisesRegex(ValueError, "frozen pool"):
            label_action_arms(arms, [labelled], {"1": self.gold}, str(self.root))
        arms[0] = dict(arms[0], initial_pool_id=initial_id,
                       usage=dict(arms[0]["usage"], tokens_complete=False))
        incomplete = label_action_arms(arms, [labelled], {"1": self.gold},
                                       str(self.root))[0]
        self.assertFalse(incomplete["cost_complete"])
        self.assertIsNone(incomplete["best_observed_action"])

    def test_curriculum_distinguishes_coverage_and_selection(self):
        rows = [
            {"question_id": "covered_easy", "coverage_raw": 0.9, "covered_label": 1,
             "selection_difficulty": 0.1,
             "candidates": [{"valid": True, "raw": 0.9, "label": 1}]},
            {"question_id": "covered_hard", "coverage_raw": 0.8, "covered_label": 1,
             "selection_difficulty": 0.9,
             "candidates": [{"valid": True, "raw": 0.1, "label": 1}]},
            {"question_id": "uncovered", "coverage_raw": 0.1, "covered_label": 0,
             "selection_difficulty": None,
             "candidates": [{"valid": True, "raw": 0.1, "label": 0}]},
        ]
        self.assertEqual([row["question_id"] for row in curriculum_order(rows, "candidate")],
                         ["covered_easy", "covered_hard", "uncovered"])
        coverage = curriculum_order(rows, "coverage")
        self.assertEqual({row["covered_label"] for row in coverage[:2]}, {0, 1})

    def test_action_probe_uses_frozen_diagnosis_and_pairwise_only(self):
        labelled = label_trace(self.raw, self.gold, str(self.db_path))
        state = make_action_manifest([labelled], {"1": self.raw}, 1, 42,
                                     {"max_new_candidates": 1, "max_model_calls": 2})[0]
        self.assertEqual(_feedback(state, "revise").dimension, "filters")
        self.assertEqual(_feedback(state, "relink").dimension, "schema")

        second = dict(state["initial_candidates"][0], sql=self.gold,
                      candidate_probability=0.1)
        state["initial_candidates"].append(second)

        class FakeClient:
            def ask(self, prompt_state, questions):
                self.state = prompt_state
                self.questions = questions
                return self

            def require(self, name, kind):
                return type("Answer", (), {"noul": 0.1})()

            usage = Usage(10, 1)

        client = FakeClient()
        selected, usage = _adjudicate(state, client)
        self.assertEqual(selected, self.gold)
        self.assertEqual(usage.input_tokens, 10)
        self.assertEqual(set(client.questions), {"prefer_0_1"})

    def test_offline_cli_builds_rich_labels_action_manifest_and_audit(self):
        results = self.root / "results"
        results.mkdir()
        (results / "1.json").write_text(json.dumps(self.raw), encoding="utf-8")
        data = self.root / "gold.json"
        data.write_text(json.dumps([{"question_id": 1, "db_id": "demo",
                                     "SQL": self.gold}]), encoding="utf-8")
        rich = self.root / "rich.jsonl"
        calibration = self.root / "calibration.jsonl"
        _label(Namespace(results_dir=str(results), data_path=str(data),
                         database_root_dir=str(self.root), output_path=str(rich),
                         calibration_output=str(calibration), execution_timeout=10,
                         near_miss_max_edits=10))
        self.assertEqual(len(rich.read_text(encoding="utf-8").splitlines()), 1)
        self.assertNotIn(self.gold, calibration.read_text(encoding="utf-8"))
        manifest = self.root / "actions.jsonl"
        _prepare_actions(Namespace(labels=str(rich), results_dir=str(results),
                                   output_path=str(manifest), per_stratum=1, seed=42,
                                   max_new_candidates=1, max_model_calls=2,
                                   max_tokens=None))
        self.assertNotIn("execution_label", manifest.read_text(encoding="utf-8"))
        audit = self.root / "audit.jsonl"
        _audit(Namespace(labels=str(rich), data_path=str(data),
                         output_path=str(audit), per_group=1, seed=42))
        self.assertIn("gold_sql", audit.read_text(encoding="utf-8"))

    def test_invalid_reference_sql_is_reported_without_labelling_the_trace(self):
        results = self.root / "results"
        results.mkdir()
        (results / "1.json").write_text(json.dumps(self.raw), encoding="utf-8")
        data = self.root / "gold.json"
        data.write_text(json.dumps([{"question_id": 1, "db_id": "demo",
                                     "SQL": "SELECT missing FROM items"}]), encoding="utf-8")
        rich = self.root / "rich.jsonl"
        skipped = self.root / "skipped.jsonl"
        args = Namespace(results_dir=str(results), data_path=str(data),
                         database_root_dir=str(self.root), output_path=str(rich),
                         calibration_output=None, execution_timeout=10,
                         near_miss_max_edits=10, skip_invalid_gold=False,
                         skipped_output=str(skipped))
        with self.assertRaisesRegex(ValueError, "Gold SQL did not execute"):
            _label(args)
        args.skip_invalid_gold = True
        _label(args)
        self.assertEqual(rich.read_text(encoding="utf-8"), "")
        rejected = json.loads(skipped.read_text(encoding="utf-8"))
        self.assertEqual(rejected["question_id"], "1")
        self.assertIn("no such column", rejected["gold_execution"]["error"])

    def test_ordered_gold_does_not_accept_reversed_result_order(self):
        raw = self._raw_trace()
        first = raw["rounds"][0]
        first["candidates"][0]["sql"] = "SELECT name FROM items ORDER BY name ASC"
        first["candidates"][0]["canonical_sql"] = first["candidates"][0]["sql"]
        ordered_gold = "SELECT name FROM items ORDER BY name DESC"
        labelled = label_trace(raw, ordered_gold, str(self.db_path))
        self.assertTrue(labelled["order_sensitive_label"])
        self.assertEqual(labelled["rounds"][0]["covered_label"], 0)

    def test_adjudication_probe_reuses_frozen_pool_without_model_call(self):
        labelled = label_trace(self.raw, self.gold, str(self.db_path))
        state = make_action_manifest([labelled], {"1": self.raw}, 1, 42,
                                     {"max_new_candidates": 1, "max_model_calls": 2})[0]
        with (patch("alphasql.runner.selector_action_probe.JevSettings.from_environment",
                    return_value=JevSettings(api_key="test")),
              patch("alphasql.runner.selector_action_probe.JevClient")):
            arm = probe_action(state, "adjudicate", str(self.root), self.root / "arms")
        self.assertEqual(arm["selected_sql"], self.wrong)
        self.assertEqual(arm["new_candidates"], [])
        self.assertEqual(arm["usage"]["model_calls"], 0)


if __name__ == "__main__":
    unittest.main()
