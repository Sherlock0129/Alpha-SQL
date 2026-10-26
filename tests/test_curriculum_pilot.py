import copy
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from alphasql.algorithm.selection.curriculum import (
    CurriculumController, PairPool, SharedDrawStreams, weak_membership,
)
from alphasql.runner.curriculum_pilot import (
    ARMS, audit_scope, equivalence_report, offline_only, selection_evaluation, train_arm,
)
from alphasql.runner.schema_signal_experiment import build_records


def fixture():
    rows = []
    sqls = [("SELECT a FROM t", "SELECT b FROM t WHERE b=2"),
            ("SELECT count(*) FROM t", "SELECT sum(a) FROM t"),
            ("SELECT a FROM t", "SELECT a FROM t JOIN u ON t.id=u.id")]
    for q, pair in enumerate(sqls):
        for i, sql in enumerate(pair):
            rows.append({"question_id": str(q), "candidate_index": i, "sql": sql,
                         "label": int(i == 0), "candidate_source": "mcts",
                         "features": {"exec_success": 1.0,
                                      "exec_result_group_size": float(2 if i == (q % 2) else 1),
                                      "schema_column_coverage": float(i), "sql_length": float(len(sql))}})
    return rows


class CurriculumPilotTests(unittest.TestCase):
    def test_q0_question_balanced_and_keep_is_sc_correct(self):
        records = fixture()
        # Add a second negative to q0: question probabilities must stay equal.
        extra = copy.deepcopy(records[1])
        extra["candidate_index"] = 2
        records.append(extra)
        pool = PairPool.from_records(records)
        for qid in set(pool.question_ids):
            self.assertAlmostEqual(sum(p for q, p in zip(pool.question_ids, pool.q0) if q == qid), 1 / 3)
        self.assertTrue(all(k == 0 for q, k in zip(pool.question_ids, pool.keep) if q == "1"))
        np.testing.assert_allclose(pool.membership.sum(axis=1), 1)

    def test_eta_zero_matches_lp_exactly_over_multiple_probes(self):
        pool = PairPool.from_records(fixture())
        lp = CurriculumController(pool, "loss_progress")
        ea0 = CurriculumController(pool, "error_aware", eta=0)
        rng = np.random.default_rng(987)
        for _ in range(30):
            losses = rng.random(len(pool.q0)) * 2
            margins = rng.normal(size=len(pool.q0))
            np.testing.assert_array_equal(lp.probe(losses, margins), ea0.probe(losses, margins))
            self.assertTrue(np.all(lp.probabilities >= .25 * pool.q0))

    def test_semantic_term_changes_distribution(self):
        pool = PairPool.from_records(fixture())
        lp = CurriculumController(pool, "loss_progress")
        ea = CurriculumController(pool, "error_aware")
        loss = np.array([.1, 1.0, 2.0])
        margin = np.array([2.0, -1.0, -3.0])
        self.assertGreater(np.max(np.abs(lp.probe(loss, margin) - ea.probe(loss, margin))), 1e-8)

    def test_shared_keep_anchors_and_warmup(self):
        records = fixture()
        pool = PairPool.from_records(records)
        a, b = SharedDrawStreams(101, records), SharedDrawStreams(101, records)
        for step in range(20):
            left = a.draw(pool, pool.q0)
            right = b.draw(pool, np.array([.8, .1, .1]), warmup=step < 5)
            self.assertEqual(left[1:], right[1:])
            if step < 5:
                self.assertEqual(left, right)

    def test_weak_labels_normalized_and_failure_unknown(self):
        records = fixture()
        membership = weak_membership(records[0], records[1])
        self.assertGreater(membership[0], 0)
        self.assertGreater(membership[1], 0)
        self.assertAlmostEqual(float(membership.sum()), 1)
        negative = copy.deepcopy(records[1])
        negative["features"]["exec_success"] = 0
        np.testing.assert_array_equal(weak_membership(records[0], negative), [0, 0, 0, 0, 1])

    def test_empty_execution_remains_valid_and_invalid_masked(self):
        records = fixture()[:2]
        records[0]["features"]["exec_nonempty"] = 0
        records[1]["features"]["exec_success"] = 0
        metrics, rows = selection_evaluation(records, [-10, 100])
        self.assertEqual(rows[0]["selected_candidate_index"], 0)
        self.assertEqual(metrics["correct"], 1)

    def test_scope_rejects_test_and_split_overlap(self):
        split = {"train_ids": ["0"], "validation_ids": ["1"], "test_ids": ["2"]}
        with self.assertRaisesRegex(ValueError, "outside"):
            audit_scope(fixture(), split)
        train, val = audit_scope(fixture()[:4], split)
        self.assertEqual(len(train), 2)
        self.assertEqual(len(val), 2)
        split["test_ids"] = ["1"]
        with self.assertRaisesRegex(ValueError, "overlaps"):
            audit_scope(fixture()[:4], split)

    def test_builder_filter_precedes_gold_db_and_pickle_access(self):
        with patch("alphasql.runner.schema_signal_experiment._load_dataset", return_value={"test": {}}), \
                patch("alphasql.runner.schema_signal_experiment.Path.glob", return_value=[Path("test.pkl")]), \
                patch("alphasql.runner.schema_signal_experiment.DatabaseManager.get_database_schema") as schema:
            self.assertEqual(build_records("unused", "unused", "unused", question_ids=["train"]), [])
            schema.assert_not_called()

    def test_network_is_blocked(self):
        with offline_only(), self.assertRaisesRegex(RuntimeError, "forbidden"):
            socket.create_connection(("example.com", 443))

    def test_actual_training_trajectory_parity(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        train = fixture()
        pool = PairPool.from_records(train)
        features = sorted(train[0]["features"])
        settings = {"hidden_sizes": [8, 4], "dropout": .1, "lr": .001,
                    "weight_decay": .0001, "decay": .9, "rho": .25, "kappa": 10,
                    "eta": 1., "updates": 12, "probe_every": 2, "warmup": 3,
                    "pair_batch": 8, "anchor_batch": 16, "pointwise_weight": .2, "clip": 5}
        temp_root = Path(__file__).resolve().parents[1] / "tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as tmp:
            runs = {name: train_arm(name, train, train, features, pool, settings, 101,
                                   torch.device("cpu"), Path(tmp) / name) for name in ARMS}
            report = equivalence_report(runs, settings["warmup"])
            self.assertTrue(report["passed"], report)
            self.assertEqual(report["max_parameter_absolute_difference"], 0)


if __name__ == "__main__":
    unittest.main()
