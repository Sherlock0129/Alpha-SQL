import unittest

from alphasql.algorithm.selection.execution_evidence import pool_identity
from alphasql.runner.counterfactual_probe import (
    _execute_fixture, _safe_python, _validate_fixture, eligible_pair,
)
from alphasql.runner.selection_v2 import _fit_ranker, _rank_score, _select


class SelectionV2Test(unittest.TestCase):
    def test_execution_evidence_identity_depends_on_sql(self):
        row = {"db_id": "db", "question_id": 1,
               "rounds": [{"candidates": [{"sql": "SELECT 1"}]}]}
        before = pool_identity(row)
        row["rounds"][0]["candidates"][0]["sql"] = "SELECT 2"
        self.assertNotEqual(before, pool_identity(row))

    def test_pairwise_ranker_learns_execution_signal(self):
        def item(label, cluster):
            return {"index": label, "label": label, "raw": .5,
                    "views": {}, "execution": {"features": {"cluster_fraction": cluster}}}
        pools = [{"items": [item(1, 1.0), item(0, .2)]} for _ in range(4)]
        model = _fit_ranker(pools, ["exec:cluster_fraction"], .1)
        self.assertGreater(_rank_score(pools[0]["items"][0], model, .4),
                           _rank_score(pools[0]["items"][1], model, .4))

    def test_restricted_python_matches_simple_sql(self):
        schema = {"product": {"name": "TEXT", "price": "INTEGER"}}
        fixture = _validate_fixture({"tables": {"product": [
            {"name": "A", "price": 110}, {"name": "B", "price": 90}]}}, schema)
        sql = _execute_fixture(schema, fixture,
                               "SELECT name FROM product WHERE price > 100")
        python = _safe_python(
            '[(r["name"],) for r in tables["product"] if r["price"] > 100]', fixture)
        self.assertEqual(sql, python)

    def test_restricted_python_rejects_code_execution(self):
        with self.assertRaises(ValueError):
            _safe_python('__import__("os").system("echo unsafe")', {})
        with self.assertRaises(ValueError):
            _safe_python('sum(range(1000000000))', {})

    def test_counterfactual_gate_is_bounded(self):
        row = {"rounds": [{"candidates": [
            {"candidate_raw": .9, "canonical_sql": "SELECT 1",
             "execution": {"valid": True}},
            {"candidate_raw": .8, "canonical_sql": "SELECT 2",
             "execution": {"valid": True}}]}]}
        evidence = {"candidates": [{"result_hash": "a"}, {"result_hash": "b"}]}
        self.assertEqual(eligible_pair(row, evidence, .2), (0, 1, "close_scores"))
        self.assertIsNone(eligible_pair(row, evidence, .05))

    def test_probe_only_overrides_on_decisive_matching_pair(self):
        pool = {"items": [{"index": 0, "raw": .9}, {"index": 1, "raw": .8}],
                "probe": {"status": "decisive", "pair": [0, 1],
                          "recommended_index": 1}}
        self.assertEqual(_select(pool, "jev_probe", None, None)[0]["index"], 1)
        pool["probe"]["status"] = "projection_ambiguity"
        self.assertEqual(_select(pool, "jev_probe", None, None)[0]["index"], 0)


if __name__ == "__main__":
    unittest.main()
