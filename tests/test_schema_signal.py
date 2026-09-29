import unittest
from types import SimpleNamespace

import numpy as np

from alphasql.algorithm.selection.schema_signal import (
    LogisticReranker,
    extract_candidate_features,
    extract_linked_schema,
    generate_sql_mutations,
    extract_sql_usage,
    selection_metrics,
)
from alphasql.algorithm.selection.neural_reranker import (
    NeuralPairwiseReranker,
    build_pair_indices,
)
from alphasql.database.schema import ColumnSchema, DatabaseSchema, TableSchema
from alphasql.runner.schema_signal_experiment import split_records
from alphasql.runner.neural_schema_reranker import (
    confidence_gated_scores,
    fuse_with_self_consistency,
    tune_fusion_alpha,
)


def sample_schema():
    return DatabaseSchema(db_id="shop", tables={
        "customers": TableSchema(table_name="customers", columns={
            "id": ColumnSchema(column_type="INTEGER", primary_key=True),
            "name": ColumnSchema(column_type="TEXT"),
        }),
        "orders": TableSchema(table_name="orders", columns={
            "id": ColumnSchema(column_type="INTEGER", primary_key=True),
            "customer_id": ColumnSchema(column_type="INTEGER", foreign_keys=[("customers", "id")]),
            "amount": ColumnSchema(column_type="REAL"),
        }),
    })


class SchemaSignalTest(unittest.TestCase):
    def test_aliases_and_fk_join_are_extracted(self):
        sql = "SELECT c.name, SUM(o.amount) FROM customers c JOIN orders o ON c.id=o.customer_id GROUP BY c.name"
        usage = extract_sql_usage(sql, sample_schema())
        self.assertEqual(usage.tables, {"customers", "orders"})
        self.assertIn(("orders", "amount"), usage.columns)
        features = extract_candidate_features(
            sql, sample_schema(), {"customers", "orders"},
            {("customers", "name"), ("customers", "id"), ("orders", "amount"), ("orders", "customer_id")},
        )
        self.assertEqual(features["schema_table_coverage"], 1.0)
        self.assertEqual(features["schema_fk_connected"], 1.0)
        self.assertEqual(features["schema_fk_join_count"], 1.0)

    def test_logistic_model_learns_simple_signal(self):
        x = np.asarray([[0.0], [0.1], [0.2], [0.8], [0.9], [1.0]])
        y = np.asarray([0, 0, 0, 1, 1, 1])
        model = LogisticReranker(epochs=500).fit(x, y)
        predictions = model.predict_proba(x)
        self.assertGreater(predictions[-1], predictions[0])
        self.assertTrue(np.array_equal((predictions >= 0.5).astype(int), y))

    def test_sql_mutations_change_structure(self):
        mutations = generate_sql_mutations(
            "SELECT name FROM customers WHERE id > 10 ORDER BY name LIMIT 5"
        )
        self.assertTrue(any("WHERE" not in sql.upper() for sql in mutations))
        self.assertTrue(any("ORDER BY" not in sql.upper() for sql in mutations))
        self.assertTrue(any("<= 10" in sql for sql in mutations))

    def test_linked_schema_accepts_runtime_table_schema(self):
        path = [SimpleNamespace(selected_schema_dict={
            "orders": TableSchema(table_name="orders", columns={
                "id": ColumnSchema(column_type="INTEGER"),
                "amount": ColumnSchema(column_type="REAL"),
            })
        })]
        tables, columns = extract_linked_schema(path)
        self.assertEqual(tables, {"orders"})
        self.assertEqual(columns, {("orders", "id"), ("orders", "amount")})

    def test_selection_metrics_measure_gap_closure(self):
        records = [
            {"question_id": 1, "candidate_index": 0, "label": 0, "features": {"exec_result_group_size": 2}},
            {"question_id": 1, "candidate_index": 1, "label": 1, "features": {"exec_result_group_size": 1}},
            {"question_id": 2, "candidate_index": 0, "label": 1, "features": {"exec_result_group_size": 2}},
            {"question_id": 2, "candidate_index": 1, "label": 0, "features": {"exec_result_group_size": 1}},
        ]
        metrics = selection_metrics(records, [0.1, 0.9, 0.9, 0.1])
        self.assertEqual(metrics["oracle_ex"], 1.0)
        self.assertEqual(metrics["baseline_ex"], 0.5)
        self.assertEqual(metrics["model_ex"], 1.0)
        self.assertEqual(metrics["gap_closure"], 1.0)

    def test_database_split_keeps_databases_disjoint(self):
        records = [
            {"question_id": question, "db_id": database}
            for database in ("a", "b", "c")
            for question in (database + "1", database + "2")
        ]
        train, evaluation, key = split_records(records, 0.34, 42, "database")
        self.assertEqual(key, "db_id")
        self.assertTrue({row["db_id"] for row in train}.isdisjoint({row["db_id"] for row in evaluation}))

    def test_pairwise_neural_reranker_learns_with_cpu_fallback(self):
        records = [
            {"question_id": "a", "label": 1, "features": {}},
            {"question_id": "a", "label": 0, "features": {"exec_success": 1.0}},
            {"question_id": "b", "label": 1, "features": {}},
            {"question_id": "b", "label": 0, "features": {"schema_table_coverage": 1.0}},
        ]
        positive, negative, weights = build_pair_indices(records)
        self.assertEqual(len(positive), 2)
        self.assertTrue(np.all(weights > 1.0))
        values = np.asarray([[1.0], [0.0], [0.9], [0.1]], dtype=np.float32)
        labels = np.asarray([1, 0, 1, 0])
        model = NeuralPairwiseReranker(
            hidden_sizes=(8,), dropout=0.0, epochs=40, learning_rate=0.02,
            batch_size=4, device="cpu",
        ).fit(values, labels, records)
        scores = model.predict_proba(values)
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[2], scores[3])

    def test_zero_residual_weight_exactly_preserves_self_consistency(self):
        records = [
            {"question_id": "a", "candidate_index": 0, "label": 1,
             "features": {"exec_result_group_size": 2}},
            {"question_id": "a", "candidate_index": 1, "label": 0,
             "features": {"exec_result_group_size": 1}},
        ]
        scores = fuse_with_self_consistency(records, [0.01, 0.99], alpha=0.0)
        metrics = selection_metrics(records, scores)
        self.assertEqual(metrics["model_ex"], metrics["baseline_ex"])

    def test_fusion_tuning_can_use_neural_signal_to_repair_baseline(self):
        records = [
            {"question_id": "a", "candidate_index": 0, "label": 0,
             "features": {"exec_result_group_size": 2}},
            {"question_id": "a", "candidate_index": 1, "label": 1,
             "features": {"exec_result_group_size": 1}},
        ]
        alpha, trials = tune_fusion_alpha(records, [0.01, 0.99], [0.0, 1.0])
        self.assertEqual(alpha, 1.0)
        self.assertGreater(trials[1]["model_ex"], trials[0]["model_ex"])

    def test_confidence_gate_preserves_baseline_below_threshold(self):
        records = [
            {"question_id": "a", "candidate_index": 0, "label": 1,
             "features": {"exec_result_group_size": 2}},
            {"question_id": "a", "candidate_index": 1, "label": 0,
             "features": {"exec_result_group_size": 1}},
        ]
        scores = confidence_gated_scores(records, [0.50, 0.55], threshold=0.1)
        self.assertEqual(selection_metrics(records, scores)["model_ex"], 1.0)


if __name__ == "__main__":
    unittest.main()
