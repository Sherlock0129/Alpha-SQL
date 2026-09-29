import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from alphasql.algorithm.compiler.beam_search import QueryPlanBeamSearch
from alphasql.algorithm.compiler.query_plan import (
    Aggregate,
    ColumnRef,
    Comparison,
    Expression,
    JoinSpec,
    OrderDirection,
    OrderSpec,
    Predicate,
    QueryPlan,
    SchemaGraph,
    SchemaValidationError,
    SelectItem,
)
from alphasql.algorithm.compiler.sql_compiler import SQLCompiler
from alphasql.algorithm.compiler.validation import validate_sql
from alphasql.algorithm.schema_linking.jev_linker import (
    CandidateJoinPath,
    IntentDecision,
    JevSchemaLinker,
    SchemaLinkingResult,
    ScoredColumn,
    ScoredTable,
)
from alphasql.algorithm.selection.jev_reranker import (
    JevSQLReranker,
    RerankCandidate,
)
from alphasql.database.schema import ColumnSchema, DatabaseSchema, TableSchema
from alphasql.jev.client import JevClient
from alphasql.jev.models import ChoiceQuestion, NoulQuestion, ScoreQuestion
from alphasql.jev.runtime import JevSettings
from alphasql.runner.jev_candidate_runner import JevCandidateRunner, retrieve_exact_question_values


def sample_schema():
    return DatabaseSchema(db_id="shop", tables={
        "customers": TableSchema(table_name="customers", columns={
            "id": ColumnSchema(original_column_name="id", column_type="INTEGER", primary_key=True),
            "name": ColumnSchema(original_column_name="name", column_type="TEXT"),
        }),
        "orders": TableSchema(table_name="orders", columns={
            "id": ColumnSchema(original_column_name="id", column_type="INTEGER", primary_key=True),
            "customer_id": ColumnSchema(
                original_column_name="customer_id", column_type="INTEGER",
                foreign_keys=[("customers", "id")],
            ),
            "amount": ColumnSchema(original_column_name="amount", column_type="REAL"),
        }),
    })


def choice_answer(question, selected=None, confidence=0.9):
    names = list(question.criteria)
    selected = selected if selected in names else names[0]
    remaining = (1.0 - 0.8) / max(len(names) - 1, 1)
    probabilities = {name: (0.8 if name == selected else remaining) for name in names}
    if len(names) == 1:
        probabilities[selected] = 1.0
    return {"choice": selected, "probabilities": probabilities, "confidence": confidence}


class JevSchemaLinkerTest(unittest.TestCase):
    def test_retrieved_literals_are_globally_prioritized_over_generic_examples(self):
        schema = DatabaseSchema(db_id="geo", tables={
            "City": TableSchema(table_name="City", columns={
                "Name": ColumnSchema(
                    original_column_name="Name", column_type="TEXT",
                    value_examples=["Kabul", "Herat"],
                )
            }),
            "Country": TableSchema(table_name="Country", columns={
                "Continent": ColumnSchema(
                    original_column_name="Continent", column_type="TEXT",
                    value_examples=["Europe"],
                )
            }),
        })
        values = JevSchemaLinker._literal_values(
            schema, {"Country": {"Continent": ["Asia"]}}
        )
        self.assertEqual(values[0], ("Country", "Continent", "Asia"))

    def test_threshold_filtering_and_key_completion(self):
        def transport(state, questions):
            answers = {}
            for key, question in questions.items():
                if isinstance(question, NoulQuestion):
                    # Select both tables and semantic name/amount columns, but deliberately
                    # score PK/FK columns below threshold so constraint completion must add them.
                    high = key in {"table_0", "table_1", "column_0_1", "column_1_2"}
                    answers[key] = {"noul": 0.9 if high else 0.1, "confidence": 0.8}
                elif key == "join_path":
                    answers[key] = choice_answer(question, "path_0")
                elif key == "intent_aggregate":
                    answers[key] = choice_answer(question, "sum")
                else:
                    answers[key] = choice_answer(question)
            return {"answers": answers, "usage": {"input_tokens": 10, "output_tokens": 5}}

        settings = JevSettings(schema_threshold=0.5, beam_width=32)
        linked = JevSchemaLinker(JevClient(settings, transport), settings).link(
            "Total order amount by customer", "", sample_schema()
        )
        selected = {(item.table, item.column) for item in linked.columns if item.selected}
        self.assertIn(("customers", "name"), selected)
        self.assertIn(("orders", "amount"), selected)
        self.assertIn(("customers", "id"), selected)
        self.assertIn(("orders", "customer_id"), selected)
        self.assertTrue(any(item.added_by_constraint for item in linked.columns))
        self.assertEqual(linked.decisions["aggregate"].choice, "sum")


class CompilerAndBeamTest(unittest.TestCase):
    def setUp(self):
        self.graph = SchemaGraph(sample_schema())

    def test_legal_join_compiles_and_illegal_join_is_rejected(self):
        plan = QueryPlan(
            from_table="customers",
            select=[
                SelectItem(Expression(ColumnRef("customers", "name"))),
                SelectItem(Expression(ColumnRef("orders", "amount"), Aggregate.SUM), "total"),
            ],
            joins=[JoinSpec(
                "orders", ColumnRef("orders", "customer_id"), ColumnRef("customers", "id")
            )],
            where=[Predicate(Expression(ColumnRef("orders", "amount")), Comparison.GT, 10)],
            group_by=[Expression(ColumnRef("customers", "name"))],
            order_by=[OrderSpec(
                Expression(ColumnRef("orders", "amount"), Aggregate.SUM), OrderDirection.DESC
            )],
            limit=5,
        )
        sql = SQLCompiler(self.graph).compile(plan)
        self.assertIn("INNER JOIN", sql)
        self.assertIn("GROUP BY", sql)
        self.assertIn("LIMIT 5", sql)
        illegal = QueryPlan(
            from_table="customers",
            select=[SelectItem(Expression(ColumnRef("customers", "name")))],
            joins=[JoinSpec(
                "orders", ColumnRef("orders", "id"), ColumnRef("customers", "id")
            )],
        )
        with self.assertRaises(SchemaValidationError):
            self.graph.validate_plan(illegal)

    def test_beam_uses_non_argmax_probabilities(self):
        edge = self.graph.edges[0]
        linked = SchemaLinkingResult(
            tables=[ScoredTable("orders", 0.9, 0.9, True)],
            columns=[ScoredColumn("orders", "amount", 0.9, 0.9, True)],
            join_paths=[CandidateJoinPath((), 1.0, 0.9, True)],
            literals=[],
            decisions={
                "aggregate": IntentDecision("sum", {"sum": 0.7, "none": 0.3}, 0.9),
                "distinct": IntentDecision("no", {"no": 0.8, "yes": 0.2}, 0.9),
                "order": IntentDecision("none", {"none": 1.0}, 0.9),
                "comparison": IntentDecision("=", {"=": 1.0}, 0.9),
                "date_function": IntentDecision("none", {"none": 1.0}, 0.9),
            },
        )
        candidates = QueryPlanBeamSearch(self.graph, beam_width=16).generate(linked)
        aggregates = {candidate.plan.select[0].expression.aggregate for candidate in candidates}
        self.assertEqual(aggregates, {Aggregate.SUM, Aggregate.NONE})
        self.assertTrue(any(
            candidate.decision_probabilities["aggregate"] == 0.3 for candidate in candidates
        ))

    def test_empty_result_is_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.sqlite"
            with closing(sqlite3.connect(path)) as connection:
                connection.execute("CREATE TABLE items (id INTEGER)")
                connection.commit()
            execution = validate_sql(str(path), "SELECT id FROM items WHERE id = 999")
            self.assertTrue(execution.valid)
            self.assertEqual(execution.rows, [])


class JevRerankerTest(unittest.TestCase):
    def test_weighted_score_and_low_confidence_are_reported(self):
        def transport(state, questions):
            answers = {}
            for key, question in questions.items():
                if isinstance(question, ScoreQuestion):
                    answers[key] = {
                        "score": 4,
                        "probabilities": {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.1, "4": 0.9},
                        "confidence": 0.2 if key.endswith("_filters") else 0.9,
                    }
                else:
                    answers[key] = {"noul": 0.9, "confidence": 0.9}
            return {"answers": answers}

        settings = JevSettings(rerank_threshold=0.8, min_confidence=0.5)
        reranker = JevSQLReranker(JevClient(settings, transport), settings)
        results = reranker.rerank("question", "hint", [RerankCandidate(
            "SELECT 1", {"select": [1]}, {"valid": True, "row_count": 1}, {"tables": []}
        )])
        self.assertGreater(results[0].score, 0.8)
        self.assertTrue(results[0].above_threshold)
        self.assertTrue(results[0].uncertain)


class JevRunnerIntegrationTest(unittest.TestCase):
    def test_exact_question_value_retrieval_prioritizes_mentioned_literal(self):
        schema = DatabaseSchema(db_id="items", tables={
            "items": TableSchema(table_name="items", columns={
                "name": ColumnSchema(original_column_name="name", column_type="TEXT"),
            })
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "items.sqlite"
            with closing(sqlite3.connect(path)) as connection:
                connection.execute("CREATE TABLE items (name TEXT)")
                connection.executemany("INSERT INTO items(name) VALUES (?)", [("Ada",), ("Grace",)])
                connection.commit()
            values = retrieve_exact_question_values(path, schema, "Show the item named Ada")
        self.assertEqual(values["items"]["name"], ["Ada"])

    def test_mocked_vertical_slice_selects_executable_sql(self):
        schema = DatabaseSchema(db_id="items", tables={
            "items": TableSchema(table_name="items", columns={
                "id": ColumnSchema(original_column_name="id", column_type="INTEGER", primary_key=True),
                "name": ColumnSchema(original_column_name="name", column_type="TEXT"),
            })
        })

        def transport(state, questions):
            answers = {}
            for key, question in questions.items():
                if isinstance(question, NoulQuestion):
                    if key.startswith("c") and key[1:2].isdigit():
                        answers[key] = {"noul": 0.95}
                    else:
                        answers[key] = {"noul": 0.9 if key in {"table_0", "column_0_1"} else 0.1}
                elif isinstance(question, ScoreQuestion):
                    answers[key] = {
                        "score": 4,
                        "probabilities": {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.1, "4": 0.9},
                        "confidence": 0.9,
                    }
                elif key == "intent_aggregate":
                    answers[key] = choice_answer(question, "none")
                elif key == "intent_distinct":
                    answers[key] = choice_answer(question, "no")
                elif key == "intent_order":
                    answers[key] = choice_answer(question, "none")
                elif key == "intent_comparison":
                    answers[key] = choice_answer(question, "=")
                elif key == "intent_date_function":
                    answers[key] = choice_answer(question, "none")
            return {"answers": answers}

        settings = JevSettings(beam_width=8, rerank_threshold=0.5)
        runner = JevCandidateRunner(JevClient(settings, transport), settings)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "items.sqlite"
            with closing(sqlite3.connect(path)) as connection:
                connection.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)")
                connection.execute("INSERT INTO items(name) VALUES ('Ada')")
                connection.commit()
            result = runner.run("List item names", "", schema, path, question_id=7)
        self.assertIsNotNone(result.selected_sql)
        self.assertIn('"name"', result.selected_sql)
        self.assertFalse(result.uncertain)
        self.assertEqual(result.to_selection_payload()["question_id"], 7)


if __name__ == "__main__":
    unittest.main()
