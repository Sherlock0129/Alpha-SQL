"""Pure offline analysis tests; use the main project environment (sqlglot)."""
import json
from pathlib import Path
import tempfile
import unittest

from deepeye.analyze import analyze, paired_difference, schema_metrics, sql_usage, wilson
from deepeye.runner import ROOT


SCHEMA = {"tables":{
    "people":{"columns":{"id":{"column_type":"INTEGER"},"name":{"column_type":"TEXT"}}},
    "orders":{"columns":{"id":{"column_type":"INTEGER"},"person_id":{"column_type":"INTEGER"}}},
}}


class DeepEyeAnalysisTest(unittest.TestCase):
    def test_aliases_ctes_and_scopes_do_not_create_fake_tables(self):
        tables,columns,_ = sql_usage(
            "WITH x AS (SELECT p.id, p.name FROM people p) "
            "SELECT x.name FROM x JOIN orders o ON x.id = o.person_id",SCHEMA)
        self.assertEqual(tables,{"people","orders"})
        self.assertEqual(columns,{("people","id"),("people","name"),("orders","person_id")})

    def test_repeated_alias_is_resolved_in_own_scope(self):
        tables,columns,_ = sql_usage(
            "SELECT p.name FROM people p WHERE p.id IN (SELECT p.person_id FROM orders p)",SCHEMA)
        self.assertEqual(tables,{"people","orders"})
        self.assertIn(("orders","person_id"),columns)
        self.assertIn(("people","id"),columns)

    def test_schema_precision_penalizes_extra_tables_and_columns(self):
        result = schema_metrics({"people":["id","name"],"orders":["id"]},
                                "SELECT name FROM people",SCHEMA)
        self.assertEqual(result["table_precision"],0.5)
        self.assertAlmostEqual(result["column_precision"],1/3)
        self.assertEqual(result["column_recall"],1)
        self.assertTrue(result["full_schema_recall"])

    def test_parser_failure_is_not_perfect_recall(self):
        result = schema_metrics({"people":["name"]},"SELECT nonexistent FROM people",SCHEMA)
        self.assertIn("parser_error",result)
        self.assertNotIn("full_schema_recall",result)

    def test_paired_bootstrap_and_wilson(self):
        rows = [{"selected_ex_multiset":True,"sc_ex_multiset":False},
                {"selected_ex_multiset":False,"sc_ex_multiset":True}]
        result = paired_difference(rows)
        self.assertEqual(result["fixes"],1)
        self.assertEqual(result["regressions"],1)
        self.assertEqual(result["difference"],0)
        self.assertGreater(wilson(300,300)[0],0.98)

    def test_final_report_rejects_missing_questions(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"tmp",prefix="analysis-test-") as name:
            root = Path(name)
            (root/"manifest.json").write_text(json.dumps({"question_ids":[1,2]}),encoding="utf-8")
            (root/"metrics.json").write_text(json.dumps({"per_question":[{"question_id":1}]}),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"Refusing final report"):
                analyze(root)

    def test_complete_report_has_error_and_selection_diagnostics(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"tmp",prefix="analysis-test-") as name:
            root = Path(name)
            manifest = {"question_ids":[1],"model":"test-model","dataset_split":"train"}
            summary = {"evaluable":1,"questions":1,"executable_rate":1}
            row = {"question_id":1,"db_id":"toy","gold_valid":True,"coverage_set":True,
                   "coverage_multiset":True,"selected_ex_set":False,"selected_ex_multiset":False,
                   "sc_ex_set":True,"sc_ex_multiset":True,"candidate_count":2,"executable_count":2,
                   "gold_row_count":1,"candidates":[{"sql":"SELECT name FROM people","ex_multiset":True}]}
            candidate = {"question_id":1,"schema_linking":{"people":["name"]},
                         "selected_sql":"SELECT name FROM people WHERE id = 2"}
            source = {"question_id":1,"question":"Names?","gold_sql":"SELECT name FROM people",
                      "database_schema":SCHEMA}
            (root/"manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
            (root/"metrics.json").write_text(json.dumps({"summary":summary,"per_question":[row]}),encoding="utf-8")
            (root/"candidates.jsonl").write_text(json.dumps(candidate)+"\n",encoding="utf-8")
            (root/"dataset.snapshot.data").mkdir()
            (root/"dataset.snapshot.data/items.jsonl").write_text(json.dumps({"input":source})+"\n",encoding="utf-8")
            (root/"schema_linking.snapshot.data").mkdir()
            linked = {"input":source,"pipeline_artifacts":{"schema_linking":{
                "database_schema_after_schema_linking":{"tables":{"people":SCHEMA["tables"]["people"]}}}}}
            (root/"schema_linking.snapshot.data/items.jsonl").write_text(json.dumps(linked)+"\n",encoding="utf-8")
            (root/"RESUME_PROVENANCE.json").write_text(json.dumps({
                "mixed_models":True,"source_model":"qwen3.5-35b-a3b","new_model":"Qwen3.6-35B-A3B",
                "source_retained_schema_linking_ids":[1],"new_schema_linking_ids":[]}),encoding="utf-8")
            result = analyze(root)
            self.assertEqual(result["summary"]["selection_gap"],1)
            self.assertEqual(result["summary"]["error_categories"],{"selection_error":1})
            self.assertTrue((root/"REPORT.md").is_file())
            self.assertEqual(result["summary"]["schema_macro"]["column_precision"],0.5)
            self.assertEqual(result["summary"]["raw_schema_macro"]["column_precision"],1)
            self.assertEqual(result["per_question"][0]["schema_linking_model"],"qwen3.5-35b-a3b")
            self.assertEqual(result["per_question"][0]["generation_model"],"Qwen3.6-35B-A3B")
            self.assertIn("混合模型续跑",(root/"REPORT.md").read_text(encoding="utf-8"))
            errors = json.loads((root/"error_analysis.jsonl").read_text(encoding="utf-8"))
            self.assertIn("where_count_difference",errors["structural_flags"])


if __name__ == "__main__":
    unittest.main()
