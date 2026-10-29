import json
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from alphasql.algorithm.selection.content_verifier import ContentVerifier
from alphasql.algorithm.compiler.validation import CandidateExecution
from alphasql.algorithm.selection.selective_stop import StopPolicy
from alphasql.runner.fresh_selector_analyze import main as analyze
from alphasql.runner.fresh_selector_branch import decision
from alphasql.runner.fresh_selector_run import _question_usage


class FreshSelectorExperimentTest(unittest.TestCase):
    def test_branch_actions_share_one_first_pool(self):
        policy = StopPolicy("candidate_raw", .815, .1, 47, 32, ("validation_db",))
        first = {"jev": {"pool_covered": .9},
                 "candidates": [{"candidate_raw": .9, "execution": {"valid": True}}]}
        self.assertEqual(decision(first, policy), "stop")
        first["jev"]["pool_covered"] = .2
        self.assertEqual(decision(first, policy), "adjudicate")
        first["jev"]["pool_covered"] = .9
        first["candidates"][0]["candidate_raw"] = .8
        self.assertEqual(decision(first, policy), "continue")
        first["candidates"][0]["execution"]["valid"] = False
        self.assertEqual(decision(first, policy), "continue")

    def test_cost_logs_remain_separate_by_round(self):
        root = Path(__file__).resolve().parents[1] / "tmp"
        with tempfile.TemporaryDirectory(dir=root) as dirname:
            path = Path(dirname) / "1.jsonl"
            for index, prompt in ((0, 10), (1, 20)):
                round_path = path.with_name(f"1_round{index}.jsonl")
                round_path.write_text(json.dumps({"prompt_tokens": prompt,
                                                  "completion_tokens": 5}) + "\n", encoding="utf-8")
            usage = _question_usage(path)
            self.assertEqual(usage["calls"], 2)
            self.assertEqual(usage["input_tokens"], 30)
            self.assertEqual([item["calls"] for item in usage["rounds"]], [1, 1])

    def test_offline_analysis_of_shared_first_pool(self):
        root = Path(__file__).resolve().parents[1] / "tmp"
        with tempfile.TemporaryDirectory(dir=root) as dirname:
            folder = Path(dirname)
            for name in ("protocol", "baseline", "branch", "dbs/db"):
                (folder / name).mkdir(parents=True)
            sql = "SELECT name FROM items WHERE name = 'Ada'"
            dataset = [{"db_id": "db", "question": "Which name is Ada?",
                        "evidence": "", "SQL": sql}]
            (folder / "data.json").write_text(json.dumps(dataset), encoding="utf-8")
            manifest = {"protocol": "fixture", "selected_databases": ["db"],
                        "question_ids": [0],
                        "dataset_sha256": hashlib.sha256((folder / "data.json").read_bytes()).hexdigest()}
            (folder / "protocol/manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            candidate = {"sql": sql, "candidate_raw": .9,
                         "execution": {"valid": True, "row_count": 1,
                                       "columns": ["name"], "sample_rows": [["Ada"]]}}
            baseline = {"question_id": 0, "db_id": "db", "question": dataset[0]["question"],
                        "evidence": "", "full_schema": {"tables": {}},
                        "rounds": [{"jev": {"pool_covered": .9, "called": True,
                                            "usage": {"input_tokens": 20, "output_tokens": 2}},
                                    "candidates": [candidate], "pool_size": 1}],
                        "selected_sql": sql, "selected_index": 0, "candidates": [candidate],
                        "llm_usage": {"calls": 1, "input_tokens": 10, "output_tokens": 5,
                                      "rounds": [{"calls": 1, "input_tokens": 10,
                                                  "output_tokens": 5},
                                                 {"calls": 0, "input_tokens": 0,
                                                  "output_tokens": 0}]},
                        "jev_usage_total": {"input_tokens": 20, "output_tokens": 2}}
            (folder / "baseline/0.json").write_text(json.dumps(baseline), encoding="utf-8")
            StopPolicy("candidate_raw", .815, .1, 47, 32, ("validation_db",)).save(folder / "policy.json")
            ContentVerifier({"bias": 0}, [0]).save(folder / "verifier.json")
            argv = ["analyze", "--protocol-dir", str(folder / "protocol"),
                    "--baseline-dir", str(folder / "baseline"),
                    "--branch-dir", str(folder / "branch"),
                    "--policy", str(folder / "policy.json"),
                    "--verifier-model", str(folder / "verifier.json"),
                    "--data-path", str(folder / "data.json"),
                    "--database-root-dir", str(folder / "dbs"),
                    "--output-dir", str(folder / "analysis")]
            with patch.object(sys, "argv", argv), patch(
                    "alphasql.runner.fresh_selector_analyze.validate_sql",
                    return_value=CandidateExecution(True, "success", ["name"], [["Ada"]])):
                analyze()
            report = json.loads((folder / "analysis/report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["baseline"]["selected_correct"], 1)
            self.assertEqual(report["selective"]["selected_correct"], 1)
            self.assertEqual(report["baseline"]["llm_calls"], 1)


if __name__ == "__main__":
    unittest.main()
