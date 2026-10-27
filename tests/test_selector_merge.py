import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from alphasql.runner.selector_merge import merge_shards


class SelectorMergeTest(unittest.TestCase):
    def test_merge_requires_complete_unique_frozen_traces(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "tmp") as raw:
            root = Path(raw)
            tasks = root / "tasks.pkl"
            tasks.write_bytes(pickle.dumps([SimpleNamespace(question_id=1),
                                            SimpleNamespace(question_id=2)]))
            first, second = root / "shard_0", root / "shard_1"
            first.mkdir()
            second.mkdir()

            def result(question_id):
                return {"schema_version": 2, "question_id": question_id,
                        "generator": {"version": "frozen"},
                        "judge": {"model": "jev-1.14.0"},
                        "calibrator": {"kind": "raw_jev"},
                        "rounds": [{"round": 0, "pool_size": 1,
                                    "candidates": [{"execution": {"valid": True}}],
                                    "jev": {"called": True, "model": "jev-1.14.0",
                                            "candidate": [0.5], "gap_schema": 0.1,
                                            "gap_projection": 0.2, "gap_filters": 0.3,
                                            "gap_logic": 0.4}}]}

            (first / "1.json").write_text(json.dumps(result(1)), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Missing 1 questions"):
                merge_shards([first], tasks, root / "merged")
            partial = merge_shards([first], tasks, root / "merged", allow_incomplete=True)
            self.assertEqual(partial["merged_questions"], 1)
            (second / "2.json").write_text(json.dumps(result(2)), encoding="utf-8")
            complete = merge_shards([first, second], tasks, root / "merged")
            self.assertEqual(complete["merged_questions"], 2)
            (second / "1.json").write_text(json.dumps(result(1)), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate question ID"):
                merge_shards([first, second], tasks, root / "other")


if __name__ == "__main__":
    unittest.main()
