"""Offline tests for frozen run migration and provider credential isolation."""
import hashlib
import json
import os
from pathlib import Path
import socket
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from deepeye import runner
from deepeye.resume import DEFAULT_BASE_URL, DEFAULT_MODEL, prepare_continuation


class DeepEyeResumeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=runner.ROOT / "tmp", prefix="resume-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.root_patch = patch.object(runner, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.source = runner.run_dir("source")
        self.source.mkdir(parents=True)
        db_root = self.root / "data/bird/train"
        db = db_root / "train_databases/toy/toy.sqlite"
        db.parent.mkdir(parents=True)
        sqlite3.connect(db).close()
        rows = [{"question_id":i, "db_id":"toy", "question":f"Question {i}",
                 "SQL":"SELECT 1", "evidence":""} for i in range(300)]
        (db_root / "train.json").write_text(json.dumps(rows), encoding="utf-8")
        (self.source / "manifest.json").write_text(json.dumps({
            "question_ids":list(range(300)), "dataset_root":str(self.root / "data/bird"),
            "dataset_split":"train", "profile":"full", "few_shot":False,
            "model":"qwen3.5-35b-a3b", "embedding_model":"text-embedding-v4",
            "embedding_base_url":"https://dashscope.aliyuncs.com/compatible-mode/v1"}), encoding="utf-8")
        (self.source / "CHECKPOINT_RECOVERY.json").write_text(json.dumps({
            "retained_ids":list(range(286)), "retry_ids":list(range(286,300))}), encoding="utf-8")
        for name in ("dataset.snapshot", "value_retrieval.snapshot"):
            data_dir = self.source / f"{name}.data"
            data_dir.mkdir()
            (data_dir / "items.jsonl").write_text("".join(json.dumps({"input":{"question_id":i}}) + "\n"
                                                        for i in range(300)), encoding="utf-8")
            (self.source / name).write_text(json.dumps({
                "format":"structured_dataset_snapshot", "snapshot_root":data_dir.name,
                "dataset_config":{"save_path":str(self.source / "dataset.snapshot")}}), encoding="utf-8")
        index = self.source / "vector_database/toy"
        index.mkdir(parents=True)
        (index / "success_flag").touch()
        artifacts = self.source / "schema_linking.artifacts"
        artifacts.mkdir()
        (artifacts / "meta.json").write_text(json.dumps({"save_path":str(self.source / "schema_linking.snapshot")}), encoding="utf-8")
        self.checkpoint = artifacts / "schema_linking.jsonl"
        self.checkpoint.write_text("".join(json.dumps({
            "item_id":str(i), "stage_artifact":{
                "direct_linked_tables_and_columns":{"toy":["id"]},
                "reversed_linked_tables_and_columns":{"toy":["id"]}}}) + "\n"
            for i in range(286)), encoding="utf-8")

    def test_mixed_model_resume_preserves_source_and_freezes_new_model(self):
        before = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        with patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")):
            out = prepare_continuation("source", "target")
        self.assertEqual(before, hashlib.sha256(self.checkpoint.read_bytes()).hexdigest())
        self.assertEqual(before, hashlib.sha256((out / "schema_linking.artifacts/schema_linking.jsonl").read_bytes()).hexdigest())
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual((manifest["model"], manifest["base_url"]), (DEFAULT_MODEL, DEFAULT_BASE_URL))
        self.assertEqual(manifest["embedding_model"], "text-embedding-v4")
        self.assertIn("[llm_profiles.local.extra_body.chat_template_kwargs]", (out / "config.toml").read_text(encoding="utf-8"))
        self.assertEqual(len(json.loads((out / "RESUME_PROVENANCE.json").read_text(encoding="utf-8"))["new_schema_linking_ids"]), 14)
        for name in ("dataset.snapshot", "value_retrieval.snapshot"):
            snapshot = json.loads((out / name).read_text(encoding="utf-8"))
            self.assertEqual(snapshot["dataset_config"]["save_path"], str(out / "dataset.snapshot"))
        self.assertEqual(json.loads((out / "schema_linking.artifacts/meta.json").read_text(encoding="utf-8"))["save_path"],
                         str(out / "schema_linking.snapshot"))
        with self.assertRaises(FileExistsError):
            prepare_continuation("source", "target")

    def test_school_endpoint_does_not_inherit_aliyun_key(self):
        with patch("dotenv.dotenv_values", return_value={}), patch.dict(os.environ, {
                "DEEPEYE_BASE_URL":DEFAULT_BASE_URL, "DASHSCOPE_API_KEY":"old-provider-secret"
                }, clear=True):
            runner.configure_environment()
            self.assertFalse(os.environ.get("DEEPEYE_API_KEY"))
            with self.assertRaisesRegex(ValueError, "DEEPEYE_API_KEY"):
                runner.check_credentials()
            os.environ["AIGC_API_KEY"] = "school-test-secret"
            runner.configure_environment()
            self.assertEqual(os.environ["DEEPEYE_API_KEY"], "school-test-secret")
            self.assertEqual(runner.redact("school-test-secret"), "[REDACTED]")


if __name__ == "__main__":
    unittest.main()
