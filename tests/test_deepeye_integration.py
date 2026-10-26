"""Offline integration checks; no provider keys, downloads or remote inference."""
from contextlib import ExitStack, closing
import gc
import importlib
import json
import os
from pathlib import Path
import runpy
import socket
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from deepeye import runner

sys.path.insert(0, str(runner.UPSTREAM))
config_module = importlib.import_module("app.config.config")
from app.config import DatasetConfig
from app.dataset import DatasetFactory, load_dataset, save_dataset
from app.db_utils import execute_sql_without_cache


class DeepEyeIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="deepeye-test-",dir=runner.ROOT/"tmp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root/"data/bird"
        for split in ("dev","train"):
            db = self.data/split/f"{split}_databases"/"toy"/"toy.sqlite"
            db.parent.mkdir(parents=True)
            with closing(sqlite3.connect(db)) as connection:
                connection.execute("CREATE TABLE people (id INTEGER PRIMARY KEY, name TEXT, city TEXT)")
                connection.executemany("INSERT INTO people VALUES (?, ?, ?)",
                                       [(1,"Ada","Paris"),(2,"Bob","Rome"),(3,"Ada","Paris")])
                connection.commit()
            records = [dict(question_id=q, db_id="toy", question="List names of people in Paris",
                            evidence="", SQL="SELECT name FROM people WHERE city = 'Paris'",difficulty="simple")
                       for q in (7,13)]
            (self.data/split/f"{split}.json").write_text(json.dumps(records),encoding="utf-8")
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.addCleanup(self.close_connections)
        self.stack.enter_context(patch.object(runner,"ROOT",self.root))
        self.stack.enter_context(patch.dict(os.environ,{
            "DEEPEYE_MODEL":"test-model","DEEPEYE_BASE_URL":"http://127.0.0.1:1/v1",
            "DEEPEYE_API_KEY":"offline-test-secret",
            "DEEPEYE_EMBEDDING_MODEL":"test-embedding",
            "DEEPEYE_EMBEDDING_BASE_URL":"http://127.0.0.1:1/v1",
            "DEEPEYE_EMBEDDING_API_KEY":"offline-embedding-secret",
        }))
        self.stack.enter_context(patch.object(config_module,"_config_instance",None))
        config_module.Config._instance = None
        self.stack.enter_context(patch.object(socket.socket,"connect",side_effect=AssertionError("network forbidden")))
        self.args = SimpleNamespace(data_root=str(self.data),split="dev",question_ids="13",ids_file=None,
                                    max_samples=1,profile="smoke",run_name="test",workers=1,few_shot=False)

    def close_connections(self):
        from app.db_utils.execution import _get_connection_cache
        cache = _get_connection_cache()
        for connection in cache.values():
            connection.close()
        cache.clear()
        gc.collect()

    def prepare(self):
        out = runner.prepare(self.args)
        runner.load_manifest(self.args)
        return out

    def test_secret_placeholders_and_configuration_roundtrip(self):
        out = self.prepare()
        text = (out/"config.toml").read_text(encoding="utf-8")
        self.assertIn("${DEEPEYE_API_KEY}",text)
        self.assertNotIn("offline-test-secret",text)
        cfg = config_module.get_config()
        self.assertEqual(cfg.schema_linking_config.llm.api_key,"offline-test-secret")
        self.assertEqual(cfg.dataset_config.question_ids,[13])
        self.assertNotIn("offline-test-secret",(out/"config.toml").read_text(encoding="utf-8"))

    def test_missing_environment_fails_without_exposing_values(self):
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaisesRegex(ValueError,"UNSET_DEEPEYE_KEY"):
                config_module._expand_environment({"key":"${UNSET_DEEPEYE_KEY}"})

    def test_subset_before_limit_and_train_support(self):
        cfg = DatasetConfig(type="bird",split="train",root_path=str(self.data),question_ids=[13],max_samples=1)
        dataset = DatasetFactory.get_dataset(cfg)
        self.assertEqual([item.question_id for item in dataset],[13])
        self.assertTrue(Path(dataset[0].database_path).is_file())

    def test_train_ordinal_ids_are_stable_before_filtering(self):
        path = self.data/"train/train.json"
        records = json.loads(path.read_text(encoding="utf-8"))
        for record in records:
            del record["question_id"]
        path.write_text(json.dumps(records),encoding="utf-8")
        self.args.split = "train"
        self.args.question_ids = "1"
        self.prepare()
        runner.load_manifest(self.args,verify_inputs=True)
        cfg = config_module.get_config()
        self.assertEqual([item.question_id for item in DatasetFactory.get_dataset(cfg.dataset_config)],[1])

    def test_qwen_disables_thinking_in_nonstream_requests(self):
        with patch.dict(os.environ,{"DEEPEYE_MODEL":"qwen3.5-35b-a3b"}):
            self.prepare()
            cfg = config_module.get_config()
            self.assertEqual(cfg.schema_linking_config.llm.extra_body,{"enable_thinking":False})

    def test_qwen36_nonstream_configuration_is_frozen_without_network(self):
        with patch.dict(os.environ,{"DEEPEYE_MODEL":"qwen3.6-35b-a3b"}):
            out = self.prepare()
            cfg = config_module.get_config()
            self.assertEqual(cfg.schema_linking_config.llm.extra_body,{"enable_thinking":False})
            manifest = json.loads((out/"manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["model"],"qwen3.6-35b-a3b")

    def test_school_chat_client_bypasses_broken_system_proxy(self):
        from app.llm.llm import LLM
        cfg = SimpleNamespace(api_type="openai",api_key="offline-test-secret",
                              base_url="https://aigc-api.hkust-gz.edu.cn/api/v2",
                              model="Qwen3.6-35B-A3B",temperature=0.7,reasoning_effort=None)
        with patch("app.llm.llm.OpenAI") as create_client:
            LLM(cfg)._create_client()
            self.assertFalse(create_client.call_args.kwargs["http_client"]._trust_env)
            cfg.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            LLM(cfg)._create_client()
            self.assertNotIn("http_client",create_client.call_args.kwargs)

    def test_missing_question_id_is_explicit(self):
        self.args.question_ids = "999"
        with self.assertRaisesRegex(ValueError,"999"):
            self.prepare()

    def test_frozen_run_rejects_changed_inputs_and_model(self):
        self.prepare()
        self.args.question_ids = "7"
        with self.assertRaisesRegex(ValueError,"different inputs"):
            runner.prepare(self.args)
        with patch.dict(os.environ,{"DEEPEYE_MODEL":"different-model"}):
            with self.assertRaisesRegex(ValueError,"changed since prepare"):
                runner.load_manifest(self.args,verify_inputs=True)

    def test_frozen_run_detects_changed_labels(self):
        self.prepare()
        path = self.data/"dev/dev.json"
        labels = json.loads(path.read_text(encoding="utf-8"))
        labels[1]["question"] = "A different question"
        path.write_text(json.dumps(labels),encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"Dataset labels changed"):
            runner.load_manifest(self.args,verify_inputs=True)

    def test_train_few_shot_leakage_guard(self):
        self.args.split = "train"
        self.args.few_shot = True
        with self.assertRaisesRegex(ValueError,"target gold SQL"):
            self.prepare()

    def test_no_credentials_fails_before_network(self):
        with patch.dict(os.environ,{"DEEPEYE_API_KEY":"","DEEPEYE_EMBEDDING_API_KEY":""}):
            with self.assertRaisesRegex(ValueError,"Set credentials"):
                runner.check_credentials()

    def test_unsafe_run_name_rejected(self):
        with self.assertRaises(ValueError):
            runner.run_dir("../outside")

    def test_empty_sql_result_is_executable(self):
        path = self.data/"dev/dev_databases/toy/toy.sqlite"
        result = execute_sql_without_cache(str(path),"SELECT name FROM people WHERE 1=0",timeout=1)
        self.assertEqual(result.result_rows,[])
        self.assertNotEqual(result.result_type,"execution_error")

    def test_full_pipeline_with_mocked_models_and_real_sqlite(self):
        self._check_pipeline(few_shot=False)

    def test_dynamic_few_shot_pipeline_with_mocked_models(self):
        self._check_pipeline(few_shot=True)

    def _check_pipeline(self, few_shot):
        self.args.few_shot = few_shot
        out = self.prepare()
        cfg = config_module.get_config()
        from app.llm import LLM
        from app.services.schema_service import SchemaService
        from app.vector_db import vector_db
        value_module = importlib.import_module("app.pipeline.value_retrieval.value_retrieval")
        index_module = importlib.import_module("app.few_shot.index_builder")
        retriever_module = importlib.import_module("app.few_shot.retriever")
        import numpy as np
        calls = []
        sql = "SELECT name FROM people WHERE city = 'Paris'"

        def ask(model_self,messages,**kwargs):
            text = "\n".join(message["content"] for message in messages)
            calls.append(text)
            if "Mask database-specific names" in text:
                answer = json.dumps({"masked_question":"List <concept> in <value>",
                                     "masked_sql":"SELECT <column> FROM <table> WHERE <column> = <value>"})
            elif "Extract the most useful search terms" in text:
                answer = '<result>["Paris"]</result>'
            elif "pinpoint the specific tables and columns" in text:
                answer = ('<result><table table_name="people"><column column_name="name"/>'
                          '<column column_name="city"/></table></result>')
            else:
                answer = f"<result>{sql}</result>"
            n = kwargs.get("n",1)
            return [SimpleNamespace(content=answer) for _ in range(n)],{
                "prompt_tokens":10*n,"completion_tokens":5*n,"total_tokens":15*n}

        def embed(texts):
            return [np.asarray([1.0,0.0] if "paris" in str(text).lower() else [0.0,1.0],dtype=np.float32)
                    for text in texts]

        with patch.object(LLM,"ask",ask), \
             patch.object(vector_db,"get_embedding_function",return_value=embed), \
             patch.object(value_module,"get_embedding_function",return_value=embed), \
             patch.object(index_module,"get_embedding_function",return_value=embed), \
             patch.object(retriever_module,"get_embedding_function",return_value=embed), \
             patch.object(SchemaService,"_get_encoding",return_value=None):
            stages = [name for name in runner.STAGES if few_shot or "few-shot" not in name]
            for stage in stages:
                with patch.object(sys,"argv",[runner.STAGES[stage]]):
                    runpy.run_path(str(runner.UPSTREAM/"runner"/runner.STAGES[stage]),run_name="__main__")
            selected = load_dataset(cfg.sql_selection_config.save_path)
            self.assertEqual(len(selected),1)
            self.assertEqual(selected[0].final_selected_sql,sql)
            self.assertTrue(selected[0].is_stage_complete("sql_selection"))
            self.assertIn("people",selected[0].final_linked_tables_and_columns)
            self.assertGreater(len(calls),2)
            if few_shot:
                self.assertTrue(selected[0].few_shot_examples)
                self.assertTrue(Path(cfg.few_shot_index_config.save_path,"manifest.json").is_file())
            runner.export_candidates(out)
            runner.evaluate(out)
            metrics = json.loads((out/"metrics.json").read_text(encoding="utf-8"))["summary"]
            self.assertEqual(metrics["selected_ex_multiset"],1.0)
            self.assertEqual(metrics["executable_rate"],1.0)
            # Resume must use checkpoints and make no new model calls.
            before = len(calls)
            for stage in ("link","generate","revise","select"):
                with patch.object(sys,"argv",[runner.STAGES[stage]]):
                    runpy.run_path(str(runner.UPSTREAM/"runner"/runner.STAGES[stage]),run_name="__main__")
            self.assertEqual(len(calls),before)

    def test_set_and_multiset_evaluation_are_distinct(self):
        out = self.prepare()
        cfg = config_module.get_config()
        dataset = DatasetFactory.get_dataset(cfg.dataset_config)
        item = dataset[0]
        item.sql_candidates_after_revision = ["SELECT DISTINCT name FROM people WHERE city='Paris'"]
        item.final_selected_sql = item.sql_candidates_after_revision[0]
        save_dataset(dataset,cfg.sql_selection_config.save_path)
        runner.evaluate(out)
        summary = json.loads((out/"metrics.json").read_text(encoding="utf-8"))["summary"]
        self.assertEqual(summary["selected_ex_set"],1.0)
        self.assertEqual(summary["selected_ex_multiset"],0.0)

    def test_official_llm_split_requests_and_token_accounting(self):
        self.prepare()
        from app.llm import LLM
        cfg = config_module.get_config()
        model = LLM(cfg.schema_linking_config.llm)
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="<result>SQL</result>"))],
            usage=SimpleNamespace(prompt_tokens=10,completion_tokens=3,total_tokens=13))
        with patch.object(model,"_get_client",return_value=client):
            answers, usage = model.ask([{"role":"user","content":"test"}],n=3,timeout=12)
        self.assertEqual(len(answers),3)
        self.assertEqual(usage["total_tokens"],39)
        self.assertEqual(client.chat.completions.create.call_count,3)
        for call in client.chat.completions.create.call_args_list:
            self.assertEqual(call.kwargs["n"],1)
            self.assertEqual(call.kwargs["timeout"],12)

    def test_api_telemetry_has_usage_without_prompt_or_key(self):
        from app.llm.llm import _record_usage
        path = self.root/"usage.jsonl"
        response = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=7,completion_tokens=2,total_tokens=9),
            choices=[SimpleNamespace(finish_reason="stop")])
        with patch.dict(os.environ,{"DEEPEYE_USAGE_PATH":str(path),"DEEPEYE_STAGE":"link"}):
            _record_usage(response,"test-model",0)
        text = path.read_text(encoding="utf-8")
        record = json.loads(text)
        self.assertEqual(record["total_tokens"],9)
        self.assertEqual(record["stage"],"link")
        self.assertNotIn("offline-test-secret",text)
        self.assertNotIn("messages",record)

    def test_rate_limit_does_not_repeat_successful_split_samples(self):
        self.prepare()
        from app.llm import LLM
        import httpx
        from openai import RateLimitError
        model = LLM(config_module.get_config().schema_linking_config.llm)
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="<result>SQL</result>"))],
            usage=SimpleNamespace(prompt_tokens=10,completion_tokens=3,total_tokens=13))
        error = RateLimitError("offline rate limit",response=httpx.Response(429,
                              request=httpx.Request("POST","http://127.0.0.1:1")),body={"code":"insufficient_quota"})
        client = Mock()
        client.chat.completions.create.side_effect = [response,error,response,response]
        with patch.object(model,"_get_client",return_value=client), \
             patch.object(LLM._request_with_retry.retry,"sleep",return_value=None):
            answers,usage = model.ask([{"role":"user","content":"test"}],n=3)
        self.assertEqual(len(answers),3)
        self.assertEqual(client.chat.completions.create.call_count,4)
        self.assertEqual(usage["total_tokens"],39)

    def test_billing_error_stops_all_further_requests_and_is_not_swallowed(self):
        self.prepare()
        import httpx
        from openai import BadRequestError
        from app.llm import LLM
        from app.llm.llm import FatalProviderError
        from app.llm_extractor import LLMExtractor
        model = LLM(config_module.get_config().schema_linking_config.llm)
        client = Mock()
        client.chat.completions.create.side_effect = BadRequestError(
            "billing denied",response=httpx.Response(400,request=httpx.Request("POST","http://127.0.0.1:1")),
            body={"code":"Arrearage"})
        with patch.object(model,"_get_client",return_value=client):
            for _ in range(2):
                with self.assertRaisesRegex(FatalProviderError,"Arrearage"):
                    LLMExtractor(max_retry=5).extract_with_retry(
                        model,[{"role":"user","content":"test"}],lambda text:text,n=12)
        self.assertEqual(client.chat.completions.create.call_count,1)

    def test_sc_baseline_preserves_votes_and_empty_results(self):
        out = self.prepare()
        cfg = config_module.get_config()
        dataset = DatasetFactory.get_dataset(cfg.dataset_config)
        item = dataset[0]
        correct = item.gold_sql
        wrong = "SELECT name FROM people WHERE city='Rome'"
        empty = "SELECT name FROM people WHERE 1=0"
        item.sql_candidates_after_revision = [correct,wrong,wrong,empty]
        item.final_selected_sql = correct
        save_dataset(dataset,cfg.sql_selection_config.save_path)
        runner.evaluate(out)
        result = json.loads((out/"metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(result["summary"]["selected_ex_multiset"],1)
        self.assertEqual(result["summary"]["sc_ex_multiset"],0)
        self.assertEqual(result["summary"]["executable_rate"],1)
        self.assertEqual(result["per_question"][0]["sc_sql_multiset"],wrong)
        self.assertEqual(result["per_question"][0]["sc_vote_share_multiset"],0.5)

    def test_pairwise_selection_uses_execution_feedback(self):
        self.prepare()
        cfg = config_module.get_config()
        from app.llm import LLM
        from app.pipeline import SQLSelectionRunner
        dataset = DatasetFactory.get_dataset(cfg.dataset_config)
        item = dataset[0]
        correct = "SELECT name FROM people WHERE city = 'Paris'"
        wrong = "SELECT name FROM people WHERE city = 'Rome'"
        item.database_schema_after_schema_linking = item.database_schema
        item.sql_candidates_after_revision = [wrong,correct]
        item.total_time = 0.0
        item.total_llm_cost = {"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}
        save_dataset(dataset,cfg.sql_revision_config.save_path)
        prompts = []

        def ask(model_self,messages,**kwargs):
            text = messages[0]["content"]
            prompts.append(text)
            candidate_a = text.split("SQL Candidate A:")[-1].split("## Execution Result:")[0]
            vote = "A" if "Paris" in candidate_a else "B"
            return [SimpleNamespace(content=f"<result>{vote}</result>")],{
                "prompt_tokens":5,"completion_tokens":1,"total_tokens":6}

        with patch.object(LLM,"ask",ask):
            SQLSelectionRunner.from_config(cfg).run()
        selected = load_dataset(cfg.sql_selection_config.save_path)[0]
        self.assertEqual(selected.final_selected_sql,correct)
        self.assertTrue(prompts)
        self.assertIn("Execution Result",prompts[0])
        self.assertIn("Ada",prompts[0])


if __name__ == "__main__":
    unittest.main()
