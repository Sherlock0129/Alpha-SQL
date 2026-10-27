"""Run four independent action probes from one frozen SQL candidate state.

This module performs external model calls when ``probe-actions`` is invoked.
It is deliberately separate from the offline label and split commands.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from alphasql.algorithm.compiler.validation import validate_sql
from alphasql.algorithm.selection.dual_calibration import LogisticHead
from alphasql.algorithm.selection.dual_confidence import JevPoolJudge
from alphasql.algorithm.selection.generation_feedback import GenerationFeedback
from alphasql.algorithm.selection.training_data import generator_identity, pool_id, state_id
from alphasql.database.schema import TableSchema
from alphasql.jev.client import JevClient
from alphasql.jev.models import NoulQuestion, NoulResult, Usage
from alphasql.jev.runtime import JevSettings
from alphasql.runner.new_selector import DirectLLMGenerator, _generator_provenance
from alphasql.runner.task import Task


def _best_initial(state: Mapping[str, Any]) -> Mapping[str, Any]:
    candidates = state["initial_candidates"]
    if not candidates:
        raise ValueError("Action probes require a nonempty initial pool")
    valid = [item for item in candidates if item.get("execution", {}).get("valid")]
    source = valid or candidates
    return max(source, key=lambda item: float(item.get("candidate_probability", 0.0)))


def _feedback(state: Mapping[str, Any], action: str) -> GenerationFeedback:
    anchor = _best_initial(state)
    initial = state["initial_candidates"]
    if action == "revise":
        # The six raw JEV answers are retained in the immutable round snapshot.
        first = state.get("initial_jev", {})
        dimension = max(("projection", "filters", "logic"),
                        key=lambda name: float(first.get(f"gap_{name}") or 0.0))
    else:
        dimension = "schema" if action == "relink" else "uncertain"
    return GenerationFeedback(action, dimension,
                              anchor.get("canonical_sql") or anchor["sql"],
                              tuple(anchor.get("selected_tables") or ()),
                              tuple(item.get("canonical_sql") or item["sql"]
                                    for item in initial[-3:]))


def _task(state: Mapping[str, Any]) -> Task:
    schema = {name: TableSchema.from_table_schema_dict(table)
              for name, table in state.get("generation_schema", {}).items()}
    if not schema:
        raise ValueError("Action probe requires recorded generation_schema")
    return Task(question_id=int(state["question_id"]), db_id=state["db_id"],
                question=state["question"], evidence=state["evidence"],
                sql=None, table_schema_dict=schema)


def _adjudicate(state: Mapping[str, Any], client: JevClient) -> tuple[str | None, Usage]:
    valid = [item for item in state["initial_candidates"]
             if item.get("execution", {}).get("valid")]
    if not valid:
        return None, Usage()
    if len(valid) == 1:
        return valid[0]["sql"], Usage()
    top = sorted(valid, key=lambda item: -float(item.get("candidate_probability", 0)))[:4]
    questions = {f"prefer_{i}_{j}": NoulQuestion(
        f"Which SQL fully satisfies the question more plausibly? Is SQL {i} "
        f"more likely correct than SQL {j}? Compare projection, predicates, "
        "joins, aggregation and result multiplicity.")
        for i in range(len(top)) for j in range(i + 1, len(top))}
    response = client.ask({"question": state["question"], "evidence": state["evidence"],
                           "schema": state["full_schema"],
                           "candidates": [{"index": i, "sql": item["sql"],
                                           "execution": item["execution"]}
                                          for i, item in enumerate(top)]}, questions)
    scores = [0.0] * len(top)
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            preference = response.require(f"prefer_{i}_{j}", NoulResult).noul
            scores[i] += preference
            scores[j] += 1 - preference
    return top[max(range(len(top)), key=lambda i: (scores[i], -i))]["sql"], response.usage


def _llm_usage(path: Path) -> dict[str, Any]:
    usage: dict[str, Any] = {"input_tokens": 0, "output_tokens": 0,
                             "model_calls": 0, "tokens_complete": True}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            usage["model_calls"] += 1
            if item.get("prompt_tokens") is None or item.get("completion_tokens") is None:
                usage["tokens_complete"] = False
            usage["input_tokens"] += int(item.get("prompt_tokens") or 0)
            usage["output_tokens"] += int(item.get("completion_tokens") or 0)
    return usage


def probe_action(state: Mapping[str, Any], action: str, db_root: str,
                 output_dir: str | Path, model: str | None = None) -> dict[str, Any]:
    """One independent arm; every invocation resets the ordinary LLM call cap."""
    if action not in {"relink", "revise", "diversify", "adjudicate"}:
        raise ValueError(f"Unknown action: {action}")
    budget = state["budget"]
    max_calls = int(budget["max_model_calls"])
    if max_calls < 1:
        raise ValueError("At least one model call must be permitted")
    expected_pool = pool_id(state["db_id"], state["question_id"],
                            state["initial_candidates"], generator_identity(state["generator"]))
    if expected_pool != state["initial_pool_id"]:
        raise ValueError("Action input no longer matches its frozen initial pool")
    expected_state = state_id(expected_pool, 0, state["initial_jev"],
                              state["initial_candidates"], state["full_schema"],
                              state["generation_schema"], state.get("judge", {}),
                              state.get("calibrator", {}))
    if expected_state != state["state_id"]:
        raise ValueError("Action input no longer matches its frozen JEV/schema state")
    work = Path(output_dir) / str(state["state_id"]) / action
    work.mkdir(parents=True, exist_ok=True)
    usage_path = work / "llm_usage.jsonl"
    usage_path.unlink(missing_ok=True)
    settings = JevSettings.from_environment()
    settings.require_api_key()
    judge_info = state.get("judge", {})
    if judge_info.get("model") and settings.model != judge_info["model"]:
        raise ValueError("JEV model differs from the frozen initial judge")
    if judge_info.get("questions_sha256"):
        questions_path = Path(__file__).parents[1] / "algorithm" / "selection" / "dual_confidence.py"
        if hashlib.sha256(questions_path.read_bytes()).hexdigest() != judge_info["questions_sha256"]:
            raise ValueError("JEV questions changed since the initial pool was judged")
    client = JevClient(settings)
    added: list[dict[str, Any]] = []
    previous_usage = os.environ.get("ALPHASQL_USAGE_PATH")
    previous_limit = os.environ.get("LLM_MAX_REQUESTS")
    try:
        if action == "adjudicate":
            jev_called = sum(bool(item.get("execution", {}).get("valid"))
                             for item in state["initial_candidates"]) > 1
            selected_sql, jev_usage = _adjudicate(state, client)
        else:
            from alphasql.llm_call.openai_llm import reset_request_counter
            config = state["generator"]
            if config["kind"] != "DirectLLMGenerator":
                raise ValueError("Counterfactual generation requires a DirectLLMGenerator state")
            if model is not None and model != config["model"]:
                raise ValueError("Action model must match the frozen generator")
            generator = DirectLLMGenerator(db_root, work, config["model"],
                                           int(config["samples"]),
                                           float(config["temperature"]),
                                           tuple(config["prompt_routes"]),
                                           int(config["path_samples"]),
                                           config.get("base_url"),
                                           config.get("api_key_env"),
                                           config.get("extra_body"))
            if _generator_provenance(generator)["source_sha256"] != config["source_sha256"]:
                raise ValueError("Generator code or prompt templates changed since pool creation")
            if max_calls < 2:
                raise ValueError("Generation probes need one call reserved for the final JEV judge")
            os.environ["ALPHASQL_USAGE_PATH"] = str(usage_path.resolve())
            os.environ["LLM_MAX_REQUESTS"] = str(max_calls - 1)
            reset_request_counter()
            generated = generator.generate(_task(state), 1, _feedback(state, action))
            db_id = state["db_id"]
            db_path = str(Path(db_root) / db_id / f"{db_id}.sqlite")
            existing = {item.get("canonical_sql") or item["sql"]
                        for item in state["initial_candidates"]}
            for item in generated:
                if len(added) >= int(budget["max_new_candidates"]):
                    break
                if item.sql in existing:
                    continue
                execution = validate_sql(db_path, item.sql)
                added.append({"sql": item.sql, "route": item.route,
                              "selected_tables": list(item.selected_tables),
                              "selected_columns": list(item.selected_columns),
                              "execution": execution.summary()})
                existing.add(item.sql)
            combined = list(state["initial_candidates"]) + added
            judgement = JevPoolJudge(client, max_candidates=max(32, len(combined))).assess(
                state["question"], state["evidence"], state["full_schema"], combined)
            jev_called = bool(any(item["execution"]["valid"] for item in combined))
            jev_usage = judgement.usage
            valid = [i for i, item in enumerate(combined) if item["execution"]["valid"]]
            calibrator_info = state.get("calibrator", {})
            if calibrator_info.get("kind") == "dual_logistic":
                candidate_head = LogisticHead(**calibrator_info["model"]["candidate"])
                scores = [candidate_head.predict(value) for value in judgement.candidate_raw]
            else:
                scores = list(judgement.candidate_raw)
            selected_index = max(valid, key=lambda i: (scores[i], -i))\
                if valid else None
            selected_sql = combined[selected_index]["sql"] if selected_index is not None else None
    finally:
        if previous_usage is None:
            os.environ.pop("ALPHASQL_USAGE_PATH", None)
        else:
            os.environ["ALPHASQL_USAGE_PATH"] = previous_usage
        if previous_limit is None:
            os.environ.pop("LLM_MAX_REQUESTS", None)
        else:
            os.environ["LLM_MAX_REQUESTS"] = previous_limit
    usage = _llm_usage(usage_path)
    usage["input_tokens"] += jev_usage.input_tokens
    usage["output_tokens"] += jev_usage.output_tokens
    usage["model_calls"] += int(jev_called)
    if jev_called and jev_usage.input_tokens == 0 and jev_usage.output_tokens == 0:
        usage["tokens_complete"] = False
    if usage["model_calls"] > max_calls:
        raise ValueError("Probe exceeded the shared model-call budget")
    if "max_tokens" in budget and usage["input_tokens"] + usage["output_tokens"] > int(
            budget["max_tokens"]):
        raise ValueError("Probe exceeded the shared token budget")
    return {"state_id": state["state_id"], "initial_pool_id": state["initial_pool_id"],
            "question_id": state["question_id"], "db_id": state["db_id"],
            "action": action, "budget": dict(budget), "new_candidates": added,
            "selected_sql": selected_sql, "usage": usage}
