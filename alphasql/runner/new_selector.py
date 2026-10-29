"""Adaptive direct-SQL generation with separate Jev pool/candidate judgements.

Commands:
  run    Generate or reveal SQL candidates, execute, judge, and select.
  label  Build offline execution labels from a train/development split.
  train  Fit two calibrators with question-level easy-to-hard pacing.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import copy
from typing import Any, Mapping, Protocol, Sequence

import sqlglot

from alphasql.algorithm.compiler.validation import validate_sql
from alphasql.algorithm.selection.dual_calibration import DualCalibrator, fit_dual
from alphasql.algorithm.selection.selective_stop import StopPolicy
from alphasql.algorithm.selection.dual_confidence import JevPoolJudge, PoolJudgement
from alphasql.algorithm.selection.generation_feedback import GenerationFeedback, choose_feedback
from alphasql.algorithm.selection.schema_signal import result_signature
from alphasql.database.database_manager import DatabaseManager
from alphasql.jev.client import JevClient
from alphasql.jev.models import Usage
from alphasql.jev.runtime import JevSettings
from alphasql.runner.task import Task


class CandidateGenerator(Protocol):
    def generate(self, task: Task, round_index: int,
                 feedback: GenerationFeedback | None = None) -> Sequence["GeneratedSQL" | str]: ...


@dataclass(frozen=True)
class GeneratedSQL:
    sql: str
    route: str
    selected_tables: tuple[str, ...] = ()
    selected_columns: tuple[str, ...] = ()


def _path_candidates(paths: Any, route: str | None = None) -> list[GeneratedSQL]:
    sqls: list[GeneratedSQL] = []
    for path in paths:
        if not path:
            continue
        node = path[-1]
        sql = (getattr(node, "final_sql_query", None)
               or getattr(node, "revised_sql_query", None)
               or getattr(node, "sql_query", None))
        if isinstance(sql, str) and sql.strip():
            selected_schema = getattr(node, "selected_schema_dict", None) or {}
            selected = tuple(selected_schema.keys())
            columns = tuple(f"{table}.{column}" for table, item in selected_schema.items()
                            for column in (item.get("columns", {}) if isinstance(item, Mapping)
                                           else getattr(item, "columns", {})))
            sqls.append(GeneratedSQL(sql.strip(), route or
                getattr(node, "generation_route", None) or (
                "linked_schema" if selected else "full_schema"), selected, columns))
    return sqls


class FrozenPoolGenerator:
    """Reveal a fixed, previously generated candidate pool in bounded rounds."""

    def __init__(self, candidate_dir: str | Path, per_round: int = 8) -> None:
        if per_round < 1:
            raise ValueError("per_round must be positive")
        self.candidate_dir = Path(candidate_dir)
        self.per_round = per_round
        self.supports_feedback = False

    def generate(self, task: Task, round_index: int,
                 feedback: GenerationFeedback | None = None) -> Sequence[GeneratedSQL]:
        path = self.candidate_dir / f"{task.question_id}.pkl"
        with path.open("rb") as handle:
            saved = pickle.load(handle)
        sqls = saved if saved and all(isinstance(item, GeneratedSQL) for item in saved) else _path_candidates(saved)
        start = round_index * self.per_round
        return sqls[start:start + self.per_round]


class DirectLLMGenerator:
    """Use D&C, skeleton and query-plan prompts with verifier-guided revisions."""

    def __init__(self, db_root: str, work_dir: str | Path, model: str,
                 samples: int = 3, temperature: float = 0.8,
                 prompt_routes: Sequence[str] = ("skeleton", "query_plan"),
                 path_samples: int = 1, base_url: str | None = None,
                 api_key_env: str | None = None,
                 extra_body: Mapping[str, Any] | None = None,
                 direct_connection: bool = False) -> None:
        from alphasql.runner.prompt_paths import PROMPT_ROUTES
        if samples < 1 or path_samples < 1:
            raise ValueError("Sample counts must be positive")
        if set(prompt_routes) - set(PROMPT_ROUTES):
            raise ValueError(f"Unknown prompt routes: {set(prompt_routes) - set(PROMPT_ROUTES)}")
        if bool(base_url) != bool(api_key_env):
            raise ValueError("Explicit LLM endpoint and API key variable must be supplied together")
        self.db_root, self.work_dir, self.model = db_root, Path(work_dir), model
        self.samples, self.temperature = samples, temperature
        self.prompt_routes, self.path_samples = tuple(prompt_routes), path_samples
        self.base_url, self.api_key_env = base_url, api_key_env
        self.extra_body = dict(extra_body) if extra_body else None
        self.direct_connection = direct_connection
        self.supports_feedback = True

    def generate(self, task: Task, round_index: int,
                 feedback: GenerationFeedback | None = None) -> Sequence[GeneratedSQL]:
        if not task.table_schema_dict:
            raise ValueError("Direct generation needs a preprocessed Task with table_schema_dict")
        from alphasql.llm_call.openai_llm import reset_request_counter
        reset_request_counter()
        from alphasql.algorithm.mcts.mcts_action import MCTSNodeType
        from alphasql.runner.direct_candidate_runner import (
            _root_node, deduplicate_paths, generate_one,
        )
        from alphasql.runner.prompt_paths import generate_prompt_paths
        round_dir = self.work_dir / str(task.question_id) / f"round_{round_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        kwargs = {"model": self.model, "n": 1, "top_p": 0.95,
                  "max_tokens": 4096, "temperature": self.temperature,
                  "n_strategy": "multiple"}
        if self.api_key_env:
            key = os.environ.get(self.api_key_env)
            if not key:
                raise ValueError(f"{self.api_key_env} is required for {self.base_url}; no LLM call was made")
            kwargs.update(api_key=key, base_url=self.base_url)
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        if self.direct_connection:
            kwargs["direct_connection"] = True
        if feedback:
            (round_dir / "feedback.json").write_text(
                json.dumps(feedback.to_dict(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
        if feedback and feedback.action in {"relink", "revise"}:
            return self._feedback_generation(task, round_dir, kwargs, feedback)
        hint = task.evidence
        if feedback:
            hint = (hint + "\n\n" + feedback.hint()).strip()
        prompt_task = task.model_copy(update={"sql": None, "evidence": hint})
        path = round_dir / f"{task.question_id}.pkl"
        path.unlink(missing_ok=True)
        summary = generate_one((prompt_task, self.db_root, str(round_dir),
                                kwargs, self.samples))
        if path.exists():
            with path.open("rb") as handle:
                paths = pickle.load(handle)
        else:
            paths = []
        warnings = list(summary["warnings"])
        if self.prompt_routes:
            root = _root_node(prompt_task, self.db_root)
            linked = next((node for candidate_path in paths for node in candidate_path
                           if node.node_type == MCTSNodeType.SCHEMA_SELECTION), root)
            for route in self.prompt_routes:
                parent = linked if route == "skeleton" else root
                new_paths, route_warnings = generate_prompt_paths(
                    parent, kwargs, (route,), self.path_samples)
                paths.extend(new_paths)
                warnings.extend(route_warnings)
        paths = deduplicate_paths(paths)
        if not paths:
            self._save_warnings(round_dir, warnings)
            raise RuntimeError(f"SQL generation failed: {warnings}")
        with path.open("wb") as handle:
            pickle.dump(paths, handle)
        self._save_warnings(round_dir, warnings)
        return _path_candidates(paths)

    @staticmethod
    def _save_warnings(round_dir: Path, warnings: Sequence[str]) -> None:
        (round_dir / "warnings.json").write_text(
            json.dumps(list(warnings), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")

    def _feedback_generation(self, task: Task, round_dir: Path,
                             kwargs: dict[str, Any],
                             feedback: GenerationFeedback) -> Sequence[GeneratedSQL]:
        import copy
        import alphasql.algorithm.mcts.mcts_action as actions
        from alphasql.algorithm.mcts.mcts_action import (
            MCTSNodeType, SQLGenerationAction, SQLRevisionAction, SchemaSelectionAction,
        )
        from alphasql.database.utils import build_table_ddl_statement
        from alphasql.llm_call.runtime import configure_environment
        from alphasql.runner.direct_candidate_runner import (
            _root_node, deduplicate_paths, terminalize,
        )
        from alphasql.runner.prompt_paths import REVISION_ROUTE, generate_prompt_paths

        configure_environment()
        actions.SQL_GENERATION_LLM_KWARGS_N = self.samples
        actions.SQL_REVISION_LLM_KWARGS_N = self.samples
        root = _root_node(task.model_copy(update={"sql": None}), self.db_root)
        root.hint = (root.hint + "\n\n" + feedback.hint()).strip()
        paths = []
        warnings = []
        if feedback.action == "relink":
            try:
                linked_nodes = SchemaSelectionAction().create_children_nodes(root, kwargs)
            except Exception as error:
                warnings.append(f"schema_selection: {type(error).__name__}: {error}")
                linked_nodes = [root]
            for linked in linked_nodes:
                try:
                    for node in SQLGenerationAction().create_children_nodes(linked, kwargs):
                        node.generation_route = "relink"
                        paths.append(terminalize(node))
                except Exception as error:
                    warnings.append(f"relink_generation: {type(error).__name__}: {error}")
                if self.prompt_routes:
                    new_paths, route_warnings = generate_prompt_paths(
                        linked, kwargs, self.prompt_routes, self.path_samples)
                    paths.extend(new_paths)
                    warnings.extend(route_warnings)
        else:
            if not feedback.anchor_sql:
                raise ValueError("SQL revision needs an anchor SQL")
            anchor = copy.copy(root)
            anchor.node_type = MCTSNodeType.SQL_GENERATION
            anchor.parent_node = root
            anchor.parent_action = SQLGenerationAction()
            anchor.depth = root.depth + 1
            anchor.sql_query = feedback.anchor_sql
            anchor.path_nodes = root.path_nodes + [anchor]
            selected = {name: task.table_schema_dict[name]
                        for name in feedback.anchor_schema_tables
                        if name in task.table_schema_dict}
            if selected:
                anchor.selected_schema_dict = selected
                anchor.selected_schema_context = "\n".join(
                    build_table_ddl_statement(
                        table.to_dict(), add_value_description=True,
                        add_column_description=True, add_value_examples=True,
                        add_expanded_column_name=True)
                    for table in selected.values())
            try:
                for node in SQLRevisionAction().create_children_nodes(anchor, kwargs):
                    node.generation_route = "revise"
                    paths.append(terminalize(node))
            except Exception as error:
                warnings.append(f"sql_revision: {type(error).__name__}: {error}")
            if self.prompt_routes:
                new_paths, route_warnings = generate_prompt_paths(
                    anchor, kwargs, (REVISION_ROUTE,), self.path_samples,
                    feedback.anchor_sql)
                paths.extend(new_paths)
                warnings.extend(route_warnings)
        paths = deduplicate_paths(paths)
        if not paths:
            self._save_warnings(round_dir, warnings)
            raise RuntimeError(f"{feedback.action} produced no SQL candidates: {warnings}")
        with (round_dir / f"{task.question_id}.pkl").open("wb") as handle:
            pickle.dump(paths, handle)
        self._save_warnings(round_dir, warnings)
        return _path_candidates(paths)


def _canonical(sql: str) -> str:
    try:
        return sqlglot.parse_one(sql, read="sqlite").sql(
            dialect="sqlite", normalize=True, pretty=False, comments=False)
    except Exception:
        return " ".join(sql.split())


def _schema_context(db_id: str, db_root: str) -> dict[str, Any]:
    schema = DatabaseManager.get_database_schema(db_id, db_root)
    return {"db_id": db_id, "tables": {
        table_name: {"columns": {
            column_name: {"type": column.column_type,
                          "description": column.column_description,
                          "foreign_keys": column.foreign_keys}
            for column_name, column in table.columns.items()}}
        for table_name, table in schema.tables.items()
    }}


def _generator_provenance(generator: CandidateGenerator,
                          version: str | None = None) -> dict[str, Any]:
    config: dict[str, Any] = {"kind": type(generator).__name__}
    if isinstance(generator, DirectLLMGenerator):
        config.update(model=generator.model, samples=generator.samples,
                      temperature=generator.temperature,
                      prompt_routes=list(generator.prompt_routes),
                      path_samples=generator.path_samples,
                      base_url=generator.base_url,
                      api_key_env=generator.api_key_env,
                      extra_body=generator.extra_body,
                      direct_connection=generator.direct_connection)
    elif isinstance(generator, FrozenPoolGenerator):
        config.update(candidate_dir=str(generator.candidate_dir.resolve()),
                      per_round=generator.per_round)
        manifest_path = generator.candidate_dir.parent / "manifest.json"
        if manifest_path.is_file():
            config["source_generator"] = json.loads(
                manifest_path.read_text(encoding="utf-8"))["generator"]
    source_paths = [Path(__file__), Path(__file__).with_name("prompt_paths.py"),
                    Path(__file__).with_name("direct_candidate_runner.py"),
                    Path(__file__).parents[1] / "algorithm" / "mcts" / "mcts_action.py",
                    Path(__file__).parents[1] / "llm_call" / "openai_llm.py",
                    Path(__file__).parents[1] / "llm_call" / "prompt_factory.py"]
    source_paths.extend(sorted((Path(__file__).parents[1] / "templates").glob("*.txt")))
    digest = hashlib.sha256()
    for path in source_paths:
        if path.is_file():
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
    config["source_sha256"] = digest.hexdigest()
    config["version"] = version or "sha256:" + hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()
    return config


def _judge_provenance(judge: JevPoolJudge) -> dict[str, Any]:
    settings = getattr(getattr(judge, "client", None), "settings", None)
    path = Path(__file__).parents[1] / "algorithm" / "selection" / "dual_confidence.py"
    return {"kind": type(judge).__name__,
            "model": getattr(settings, "model", None),
            "questions_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@dataclass(frozen=True)
class SelectorSettings:
    max_rounds: int = 3
    max_candidates: int = 32
    coverage_stop: float = 0.8
    candidate_stop: float = 0.8
    execution_timeout: int = 60
    max_jev_calls: int | None = None

    def __post_init__(self) -> None:
        if self.max_rounds < 1 or self.max_candidates < 1 or self.execution_timeout < 1:
            raise ValueError("Round, candidate and timeout limits must be positive")
        if self.max_jev_calls is not None and self.max_jev_calls < 1:
            raise ValueError("JEV call limit must be positive")
        if not 0 <= self.coverage_stop <= 1 or not 0 <= self.candidate_stop <= 1:
            raise ValueError("Stop thresholds must be probabilities")


class AdaptiveSQLSelector:
    def __init__(self, generator: CandidateGenerator, judge: JevPoolJudge,
                 settings: SelectorSettings | None = None,
                 calibrator: DualCalibrator | None = None,
                 generator_version: str | None = None,
                 stop_policy: StopPolicy | None = None,
                 content_verifier: Any | None = None,
                 verifier_sha256: str | None = None) -> None:
        self.generator, self.judge = generator, judge
        self.settings = settings or SelectorSettings()
        self.calibrator = calibrator
        self.stop_policy = stop_policy
        self.content_verifier = content_verifier
        if stop_policy and stop_policy.score_key == "verifier_probability":
            if content_verifier is None or stop_policy.model_sha256 != verifier_sha256:
                raise ValueError("Verifier stop policy requires its exact frozen model")
        if stop_policy and stop_policy.score_key == "candidate_probability" and calibrator is None:
            raise ValueError("Calibrated stop policy requires its candidate calibrator")
        self.generator_provenance = _generator_provenance(generator, generator_version)
        self.judge_provenance = _judge_provenance(judge)
        if stop_policy and stop_policy.generator_version and (
                self.generator_provenance.get("version") != stop_policy.generator_version):
            raise ValueError("Stop policy generator version differs from the current generator")
        if stop_policy and stop_policy.judge_model and (
                self.judge_provenance.get("model") not in
                {stop_policy.judge_model, "jev-latest"}):
            raise ValueError("Stop policy judge model differs from the current JEV model")
        self.calibrator_provenance = ({"kind": "dual_logistic",
            "sha256": hashlib.sha256(json.dumps(calibrator.to_dict(),
                                              sort_keys=True).encode("utf-8")).hexdigest(),
            "model": calibrator.to_dict()}
            if calibrator else {"kind": "raw_jev"})
        if self.settings.max_candidates > judge.max_candidates:
            raise ValueError("Selector candidate limit exceeds Jev judge limit")

    def run(self, task: Task, db_root: str) -> dict[str, Any]:
        db_path = Path(db_root) / task.db_id / f"{task.db_id}.sqlite"
        if not db_path.is_file():
            raise FileNotFoundError(db_path)
        schema = _schema_context(task.db_id, db_root)
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        rounds: list[dict[str, Any]] = []
        last: PoolJudgement | None = None
        feedback: GenerationFeedback | None = None
        usage_total = {"input_tokens": 0, "output_tokens": 0}
        jev_calls_used = 0
        accepted_stop = False
        generation_task = task.model_copy(update={"sql": None})
        for round_index in range(self.settings.max_rounds):
            applied_feedback = (feedback if getattr(self.generator, "supports_feedback", True)
                                else None)
            generated = self.generator.generate(generation_task, round_index,
                                                applied_feedback)
            new_count = 0
            for generated_sql in generated:
                item = (generated_sql if isinstance(generated_sql, GeneratedSQL)
                        else GeneratedSQL(generated_sql, "external"))
                sql = item.sql
                canonical = _canonical(sql)
                if not canonical or canonical in seen:
                    continue
                if len(candidates) >= self.settings.max_candidates:
                    break
                seen.add(canonical)
                execution = validate_sql(str(db_path), sql, self.settings.execution_timeout)
                candidates.append({"sql": sql, "canonical_sql": canonical,
                                   "source_round": round_index,
                                   "route": item.route,
                                   "selected_tables": list(item.selected_tables),
                                   "selected_columns": list(item.selected_columns),
                                   "execution": execution.summary()})
                new_count += 1
            if candidates:
                last = self.judge.assess(task.question, task.evidence, schema, candidates)
                if any(item["execution"]["valid"] for item in candidates):
                    jev_calls_used += 1
                if (self.stop_policy and self.stop_policy.judge_model and last.model
                        and last.model != self.stop_policy.judge_model):
                    raise ValueError("Actual JEV response model differs from the stop policy")
                usage_total["input_tokens"] += last.usage.input_tokens
                usage_total["output_tokens"] += last.usage.output_tokens
                has_valid = any(item["execution"]["valid"] for item in candidates)
                coverage = (self.calibrator.coverage.predict(last.coverage_raw)
                            if self.calibrator and has_valid else last.coverage_raw)
                candidate_probs = [
                    (self.calibrator.candidate.predict(raw) if self.calibrator else raw)
                    if candidates[i]["execution"]["valid"] else 0.0
                    for i, raw in enumerate(last.candidate_raw)
                ]
            else:
                coverage, candidate_probs = 0.0, []
            best = max(candidate_probs, default=0.0)
            if self.content_verifier:
                from alphasql.algorithm.selection.content_verifier import content_features
                valid_indices = [i for i, item in enumerate(candidates)
                                 if item["execution"]["valid"]]
                verifier_probs = [0.0] * len(candidates)
                features = [{"features": content_features(
                    task.question, task.evidence, schema,
                    candidates[i]["sql"], candidates[i]["execution"])} for i in valid_indices]
                for i, score in zip(valid_indices, self.content_verifier.predict(features)):
                    verifier_probs[i] = float(score)
            else:
                verifier_probs = []
            snapshot = [dict(copy.deepcopy(item), candidate_raw=last.candidate_raw[i],
                             candidate_probability=candidate_probs[i],
                             **({"verifier_probability": verifier_probs[i]}
                                if verifier_probs else {}))
                        for i, item in enumerate(candidates)] if last else []
            round_record = {"round": round_index, "new_candidates": new_count,
                            "pool_size": len(candidates), "coverage": coverage,
                            "best_candidate_probability": best,
                            "best_verifier_probability": max(verifier_probs, default=None),
                            "question": task.question, "evidence": task.evidence,
                            "schema_hypotheses": [
                                {"candidate_index": i,
                                 "tables": list(item.get("selected_tables", [])),
                                 "columns": list(item.get("selected_columns", [])),
                                 "scope": "linked" if item.get("selected_tables") else "full"}
                                for i, item in enumerate(candidates)],
                            "candidates": snapshot,
                            "new_candidate_indices": [i for i, item in enumerate(candidates)
                                                      if item["source_round"] == round_index],
                            "jev": {"called": bool(last and any(
                                item["execution"]["valid"] for item in candidates)),
                                    "model": last.model if last else "",
                                    "pool_covered": last.coverage_raw if last else 0.0,
                                    "candidate": list(last.candidate_raw) if last else [],
                                    "gap_schema": last.diagnostics.get("schema") if last else None,
                                    "gap_projection": last.diagnostics.get("projection") if last else None,
                                    "gap_filters": last.diagnostics.get("filters") if last else None,
                                    "gap_logic": last.diagnostics.get("logic") if last else None,
                                    "usage": {"input_tokens": last.usage.input_tokens,
                                              "output_tokens": last.usage.output_tokens}
                                    if last else {"input_tokens": 0, "output_tokens": 0}},
                            "generation_action": applied_feedback.action if applied_feedback
                            else "initial" if round_index == 0 else "fixed_pool",
                            "feedback_applied_detail": applied_feedback.to_dict()
                            if applied_feedback else None,
                            "feedback_applied": applied_feedback is not None,
                            "diagnostics": dict(last.diagnostics) if last else {}}
            rounds.append(round_record)
            if self.stop_policy:
                raw_best = max((last.candidate_raw[i] for i, item in enumerate(candidates)
                                if item["execution"]["valid"]), default=0.0) if last else 0.0
                stop_score = (raw_best if self.stop_policy.score_key == "candidate_raw"
                              else max(verifier_probs, default=0.0)
                              if self.stop_policy.score_key == "verifier_probability" else best)
                pool_score = last.coverage_raw if last else 0.0
                round_record["stop_score"] = stop_score
                if self.stop_policy.accepts(stop_score, pool_score):
                    round_record["decision_action"] = "stop"
                    accepted_stop = True
                    break
                if (self.stop_policy.pool_review_below is not None
                        and pool_score < self.stop_policy.pool_review_below
                        and any(item["execution"]["valid"] for item in candidates)):
                    if jev_calls_used >= (self.settings.max_jev_calls or self.settings.max_rounds):
                        round_record["decision_action"] = "budget_exhausted"
                        break
                    if not hasattr(self.judge, "adjudicate"):
                        raise ValueError("Stop policy requires a judge with adjudicate support")
                    review = self.judge.adjudicate(task.question, task.evidence, schema,
                                                   candidates, verifier_probs or candidate_probs)
                    review_usage = review.pop("usage")
                    usage_total["input_tokens"] += review_usage.input_tokens
                    usage_total["output_tokens"] += review_usage.output_tokens
                    jev_calls_used += int(review.get("called", False))
                    round_record["adjudication"] = review
                    round_record["decision_action"] = "adjudicate"
                    break
                if jev_calls_used >= (self.settings.max_jev_calls or self.settings.max_rounds):
                    round_record["decision_action"] = "budget_exhausted"
                    break
                round_record["decision_action"] = "continue"
            elif (coverage >= self.settings.coverage_stop
                  and best >= self.settings.candidate_stop):
                round_record["decision_action"] = "stop_legacy"
                accepted_stop = True
                break
            if len(candidates) >= self.settings.max_candidates:
                round_record["decision_action"] = "budget_exhausted"
                break
            if round_index + 1 < self.settings.max_rounds:
                feedback = (choose_feedback(last, candidates,
                                            candidate_scores=candidate_probs)
                            if last else GenerationFeedback("diversify", "uncertain", None))
                round_record["feedback_next"] = feedback.to_dict()
            else:
                round_record["decision_action"] = "budget_exhausted"
        if last is None:
            last = PoolJudgement(0.0, (), Usage())
        valid = [i for i, item in enumerate(candidates) if item["execution"]["valid"]]
        for i, item in enumerate(candidates):
            raw = last.candidate_raw[i]
            item["candidate_raw"] = raw
            item["candidate_probability"] = (
                self.calibrator.candidate.predict(raw) if self.calibrator and i in valid
                else raw if i in valid else 0.0)
            if self.content_verifier:
                item["verifier_probability"] = rounds[-1]["candidates"][i]["verifier_probability"]
        selection_key = ("verifier_probability" if self.stop_policy and
                         self.stop_policy.score_key == "verifier_probability" else
                         "candidate_probability")
        selected = max(valid, key=lambda i: (candidates[i][selection_key], -i)) if valid else None
        return {
            "schema_version": 2, "question_id": task.question_id, "db_id": task.db_id,
            "question": task.question, "evidence": task.evidence,
            "full_schema": schema,
            "generation_schema": {
                name: table.to_dict() if hasattr(table, "to_dict") else table
                for name, table in (task.table_schema_dict or {}).items()},
            "generator": self.generator_provenance,
            "judge": self.judge_provenance,
            "calibrator": self.calibrator_provenance,
            "stop_policy": self.stop_policy.__dict__ if self.stop_policy else None,
            "selected_index": selected,
            "selected_sql": candidates[selected]["sql"] if selected is not None else None,
            "selection_score_key": selection_key,
            "accepted_stop": accepted_stop,
            "decision_action": rounds[-1].get("decision_action"),
            "coverage_raw": last.coverage_raw,
            "coverage_probability": (self.calibrator.coverage.predict(last.coverage_raw)
                                     if self.calibrator and valid else last.coverage_raw),
            "confidence_kind": "calibrated" if self.calibrator else "raw_jev_judgement",
            "uncertain": ((selected is None or not accepted_stop) if self.stop_policy else
                          (selected is None or rounds[-1]["coverage"] < self.settings.coverage_stop
                           or rounds[-1]["best_candidate_probability"] < self.settings.candidate_stop)),
            "candidates": candidates, "rounds": rounds,
            "jev_usage_total": usage_total,
            "jev_calls_total": jev_calls_used,
        }


def label_result(result: Mapping[str, Any], gold_sql: str, db_path: str,
                 timeout: int = 60) -> dict[str, Any]:
    """Use reference execution only offline; return no gold SQL or result values."""
    gold = validate_sql(db_path, gold_sql, timeout)
    if not gold.valid:
        raise ValueError(f"Gold SQL did not execute for question {result['question_id']}")
    gold_signature = result_signature(gold.rows)
    candidates = []
    for item in result["candidates"]:
        execution = validate_sql(db_path, item["sql"], timeout)
        candidates.append({"valid": execution.valid, "raw": item["candidate_raw"],
                           "label": int(execution.valid and
                                        result_signature(execution.rows) == gold_signature)})
    return {"question_id": str(result["question_id"]), "db_id": result["db_id"],
            "coverage_raw": result["coverage_raw"],
            "covered_label": int(any(item["label"] for item in candidates)),
            "candidates": candidates}


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    ids = [str(row["question_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate question_id in labels")
    return rows


def _tasks(args: argparse.Namespace) -> list[Task]:
    with Path(args.tasks_file_path).open("rb") as handle:
        tasks = pickle.load(handle)
    if args.question_id is not None:
        tasks = [task for task in tasks if str(task.question_id) == args.question_id]
    if args.limit is not None:
        tasks = tasks[:args.limit]
    return tasks


def _direct_generator(args: argparse.Namespace, output: Path) -> DirectLLMGenerator:
    from alphasql.llm_call.runtime import configure_environment
    configure_environment()
    provider = getattr(args, "llm_provider", "project")
    if provider == "hkustgz":
        model = args.model or "Qwen3.8-27B"
        base_url = "https://aigc-api.hkust-gz.edu.cn/api/v2"
        api_key_env = "AIGC_API_KEY"
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        direct_connection = True
        if not os.environ.get(api_key_env):
            raise ValueError(f"{api_key_env} is required for {base_url}; no task was started")
    else:
        model = args.model or "qwen3-coder-flash"
        base_url = api_key_env = extra_body = None
        direct_connection = False
    return DirectLLMGenerator(args.database_root_dir, output / "generated",
                              model, args.samples, args.temperature,
                              getattr(args, "prompt_paths", ("skeleton", "query_plan")),
                              getattr(args, "prompt_path_samples", 1),
                              base_url, api_key_env, extra_body, direct_connection)


def _collect_pool(args: argparse.Namespace) -> None:
    """Freeze initial SQL candidates without calling JEV or using reference SQL."""
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    candidate_dir = output / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    tasks = _tasks(args)
    generator = _direct_generator(args, output)
    provenance = _generator_provenance(generator, args.generator_version)
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("generator") != provenance:
            raise ValueError("Initial pool output has a different frozen generator")
    else:
        manifest_path.write_text(json.dumps({"pool_schema_version": 1,
                                            "generator": provenance}, ensure_ascii=False,
                                           indent=2) + "\n", encoding="utf-8")
    failures = []
    collected = 0
    for task in tasks:
        result_path = output / f"{task.question_id}.json"
        pool_path = candidate_dir / f"{task.question_id}.pkl"
        if result_path.exists() and pool_path.exists() and not args.overwrite:
            continue
        try:
            generated = generator.generate(task.model_copy(update={"sql": None}), 0)
            selected: list[GeneratedSQL] = []
            records: list[dict[str, Any]] = []
            seen: set[str] = set()
            db_path = Path(args.database_root_dir) / task.db_id / f"{task.db_id}.sqlite"
            for item in generated:
                canonical = _canonical(item.sql)
                if not canonical or canonical in seen:
                    continue
                if len(selected) >= args.max_candidates:
                    break
                seen.add(canonical)
                execution = validate_sql(str(db_path), item.sql, args.execution_timeout)
                selected.append(item)
                records.append({"sql": item.sql, "canonical_sql": canonical,
                                "route": item.route,
                                "selected_tables": list(item.selected_tables),
                                "selected_columns": list(item.selected_columns),
                                "execution": execution.summary()})
            if not selected:
                raise RuntimeError("Initial generation produced no SQL candidates")
            result = {"pool_schema_version": 1, "question_id": task.question_id,
                      "db_id": task.db_id, "question": task.question,
                      "evidence": task.evidence,
                      "full_schema": _schema_context(task.db_id, args.database_root_dir),
                      "generator": provenance, "candidates": records,
                      "jev_status": "deferred"}
            temporary_pool = pool_path.with_suffix(".pkl.tmp")
            with temporary_pool.open("wb") as handle:
                pickle.dump(selected, handle)
            temporary_pool.replace(pool_path)
            temporary_result = result_path.with_suffix(".json.tmp")
            temporary_result.write_text(json.dumps(result, ensure_ascii=False,
                                                   allow_nan=False, indent=2) + "\n",
                                        encoding="utf-8")
            temporary_result.replace(result_path)
            collected += 1
        except Exception as exc:
            failures.append({"question_id": task.question_id,
                             "error_type": type(exc).__name__, "error": str(exc)})
    (output / "failures.json").write_text(json.dumps(failures, ensure_ascii=False,
                                                   indent=2) + "\n", encoding="utf-8")
    print({"tasks": len(tasks), "collected": collected, "failures": len(failures),
           "output": str(output), "jev_called": False})


def _run(args: argparse.Namespace) -> None:
    settings = JevSettings.from_environment()
    settings.require_api_key()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tasks = _tasks(args)
    generator: CandidateGenerator
    if args.generator == "pool":
        if not args.candidate_dir:
            raise ValueError("--candidate-dir is required for pool generation")
        generator = FrozenPoolGenerator(args.candidate_dir, args.per_round)
    else:
        generator = _direct_generator(args, output)
    judge = JevPoolJudge(JevClient(settings), args.max_candidates)
    calibrator = DualCalibrator.load(args.calibrator) if args.calibrator else None
    stop_policy = StopPolicy.load(args.stop_policy) if getattr(args, "stop_policy", None) else None
    verifier = None
    verifier_sha256 = None
    if getattr(args, "verifier_model", None):
        from alphasql.algorithm.selection.content_verifier import ContentVerifier
        verifier = ContentVerifier.load(args.verifier_model)
        verifier_sha256 = hashlib.sha256(Path(args.verifier_model).read_bytes()).hexdigest()
    pipeline = AdaptiveSQLSelector(generator, judge,
        SelectorSettings(args.max_rounds, args.max_candidates,
                         args.coverage_stop, args.candidate_stop, args.execution_timeout,
                         getattr(args, "max_jev_calls", None)),
        calibrator, getattr(args, "generator_version", None), stop_policy,
        verifier, verifier_sha256)
    failures = []
    for task in tasks:
        path = output / f"{task.question_id}.json"
        if path.exists() and not args.overwrite:
            continue
        try:
            result = pipeline.run(task, args.database_root_dir)
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                            allow_nan=False, default=str) + "\n", encoding="utf-8")
            temporary.replace(path)
        except Exception as exc:
            failures.append({"question_id": task.question_id,
                             "error_type": type(exc).__name__, "error": str(exc)})
    (output / "failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print({"tasks": len(tasks), "failures": len(failures), "output": str(output)})


def _label(args: argparse.Namespace) -> None:
    dataset = json.loads(Path(args.data_path).read_text(encoding="utf-8"))
    by_id = {str(row.get("question_id", i)): row for i, row in enumerate(dataset)}
    labelled = []
    for path in sorted(Path(args.results_dir).glob("*.json")):
        if path.name == "failures.json":
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        row = by_id[str(result["question_id"])]
        if row["db_id"] != result["db_id"]:
            raise ValueError(f"Database mismatch for {result['question_id']}")
        db_path = Path(args.database_root_dir) / row["db_id"] / f"{row['db_id']}.sqlite"
        labelled.append(label_result(result, row.get("SQL") or row.get("sql"),
                                     str(db_path), args.execution_timeout))
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in labelled),
                      encoding="utf-8")
    print({"labelled_questions": len(labelled), "output": str(output)})


def _train(args: argparse.Namespace) -> None:
    train, validation = _read_jsonl(args.train_labels), _read_jsonl(args.validation_labels)
    model, report = fit_dual(train, validation, args.epochs, args.learning_rate)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.save(output / "dual_calibrator.json")
    (output / "validation_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(report)


def _split(args: argparse.Namespace) -> None:
    rows = _read_jsonl(args.labels)
    databases = sorted({str(row["db_id"]) for row in rows})
    if len(databases) < 2 or not 0 < args.validation_fraction < 1:
        raise ValueError("Need at least two databases and a validation fraction in (0, 1)")
    random.Random(args.seed).shuffle(databases)
    n_validation = max(1, min(len(databases) - 1,
                              round(len(databases) * args.validation_fraction)))
    validation_dbs = set(databases[:n_validation])
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    partitions = {
        "train": [row for row in rows if str(row["db_id"]) not in validation_dbs],
        "validation": [row for row in rows if str(row["db_id"]) in validation_dbs],
    }
    for name, part in partitions.items():
        (output / f"{name}_labels.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in part),
            encoding="utf-8")
    manifest = {"source_sha256": hashlib.sha256(Path(args.labels).read_bytes()).hexdigest(),
                "seed": args.seed, "validation_fraction": args.validation_fraction,
                "train_questions": len(partitions["train"]),
                "validation_questions": len(partitions["validation"]),
                "validation_databases": sorted(validation_dbs)}
    (output / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--tasks-file-path", required=True)
    run.add_argument("--database-root-dir", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--generator", choices=("direct", "pool"), default="direct")
    run.add_argument("--candidate-dir")
    run.add_argument("--llm-provider", choices=("project", "hkustgz"), default="project")
    run.add_argument("--model", help="Default: project qwen3-coder-flash or HKUST(GZ) Qwen3.8-27B")
    run.add_argument("--generator-version", help="Immutable tag for the generator and prompts")
    run.add_argument("--samples", type=int, default=3)
    run.add_argument("--prompt-paths", nargs="*", choices=("skeleton", "query_plan"),
                     default=("skeleton", "query_plan"),
                     help="Additional LLM prompt paths; pass with no values to disable")
    run.add_argument("--prompt-path-samples", type=int, default=1)
    run.add_argument("--temperature", type=float, default=0.8)
    run.add_argument("--per-round", type=int, default=8)
    run.add_argument("--max-rounds", type=int, default=3)
    run.add_argument("--max-candidates", type=int, default=32)
    run.add_argument("--max-jev-calls", type=int,
                     help="Maximum JEV requests per question, including adjudication")
    run.add_argument("--coverage-stop", type=float, default=0.8)
    run.add_argument("--candidate-stop", type=float, default=0.8)
    run.add_argument("--execution-timeout", type=int, default=60)
    run.add_argument("--calibrator")
    run.add_argument("--stop-policy", help="Validation-fitted selective stopping policy")
    run.add_argument("--verifier-model", help="Frozen content verifier JSON model")
    run.add_argument("--question-id")
    run.add_argument("--limit", type=int)
    run.add_argument("--overwrite", action="store_true")
    run.set_defaults(func=_run)
    pool = sub.add_parser("collect-pool", help="Freeze initial SQL candidates without JEV")
    pool.add_argument("--tasks-file-path", required=True)
    pool.add_argument("--database-root-dir", required=True)
    pool.add_argument("--output-dir", required=True)
    pool.add_argument("--llm-provider", choices=("project", "hkustgz"), default="project")
    pool.add_argument("--model")
    pool.add_argument("--generator-version")
    pool.add_argument("--samples", type=int, default=3)
    pool.add_argument("--prompt-paths", nargs="*", choices=("skeleton", "query_plan"),
                      default=("skeleton", "query_plan"))
    pool.add_argument("--prompt-path-samples", type=int, default=1)
    pool.add_argument("--temperature", type=float, default=0.8)
    pool.add_argument("--max-candidates", type=int, default=16)
    pool.add_argument("--execution-timeout", type=int, default=60)
    pool.add_argument("--question-id")
    pool.add_argument("--limit", type=int)
    pool.add_argument("--overwrite", action="store_true")
    pool.set_defaults(func=_collect_pool)
    label = sub.add_parser("label")
    label.add_argument("--results-dir", required=True)
    label.add_argument("--data-path", required=True)
    label.add_argument("--database-root-dir", required=True)
    label.add_argument("--output-path", required=True)
    label.add_argument("--execution-timeout", type=int, default=60)
    label.set_defaults(func=_label)
    split = sub.add_parser("split")
    split.add_argument("--labels", required=True)
    split.add_argument("--output-dir", required=True)
    split.add_argument("--validation-fraction", type=float, default=0.2)
    split.add_argument("--seed", type=int, default=42)
    split.set_defaults(func=_split)
    train = sub.add_parser("train")
    train.add_argument("--train-labels", required=True)
    train.add_argument("--validation-labels", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--learning-rate", type=float, default=0.03)
    train.set_defaults(func=_train)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
