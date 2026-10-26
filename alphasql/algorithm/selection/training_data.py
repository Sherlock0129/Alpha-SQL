"""Offline, gold-isolated training records for adaptive SQL selection.

Execution agreement on one database is a proxy label, not a proof of semantic
equivalence. Structural diagnoses and near-miss tags are weak labels and must
be audited before they are used as supervised targets.
"""

from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import sqlglot
from sqlglot import exp
from sqlglot.diff import diff

from alphasql.algorithm.compiler.validation import validate_sql
from alphasql.algorithm.selection.schema_signal import result_signature


ACTION_NAMES = ("relink", "revise", "diversify", "adjudicate")
DIAGNOSTIC_NAMES = ("schema", "projection", "filters", "logic")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def generator_identity(generator: Mapping[str, Any]) -> str:
    return _digest(generator)


def pool_id(db_id: str, question_id: Any, candidates: Sequence[Mapping[str, Any]],
            generator_fingerprint: str) -> str:
    """Content-addressed ordered pool; identical starting states share an ID."""
    return _digest({"db_id": db_id, "question_id": str(question_id),
                    "generator_fingerprint": generator_fingerprint,
                    "candidates": [{"sql": item.get("canonical_sql") or item["sql"],
                                    "route": item.get("route"),
                                    "selected_tables": item.get("selected_tables", []),
                                    "selected_columns": item.get("selected_columns", [])}
                                   for item in candidates]})


def state_id(pool_fingerprint: str, round_index: int, jev: Mapping[str, Any],
             candidates: Sequence[Mapping[str, Any]], full_schema: Mapping[str, Any],
             generation_schema: Mapping[str, Any], judge: Mapping[str, Any],
             calibrator: Mapping[str, Any]) -> str:
    """Fingerprint every observable input that can influence the next action."""
    return _digest({"pool_id": pool_fingerprint, "round": round_index,
                    "jev": jev,
                    "candidate_scores": [item.get("candidate_probability") for item in candidates],
                    "full_schema": full_schema, "generation_schema": generation_schema,
                    "judge": judge, "calibrator": calibrator})


def _normal_sql(node: exp.Expression | None) -> str | None:
    return node.sql(dialect="sqlite", normalize=True, pretty=False) if node else None


def _canonical_sql(sql: str) -> str | None:
    try:
        parsed = sqlglot.parse_one(sql, read="sqlite")
        return parsed.sql(dialect="sqlite", normalize=True) if parsed else None
    except Exception:
        return None


def _execution_signature(rows: Sequence[Sequence[Any]], ordered: bool) -> Any:
    if ordered:
        return tuple(result_signature([row]) for row in rows)
    return result_signature(rows)


def _gold_requires_order(gold_sql: str) -> bool:
    try:
        parsed = sqlglot.parse_one(gold_sql, read="sqlite")
        return bool(parsed and parsed.args.get("order"))
    except Exception:
        return False


def weak_diagnosis(candidate_sql: str, gold_sql: str) -> dict[str, Any]:
    """Multi-label AST differences; they are hypotheses, never gold diagnoses."""
    try:
        candidate = sqlglot.parse_one(candidate_sql, read="sqlite")
        gold = sqlglot.parse_one(gold_sql, read="sqlite")
    except Exception:
        return {"source": "unavailable_parse_error", "labels": None}
    candidate_select, gold_select = candidate.find(exp.Select), gold.find(exp.Select)
    if not candidate_select or not gold_select:
        return {"source": "unavailable_structure", "labels": None}
    table_names = lambda root: sorted(table.sql(dialect="sqlite", normalize=True)
                                      for table in root.find_all(exp.Table))
    projections = lambda node: [_normal_sql(item) for item in node.expressions]
    logic_nodes = ("group", "order", "limit", "qualify", "distinct")
    labels = {
        "schema": (table_names(candidate) != table_names(gold)
                   or [_normal_sql(node) for node in candidate.find_all(exp.Join)]
                   != [_normal_sql(node) for node in gold.find_all(exp.Join)]),
        "projection": projections(candidate_select) != projections(gold_select),
        "filters": any(_normal_sql(candidate_select.args.get(name))
                       != _normal_sql(gold_select.args.get(name))
                       for name in ("where", "having")),
        "logic": (any(_normal_sql(candidate_select.args.get(name))
                      != _normal_sql(gold_select.args.get(name)) for name in logic_nodes)
                  or type(candidate) is not type(gold)
                  or sorted(type(node).__name__ for node in candidate.find_all(exp.AggFunc))
                  != sorted(type(node).__name__ for node in gold.find_all(exp.AggFunc))),
    }
    return {"source": "weak_ast_difference", "labels": labels}


def _ast_edit_count(candidate_sql: str, gold_sql: str) -> int | None:
    try:
        edits = diff(sqlglot.parse_one(candidate_sql, read="sqlite"),
                     sqlglot.parse_one(gold_sql, read="sqlite"))
        return sum(type(edit).__name__ != "Keep" for edit in edits)
    except Exception:
        return None


def _difficulty(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    positives = [float(item.get("candidate_probability", item.get("candidate_raw", 0)))
                 for item in candidates if item["execution_label"]]
    negatives = [float(item.get("candidate_probability", item.get("candidate_raw", 0)))
                 for item in candidates if not item["execution_label"]
                 and item.get("execution", {}).get("valid")]
    if not positives:
        return {"stratum": "generation_hard", "generation_difficulty": 1.0,
                "selection_difficulty": None, "positive_negative_margin": None}
    best_positive = max(positives)
    margin = best_positive - max(negatives, default=0.0)
    hard = best_positive < 0.5 or margin <= 0
    return {"stratum": "selection_hard" if hard else "covered_easy",
            "generation_difficulty": 0.0,
            "selection_difficulty": max(0.0, 1.0 - best_positive,
                                        (1.0 - margin) / 2.0),
            "positive_negative_margin": margin}


def label_trace(result: Mapping[str, Any], gold_sql: str, db_path: str,
                timeout: int = 60, near_miss_max_edits: int = 4) -> dict[str, Any]:
    """Label every pool snapshot without exposing the reference SQL in output."""
    if result.get("schema_version", 0) < 2 or not result.get("rounds"):
        raise ValueError("Rich labels require a version-2 run with round snapshots")
    required = ("question", "evidence", "full_schema", "generator")
    if any(key not in result for key in required):
        raise ValueError("Run is missing question, evidence, schema or provenance")
    gold = validate_sql(db_path, gold_sql, timeout)
    if not gold.valid:
        raise ValueError(f"Gold SQL did not execute for question {result['question_id']}")
    ordered = _gold_requires_order(gold_sql)
    reference = _execution_signature(gold.rows, ordered)
    cache: dict[str, tuple[bool, int, int | None, dict[str, Any]]] = {}

    def annotate(item: Mapping[str, Any]) -> dict[str, Any]:
        sql = item["sql"]
        if sql not in cache:
            execution = validate_sql(db_path, sql, timeout)
            label = int(execution.valid and
                        _execution_signature(execution.rows, ordered) == reference)
            edits = _ast_edit_count(sql, gold_sql) if execution.valid and not label else None
            diagnosis = (weak_diagnosis(sql, gold_sql) if execution.valid and not label
                         else {"source": "not_applicable", "labels": None})
            cache[sql] = execution.valid, label, edits, diagnosis
        valid, label, edits, diagnosis = cache[sql]
        record = copy.deepcopy(dict(item))
        record["execution_label"] = label
        record["label_source"] = "single_db_execution_equivalence"
        record["execution_rechecked_valid"] = valid
        record["near_miss"] = bool(not label and valid and edits is not None
                                   and 0 < edits <= near_miss_max_edits)
        record["near_miss_ast_edits"] = edits
        record["weak_diagnosis"] = diagnosis
        return record

    generator_fingerprint = generator_identity(result["generator"])
    rounds = []
    previous: dict[str, Any] | None = None
    for source_round in result["rounds"]:
        candidates = [annotate(item) for item in source_round["candidates"]]
        if len(candidates) != source_round["pool_size"]:
            raise ValueError("Round candidate snapshot does not match pool_size")
        state = copy.deepcopy(dict(source_round))
        state["candidates"] = candidates
        state["pool_id"] = pool_id(result["db_id"], result["question_id"],
                                   candidates, generator_fingerprint)
        state["state_id"] = state_id(state["pool_id"], state["round"],
                                      state["jev"], candidates,
                                      result["full_schema"],
                                      result.get("generation_schema", {}),
                                      result.get("judge", {}),
                                      result.get("calibrator", {}))
        state["covered_label"] = int(any(item["execution_label"] for item in candidates))
        valid_indices = [i for i, item in enumerate(candidates)
                         if item.get("execution", {}).get("valid")]
        selected = max(valid_indices,
                       key=lambda i: (candidates[i].get("candidate_probability", 0), -i))\
            if valid_indices else None
        state["selected_index"] = selected
        state["selected_correct_label"] = (candidates[selected]["execution_label"]
                                            if selected is not None else 0)
        state["difficulty"] = _difficulty(candidates)
        state["diagnosis_target"] = (candidates[selected]["weak_diagnosis"]
                                     if selected is not None and not state["selected_correct_label"]
                                     else {"source": "not_applicable", "labels": None})
        if previous is not None:
            state["observed_action_outcome"] = {
                "source": "single_observed_branch_not_counterfactual",
                "from_state_id": previous["state_id"],
                "action": state["generation_action"],
                "new_correct_count": sum(candidates[i]["execution_label"]
                                         for i in state["new_candidate_indices"]),
                "delta_coverage": state["covered_label"] - previous["covered_label"],
                "delta_selected_correct": (state["selected_correct_label"]
                                           - previous["selected_correct_label"]),
            }
        rounds.append(state)
        previous = state
    return {"schema_version": 1, "question_id": result["question_id"],
            "db_id": result["db_id"], "question": result["question"],
            "evidence": result["evidence"], "full_schema": copy.deepcopy(result["full_schema"]),
            "generation_schema": copy.deepcopy(result.get("generation_schema", {})),
            "generator": copy.deepcopy(result["generator"]),
            "judge": copy.deepcopy(result.get("judge", {})),
            "calibrator": copy.deepcopy(result.get("calibrator", {})),
            "initial_pool_id": rounds[0]["pool_id"],
            "label_caveat": "Single-database execution agreement is a proxy, not semantic proof",
            "order_sensitive_label": ordered,
            "rounds": rounds}


def split_by_database(rows: Sequence[Mapping[str, Any]], validation_fraction: float,
                      test_fraction: float, seed: int) -> dict[str, list[Mapping[str, Any]]]:
    if validation_fraction <= 0 or test_fraction <= 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must be positive and sum below one")
    databases = sorted({str(row["db_id"]) for row in rows})
    if len(databases) < 3:
        raise ValueError("Three-way database split needs at least three databases")
    random.Random(seed).shuffle(databases)
    n_test = max(1, round(len(databases) * test_fraction))
    n_validation = max(1, round(len(databases) * validation_fraction))
    if n_test + n_validation >= len(databases):
        raise ValueError("Fractions leave no training database")
    test_dbs = set(databases[:n_test])
    validation_dbs = set(databases[n_test:n_test + n_validation])
    partitions = {"train": [], "validation": [], "test": []}
    seen = set()
    for row in rows:
        identity = (str(row["db_id"]), str(row["question_id"]))
        if identity in seen:
            raise ValueError(f"Duplicate question: {identity}")
        seen.add(identity)
        partition = ("test" if identity[0] in test_dbs else
                     "validation" if identity[0] in validation_dbs else "train")
        partitions[partition].append(row)
    return partitions


def make_action_manifest(labelled: Sequence[Mapping[str, Any]],
                         raw_by_question: Mapping[str, Mapping[str, Any]],
                         per_stratum: int, seed: int,
                         budget: Mapping[str, int]) -> list[dict[str, Any]]:
    """Sample identical round-zero states; emitted prompts contain no gold labels."""
    if per_stratum < 1 or budget.get("max_new_candidates", 0) < 1:
        raise ValueError("Positive per-stratum sample count and candidate budget required")
    if len({_digest({"generator": row["generator"], "judge": row.get("judge", {}),
                     "calibrator": row.get("calibrator", {})}) for row in labelled}) != 1:
        raise ValueError("Action probes need one frozen generator, judge and calibrator")
    groups: dict[str, list[Mapping[str, Any]]] = {name: [] for name in
                                                  ("generation_hard", "selection_hard", "covered_easy")}
    for row in labelled:
        if not row["rounds"]:
            continue
        groups[row["rounds"][0]["difficulty"]["stratum"]].append(row)
    rng = random.Random(seed)
    manifest = []
    for stratum, group in groups.items():
        rng.shuffle(group)
        for labelled_row in group[:per_stratum]:
            key = str(labelled_row["question_id"])
            raw = raw_by_question[key]
            if raw["db_id"] != labelled_row["db_id"]:
                raise ValueError(f"Database mismatch for {key}")
            first = raw["rounds"][0]
            actual_pool_id = pool_id(raw["db_id"], key, first["candidates"],
                                     generator_identity(raw["generator"]))
            if actual_pool_id != labelled_row["initial_pool_id"]:
                raise ValueError(f"Initial pool changed for {key}")
            actual_state_id = state_id(actual_pool_id, first["round"], first["jev"],
                                       first["candidates"], raw["full_schema"],
                                       raw.get("generation_schema", {}),
                                       raw.get("judge", {}), raw.get("calibrator", {}))
            if actual_state_id != labelled_row["rounds"][0]["state_id"]:
                raise ValueError(f"Initial JEV, schema or calibration state changed for {key}")
            manifest.append({"state_id": labelled_row["rounds"][0]["state_id"],
                             "initial_pool_id": actual_pool_id,
                             "question_id": raw["question_id"], "db_id": raw["db_id"],
                             "question": raw["question"], "evidence": raw["evidence"],
                             "full_schema": copy.deepcopy(raw["full_schema"]),
                             "generation_schema": copy.deepcopy(raw.get("generation_schema", {})),
                             "generator": copy.deepcopy(raw["generator"]),
                             "judge": copy.deepcopy(raw.get("judge", {})),
                             "calibrator": copy.deepcopy(raw.get("calibrator", {})),
                             "initial_candidates": copy.deepcopy(first["candidates"]),
                             "initial_jev": copy.deepcopy(first["jev"]),
                             "actions": list(ACTION_NAMES), "budget": dict(budget)})
    return manifest


def label_action_arms(arms: Sequence[Mapping[str, Any]],
                      labelled: Sequence[Mapping[str, Any]],
                      gold_by_question: Mapping[str, str], db_root: str,
                      timeout: int = 60) -> list[dict[str, Any]]:
    """Score all four actions from the same state under the same declared budget."""
    states = {row["rounds"][0]["state_id"]: row for row in labelled}
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for arm in arms:
        if arm["action"] not in ACTION_NAMES:
            raise ValueError(f"Unknown intervention action: {arm['action']}")
        grouped.setdefault(str(arm["state_id"]), []).append(arm)
    output = []
    for state_id, group in grouped.items():
        if state_id not in states:
            raise ValueError(f"Unknown initial state {state_id}")
        base = states[state_id]
        first = base["rounds"][0]
        actions = [arm["action"] for arm in group]
        if len(actions) != len(set(actions)):
            raise ValueError(f"Duplicate action for state {state_id}")
        budgets = {_digest(arm["budget"]) for arm in group}
        if len(budgets) != 1:
            raise ValueError(f"Actions do not share a budget for state {state_id}")
        reference_sql = gold_by_question[str(base["question_id"])]
        db_id = str(base["db_id"])
        db_path = str(Path(db_root) / db_id / f"{db_id}.sqlite")
        gold = validate_sql(db_path, reference_sql, timeout)
        if not gold.valid:
            raise ValueError(f"Gold SQL invalid for {base['question_id']}")
        ordered = _gold_requires_order(reference_sql)
        reference = _execution_signature(gold.rows, ordered)
        initial_sqls = [item["sql"] for item in first["candidates"]]
        initial_canonical = {canonical for sql in initial_sqls
                             if (canonical := _canonical_sql(sql)) is not None}
        scored = []
        for arm in group:
            if str(arm["question_id"]) != str(base["question_id"]) or arm["db_id"] != db_id:
                raise ValueError("Intervention question or database mismatch")
            if arm["initial_pool_id"] != base["initial_pool_id"]:
                raise ValueError("Intervention did not start from the frozen pool")
            budget = arm["budget"]
            new_sqls = [item["sql"] if isinstance(item, Mapping) else str(item)
                        for item in arm.get("new_candidates", [])]
            if len(new_sqls) > int(budget["max_new_candidates"]):
                raise ValueError("Intervention exceeded candidate budget")
            if arm["action"] == "adjudicate" and new_sqls:
                raise ValueError("Adjudication must not generate SQL")
            usage = arm.get("usage")
            if not isinstance(usage, Mapping) or not all(
                    name in usage for name in ("model_calls", "input_tokens", "output_tokens")):
                raise ValueError("Each action must report model calls and token usage")
            if ("max_model_calls" in budget and
                    int(usage.get("model_calls", 0)) > int(budget["max_model_calls"])):
                raise ValueError("Intervention exceeded call budget")
            observed_tokens = int(usage.get("input_tokens", 0)) + int(usage.get("output_tokens", 0))
            if "max_tokens" in budget and observed_tokens > int(budget["max_tokens"]):
                raise ValueError("Intervention exceeded token budget")
            new_labels = []
            canonical = set(initial_canonical)
            for sql in new_sqls:
                execution = validate_sql(db_path, sql, timeout)
                new_labels.append(int(execution.valid and
                                      _execution_signature(execution.rows, ordered) == reference))
                parsed = _canonical_sql(sql) if execution.valid else None
                if parsed is not None:
                    canonical.add(parsed)
            selected_sql = arm.get("selected_sql")
            if selected_sql:
                selected_parsed = _canonical_sql(selected_sql)
                if selected_parsed is None or selected_parsed not in canonical:
                    raise ValueError("Selected SQL is not in the intervention pool")
                selected_execution = validate_sql(db_path, selected_sql, timeout)
                selected_correct = int(selected_execution.valid and
                                       _execution_signature(selected_execution.rows,
                                                            ordered) == reference)
            else:
                selected_correct = 0
            scored.append({"action": arm["action"], "new_candidate_count": len(new_sqls),
                           "new_correct_count": sum(new_labels),
                           "after_covered_label": int(bool(first["covered_label"])
                                                      or any(new_labels)),
                           "after_selected_correct_label": selected_correct,
                           "delta_coverage": int(bool(first["covered_label"])
                                                 or any(new_labels)) - first["covered_label"],
                           "delta_selected_correct": selected_correct
                           - first["selected_correct_label"],
                           "usage": dict(usage)})
        complete = set(actions) == set(ACTION_NAMES)
        best = None
        cost_complete = all(arm["usage"].get("tokens_complete", True) for arm in scored)
        if complete and cost_complete:
            winners = [arm for arm in scored if arm["delta_selected_correct"] > 0]
            if winners:
                best = min(winners, key=lambda item: (
                    int(item["usage"].get("input_tokens", 0))
                    + int(item["usage"].get("output_tokens", 0)),
                    int(item["usage"].get("model_calls", 0)), item["action"]))["action"]
            else:
                best = "stop"
        output.append({"state_id": state_id, "initial_pool_id": base["initial_pool_id"],
                       "question_id": base["question_id"], "db_id": db_id,
                       "budget": copy.deepcopy(group[0]["budget"]),
                       "before_covered_label": first["covered_label"],
                       "before_selected_correct_label": first["selected_correct_label"],
                       "arms": scored, "complete": complete,
                       "cost_complete": cost_complete,
                       "best_observed_action": best,
                       "label_source": "single_offline_intervention_outcome"})
    return output


def stratum_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(row["rounds"][0]["difficulty"]["stratum"]
                               for row in rows).items()))
