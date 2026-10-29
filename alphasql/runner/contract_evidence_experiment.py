"""Candidate-blind intent contract followed by targeted JEV SQL obligations.

All collection is label-free. The local evidence model was fitted on training
databases; this module never reads gold SQL until the evaluate command.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Mapping

import sqlglot
from sqlglot import exp

from alphasql.algorithm.selection.execution_evidence import pool_identity
from alphasql.jev.client import JevClient, JevTimeoutError
from alphasql.jev.models import ChoiceQuestion, ChoiceResult, NoulQuestion, NoulResult
from alphasql.jev.runtime import JevSettings
from alphasql.llm_call.runtime import configure_environment
from alphasql.runner.selection_v2 import _rank_score, _read, _save


CONTRACT_OPTIONS = {
    "metric": {
        "count": "Count matching records or entities (possibly distinct).",
        "sum": "Sum a numeric measure.",
        "average": "Average a numeric measure.",
        "extreme": "Find a minimum or maximum, or an entity with that extreme.",
        "other": "No such main aggregation is requested.",
        "ambiguous": "The main aggregation cannot be determined from question and evidence.",
    },
    "multiplicity": {
        "unique": "Return each matching entity or value once.",
        "preserve": "Preserve each matching record, including repeated projected values.",
        "unspecified": "Question and evidence do not determine whether duplicates are allowed.",
    },
    "output_shape": {
        "single": "The answer clearly requires exactly one output attribute or measure.",
        "multiple": "The answer clearly requires two or more separate output attributes or measures.",
        "unspecified": "One versus multiple output columns is not determined.",
    },
    "population": {
        "restricted": "An explicit named cohort, condition, value, or date must restrict the queried population.",
        "all": "The requested population is all relevant records, without an explicit restriction.",
        "unspecified": "Population scope is unclear or cannot be determined.",
    },
    "missing_values": {
        "exclude": "Missing numeric values must not be treated as a minimum or maximum.",
        "include": "The question explicitly requires rows with missing values.",
        "unspecified": "The question makes no specific NULL-value requirement.",
    },
}
PROMPT_SHA256 = hashlib.sha256(json.dumps(CONTRACT_OPTIONS, sort_keys=True).encode()).hexdigest()
SLOTS = ("metric", "multiplicity", "output_shape", "population", "missing_values")
JEV_CORRECTION_CAP = 0.15  # Frozen before collecting validation outcomes.
MIN_SWITCH_ADVANTAGE = 0.08


def _terms(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z][a-z0-9_]+", text.lower()) if len(word) > 2}


def _compact_schema(row: Mapping[str, Any]) -> dict[str, Any]:
    """Choose schema from question/evidence only; never inspect candidate SQL."""
    schema = row["full_schema"]
    terms = _terms(f'{row["question"]} {row.get("evidence") or ""}')
    ranked = []
    for table, details in schema.get("tables", {}).items():
        columns = details.get("columns", {})
        score = 4 * len(_terms(table) & terms)
        score += sum(len(_terms(name) & terms) for name in columns)
        ranked.append((score, table, columns))
    ranked.sort(key=lambda value: (-value[0], value[1]))
    selected = [part for part in ranked if part[0] > 0][:6]
    if not selected:
        selected = ranked[:4]
    out = {}
    for _, table, columns in selected:
        ordered = sorted(columns.items(), key=lambda item:
                         (-len(_terms(item[0]) & terms), item[0]))[:24]
        out[table] = {name: {"type": desc.get("type"),
                             "description": str(desc.get("description") or "")[:90]}
                      for name, desc in ordered}
    return {"db_id": row["db_id"], "tables": out}


def _blind_state(row: Mapping[str, Any]) -> dict[str, Any]:
    return {"task": "Infer the question's answer contract before seeing any candidate SQL.",
            "question": row["question"], "evidence": row.get("evidence") or "",
            "relevant_schema": _compact_schema(row)}


def _facts(sql: str, evidence: Mapping[str, Any]) -> dict[str, Any]:
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        tree = None
    if tree is None:
        return {"parse_valid": False}
    select = tree.find(exp.Select)
    projections = select.expressions if select is not None else []
    return {
        "parse_valid": True,
        "aggregates": sorted({kind for kind, node in
                              (("count", exp.Count), ("sum", exp.Sum),
                               ("average", exp.Avg), ("min", exp.Min), ("max", exp.Max))
                              if tree.find(node) is not None}),
        "distinct": tree.find(exp.Distinct) is not None,
        "projection_count": len(projections),
        "where_clauses": [node.sql(dialect="sqlite")[:350]
                          for node in tree.find_all(exp.Where)][:5],
        "having_clauses": [node.sql(dialect="sqlite")[:350]
                           for node in tree.find_all(exp.Having)][:3],
        "group_by": tree.find(exp.Group) is not None,
        "order_by": tree.find(exp.Order) is not None,
        "limit": tree.find(exp.Limit) is not None,
        "literal_values": list(dict.fromkeys(node.sql(dialect="sqlite")
                                          for node in tree.find_all(exp.Literal)))[:12],
        "result_rows": evidence.get("row_count"),
        "result_columns": evidence.get("column_count"),
        "result_has_null": bool(evidence.get("features", {}).get("exec_sample_has_null")),
        "result_hash": evidence.get("result_hash"),
    }


def _valid(row: Mapping[str, Any]) -> list[int]:
    return [i for i, candidate in enumerate(row["rounds"][0]["candidates"])
            if candidate.get("execution_rechecked_valid", candidate["execution"]["valid"])]


def _slot_differences(facts: Mapping[int, Mapping[str, Any]],
                      contract: Mapping[str, str]) -> list[str]:
    if len(facts) < 2:
        return []
    fields = {
        "metric": ("aggregates", "group_by"),
        "multiplicity": ("distinct", "result_rows"),
        "output_shape": ("projection_count", "result_columns"),
        "population": ("where_clauses", "having_clauses", "literal_values"),
        "missing_values": ("where_clauses", "result_has_null"),
    }
    inactive = {"metric": {"other", "ambiguous"},
                "multiplicity": {"unspecified"},
                "output_shape": {"unspecified"},
                "population": {"unspecified"},
                "missing_values": {"unspecified"}}
    return [slot for slot, names in fields.items()
            if contract[slot] not in inactive[slot] and
            len({json.dumps([item.get(name) for name in names], sort_keys=True)
                 for item in facts.values()}) > 1]


def _usage(response: Any) -> dict[str, int]:
    return {"calls": 1, "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens}


def _collect_one(client: JevClient, row: Mapping[str, Any], execution: Mapping[str, Any],
                 path: Path) -> dict[str, Any]:
    indices = _valid(row)
    if execution["pool_sha256"] != pool_identity(row):
        raise ValueError("Execution evidence is from a different candidate pool")
    saved = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {
        "db_id": row["db_id"], "question_id": row["question_id"],
        "pool_sha256": pool_identity(row), "prompt_sha256": PROMPT_SHA256,
        "contract": {}, "obligations": {}, "triggered_slots": [], "usage": {}}
    if saved["pool_sha256"] != pool_identity(row) or saved["prompt_sha256"] != PROMPT_SHA256:
        raise ValueError("Frozen pool or prompt version changed")
    if not saved["contract"]:
        questions = {slot: ChoiceQuestion(options,
                     f"Choose the best-supported interpretation of {slot}. "
                     "Use ambiguous or unspecified when the text leaves a real choice open.")
                     for slot, options in CONTRACT_OPTIONS.items()}
        response = client.ask(_blind_state(row), questions)
        saved["contract"] = {slot: response.require(slot, ChoiceResult).choice
                             for slot in SLOTS}
        saved["contract_probabilities"] = {
            slot: response.require(slot, ChoiceResult).probabilities for slot in SLOTS}
        saved["usage"]["intent"] = _usage(response)
        saved["model"] = response.model
        _save(path, saved)
    candidates = row["rounds"][0]["candidates"]
    facts = {i: _facts(candidates[i]["sql"], execution["candidates"][i]) for i in indices}
    slots = _slot_differences(facts, saved["contract"])
    saved["triggered_slots"] = slots
    if slots and not saved.get("obligations_done"):
        # Raw JEV scores, generation routes and all gold labels are deliberately hidden.
        state = {"task": "Check each SQL against only the stated disputed obligations. "
                         "A successful execution is not proof of semantic correctness.",
                 "question": row["question"], "evidence": row.get("evidence") or "",
                 "relevant_schema": _compact_schema(row), "contract": saved["contract"],
                 "candidates": [{"index": i, "sql": candidates[i]["sql"][:1800],
                                 "structure_and_original_db_execution": facts[i]}
                                for i in indices]}
        questions = {f"{slot}_{i}": NoulQuestion(
            f"Does candidate {i} satisfy the question's {slot} requirement "
            f"({saved['contract'][slot]})? Inspect its exact SQL clause and "
            "the supplied original-database evidence. If the contract is not "
            "decisive, avoid a strong answer.") for slot in slots for i in indices}
        response = client.ask(state, questions)
        if saved.get("model") != response.model:
            raise ValueError("JEV model changed between stages")
        saved["obligations"] = {str(i): {slot: response.require(f"{slot}_{i}", NoulResult).noul
                                          for slot in slots} for i in indices}
        saved["usage"]["obligations"] = _usage(response)
        saved["obligations_done"] = True
        _save(path, saved)
    elif not slots:
        saved["obligations_done"] = True
        _save(path, saved)
    return saved


def _rows(input_path: str, split_file: str, split: str) -> list[dict[str, Any]]:
    allowed = set(json.loads(Path(split_file).read_text(encoding="utf-8"))["question_ids"][split])
    return [row for row in _read(input_path) if int(row["question_id"]) in allowed]


def collect(args: argparse.Namespace) -> None:
    configure_environment()
    client = JevClient(JevSettings.from_environment())
    failures = []
    attempted = completed = 0
    for row in _rows(args.input, args.split_file, args.split):
        if int(row["question_id"]) % args.shard_count != args.shard_index:
            continue
        path = Path(args.output_dir) / f'{row["db_id"]}_{row["question_id"]}.json'
        if path.is_file():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved["pool_sha256"] != pool_identity(row) or saved["prompt_sha256"] != PROMPT_SHA256:
                raise ValueError(f"Frozen pool or prompt mismatch: {path}")
            if saved.get("obligations_done"):
                continue
        if args.limit and attempted >= args.limit:
            break
        attempted += 1
        execution = json.loads((Path(args.execution_dir) / path.name).read_text(encoding="utf-8"))
        for trial in range(args.attempts):
            try:
                result = _collect_one(client, row, execution, path)
                completed += 1
                print(json.dumps({"question_id": row["question_id"],
                                  "triggered_slots": result["triggered_slots"],
                                  "usage": result["usage"]}), flush=True)
                break
            except JevTimeoutError as exc:
                if trial + 1 < args.attempts:
                    time.sleep(min(8, 2 ** trial))
                    continue
                failures.append({"question_id": row["question_id"], "error": str(exc)})
            except Exception as exc:
                failures.append({"question_id": row["question_id"],
                                 "error": f"{type(exc).__name__}: {exc}"})
                break
    _save(Path(args.output_dir) / f"failures_{args.shard_index}.json", failures)
    print(json.dumps({"attempted": attempted, "completed": completed,
                      "failures": len(failures)}), flush=True)


def _choices(row: Mapping[str, Any], execution: Mapping[str, Any],
             collection: Mapping[str, Any], model: Mapping[str, Any]) -> dict[str, int | None]:
    indices = _valid(row)
    if not indices:
        return {"raw_jev": None, "structure_execution": None, "two_stage_jev": None}
    candidates = row["rounds"][0]["candidates"]
    raw = {i: float(candidates[i]["candidate_raw"]) for i in indices}
    local = {i: _rank_score({"raw": raw[i],
                            "execution": execution["candidates"][i]}, model, 0.1)
             for i in indices}
    order = lambda scores: max(indices, key=lambda i: (scores[i], raw[i], -i))
    top_raw, top_local = order(raw), order(local)
    obligations = collection.get("obligations") or {}
    if not obligations:
        return {"raw_jev": top_raw, "structure_execution": top_local,
                "two_stage_jev": top_local}
    check = {i: sum(float(value) for value in obligations[str(i)].values()) /
                 len(obligations[str(i)]) for i in indices}
    scores = {i: local[i] + JEV_CORRECTION_CAP * (2 * check[i] - 1) for i in indices}
    proposed = order(scores)
    # A single correlated JEV opinion cannot overturn a large raw advantage.
    if proposed != top_local and scores[proposed] - scores[top_local] < MIN_SWITCH_ADVANTAGE:
        proposed = top_local
    return {"raw_jev": top_raw, "structure_execution": top_local,
            "two_stage_jev": proposed}


def evaluate(args: argparse.Namespace) -> None:
    rows = _rows(args.input, args.split_file, args.split)
    bundle = json.loads(Path(args.model).read_text(encoding="utf-8"))
    model = bundle["configs"]["execution"]["model"]
    records = []
    usage = Counter()
    missing = []
    for row in rows:
        stem = f'{row["db_id"]}_{row["question_id"]}.json'
        path = Path(args.collection_dir) / stem
        if not path.is_file():
            missing.append(stem)
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        execution = json.loads((Path(args.execution_dir) / stem).read_text(encoding="utf-8"))
        if result["pool_sha256"] != pool_identity(row) or execution["pool_sha256"] != pool_identity(row):
            raise ValueError(f"Candidate pool fingerprint mismatch: {stem}")
        if result["prompt_sha256"] != PROMPT_SHA256 or not result.get("obligations_done"):
            missing.append(stem)
            continue
        choices = _choices(row, execution, result, model)
        candidates = row["rounds"][0]["candidates"]
        correct = {i for i in _valid(row) if candidates[i].get("execution_label") == 1}
        if any(candidates[i].get("execution_label") is None for i in _valid(row)):
            raise ValueError("Offline execution labels missing")
        for stage in result["usage"].values():
            for key in ("calls", "input_tokens", "output_tokens"):
                usage[key] += stage[key]
        records.append({"db_id": row["db_id"], "question_id": row["question_id"],
                        "pool_sha256": result["pool_sha256"], "covered": bool(correct),
                        "choices": choices, "execution_correct": {
                            name: int(index in correct) for name, index in choices.items()},
                        "contract": result["contract"],
                        "triggered_slots": result["triggered_slots"],
                        "usage": result["usage"]})
    if missing and not args.allow_partial:
        raise ValueError(f"Incomplete collection: {len(missing)} missing; first {missing[:3]}")
    methods = ("raw_jev", "structure_execution", "two_stage_jev")
    report = {"split": args.split, "questions_requested": len(rows),
              "questions_evaluated": len(records), "missing": missing,
              "covered": sum(r["covered"] for r in records),
              "triggered": sum(bool(r["triggered_slots"]) for r in records),
              "extra_jev_usage": {**dict(usage),
                                  "total_tokens": usage["input_tokens"] + usage["output_tokens"]},
              "usage_by_stage": {stage: {key: sum(r["usage"].get(stage, {}).get(key, 0)
                                                   for r in records)
                                         for key in ("calls", "input_tokens", "output_tokens")}
                                 for stage in ("intent", "obligations")},
              "methods": {}}
    for name in methods:
        report["methods"][name] = {
            "reference_execution_top1": sum(r["execution_correct"][name] for r in records),
            "reference_execution_covered_top1": sum(r["covered"] and r["execution_correct"][name]
                                                    for r in records),
            "rescues_vs_raw": sum(r["execution_correct"][name] and
                                  not r["execution_correct"]["raw_jev"] for r in records),
            "regressions_vs_raw": sum(r["execution_correct"]["raw_jev"] and
                                      not r["execution_correct"][name] for r in records),
            "changed_choices": sum(r["choices"][name] != r["choices"]["raw_jev"]
                                   for r in records),
            "abstained_no_valid": sum(r["choices"][name] is None for r in records),
            "changed_vs_local_evidence": sum(r["choices"][name] !=
                                             r["choices"]["structure_execution"]
                                             for r in records),
        }
    if args.semantic_review:
        review = json.loads(Path(args.semantic_review).read_text(encoding="utf-8"))
        annotations = {int(item["question_id"]): item for item in review["cases"]}
        if len(annotations) != len(review["cases"]):
            raise ValueError("Duplicate semantic-review question ID")
        report["semantic_review"] = {
            "review_type": review["review_type"],
            "reviewed_questions": sum(r["question_id"] in annotations for r in records),
            "methods": {},
        }
        for name in methods:
            statuses = Counter()
            for record in records:
                ann = annotations.get(record["question_id"])
                if ann is None:
                    statuses["unreviewed"] += 1
                    continue
                choice = record["choices"][name]
                if choice is None:
                    status = "no_candidate"
                elif name != "raw_jev" and choice != record["choices"]["raw_jev"]:
                    status = ann["status"] if choice in ann.get("alternative_indices", []) else "unreviewed"
                else:
                    status = ann["status"]
                statuses[status] += 1
            if sum(statuses.values()) != len(records):
                raise AssertionError("Semantic statuses do not cover evaluated questions")
            determined = statuses["correct"] + statuses["wrong"]
            report["semantic_review"]["methods"][name] = {
                "confirmed_correct": statuses["correct"],
                "confirmed_wrong": statuses["wrong"],
                "determinate_denominator": determined,
                "uncertain": statuses["uncertain"],
                "no_candidate": statuses["no_candidate"],
                "unreviewed": statuses["unreviewed"],
                "conditional_accuracy": statuses["correct"] / determined if determined else None,
            }
    _save(args.output, report)
    Path(args.output).with_suffix(".jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "missing"},
                     ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--input", required=True); c.add_argument("--split-file", required=True)
    c.add_argument("--split", default="validation")
    c.add_argument("--execution-dir", required=True); c.add_argument("--output-dir", required=True)
    c.add_argument("--limit", type=int, default=0); c.add_argument("--attempts", type=int, default=2)
    c.add_argument("--shard-count", type=int, default=1); c.add_argument("--shard-index", type=int, default=0)
    c.set_defaults(func=collect)
    e = sub.add_parser("evaluate")
    e.add_argument("--input", required=True); e.add_argument("--split-file", required=True)
    e.add_argument("--split", default="validation")
    e.add_argument("--execution-dir", required=True); e.add_argument("--collection-dir", required=True)
    e.add_argument("--model", required=True); e.add_argument("--output", required=True)
    e.add_argument("--semantic-review")
    e.add_argument("--allow-partial", action="store_true")
    e.set_defaults(func=evaluate)
    args = parser.parse_args()
    if args.command == "collect" and (args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count):
        parser.error("invalid shard configuration")
    args.func(args)


if __name__ == "__main__":
    main()
