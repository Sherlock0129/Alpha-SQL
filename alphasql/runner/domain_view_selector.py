"""Frozen-pool domain checks and execution self-consistency comparison.

Collection is label-free. TypeSafe Noul returns probabilities only, so claimed
error locations are separate typed choices over numbered question fragments and
SQL clauses. These locations are diagnostic claims, not verified error labels.
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
from alphasql.runner.selection_v2 import _read, _save


DOMAINS = ("schema", "conditions", "output", "order_set")
CHECKS = {
    "schema_missing": "Does SQL {i} cover every entity, relation and join needed by the question and evidence, using the correct schema columns and join keys?",
    "schema_extra": "Are all tables, columns and joins in SQL {i} justified by the question and evidence, without a wrong join or unintended row multiplication?",
    "conditions_missing": "Does SQL {i} implement every requested filter, comparison, boundary, NULL condition and WHERE/HAVING restriction?",
    "conditions_extra": "Are all predicates and literal values in SQL {i} supported by the question and evidence, without extra or contradictory restrictions?",
    "output_missing": "Does SQL {i} return every requested attribute or measure at the required entity granularity, including aggregation and count-distinct semantics?",
    "output_extra": "Does SQL {i} avoid extra projected columns, duplicate rows and unsupported aggregation or grouping that change the requested output?",
    "order_set_missing": "Where relevant, does SQL {i} implement every requested ordering, top-k, tie, DISTINCT and set-operation requirement? If none is requested, answer yes.",
    "order_set_extra": "Where relevant, are SQL {i}'s ORDER BY, LIMIT, DISTINCT and set operations justified, including their direction and tie behavior? If none is present, answer yes.",
    "overall": "Considering the question, evidence, schema and execution summary, is SQL {i} fully correct? Executability alone is insufficient.",
}
PROMPT_SHA256 = hashlib.sha256(json.dumps(CHECKS, sort_keys=True).encode()).hexdigest()
RAW_WEIGHT = 0.75  # Fixed before comparing methods; not a calibrated probability.
PAIR_MARGIN = 0.12
LOCATION_THRESHOLD = 0.5


def _fragments(question: str) -> dict[str, str]:
    parts = [p.strip(" ,;?") for p in re.split(
        r"[,;?]|\b(?:where|whose|who|which|that|with|among|having)\b",
        question, flags=re.I) if p.strip(" ,;?")]
    if not parts:
        parts = [question.strip()]
    if len(parts) > 10:
        parts = parts[:9] + [" ".join(parts[9:])]
    return {f"q{n}": part for n, part in enumerate(parts)}


def _clauses(sql: str) -> dict[str, str]:
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return {"s0": sql[:300]}
    chunks: list[str] = []
    for select in tree.find_all(exp.Select):
        chunks.extend("SELECT " + item.sql(dialect="sqlite") for item in select.expressions)
        for name, kind in (("from_", "FROM"), ("where", "WHERE"), ("group", "GROUP BY"),
                           ("having", "HAVING"), ("order", "ORDER BY"), ("limit", "LIMIT")):
            node = select.args.get(name)
            if node:
                chunks.append(f"{kind} {node.sql(dialect='sqlite')}")
        if select.args.get("distinct"):
            chunks.append("DISTINCT")
    chunks.extend("JOIN " + node.sql(dialect="sqlite") for node in tree.find_all(exp.Join))
    chunks.extend(node.sql(dialect="sqlite") for node in tree.find_all(exp.SetOperation))
    chunks = list(dict.fromkeys(chunks))
    if not chunks:
        chunks = [sql]
    if len(chunks) > 20:
        chunks = chunks[:19] + ["OTHER: " + sql[:200]]
    return {f"s{n}": chunk[:300] for n, chunk in enumerate(chunks)}


def _valid(row: Mapping[str, Any]) -> list[int]:
    return [i for i, item in enumerate(row["rounds"][0]["candidates"])
            if item.get("execution_rechecked_valid", item["execution"]["valid"])]


def _state(row: Mapping[str, Any], indices: list[int]) -> dict[str, Any]:
    candidates = row["rounds"][0]["candidates"]
    return {"task": "Judge existing SQL only; do not create or revise SQL. Gold answers and labels are unavailable.",
            "question": row["question"], "evidence": row.get("evidence") or "",
            "schema": row["full_schema"], "question_fragments": _fragments(row["question"]),
            "candidates": [{"index": i, "sql": candidates[i]["sql"],
                            "sql_clauses": _clauses(candidates[i]["sql"]),
                            "execution": candidates[i]["execution"]} for i in indices]}


def _usage(base: dict[str, Any], response: Any, stage: str) -> None:
    value = {"calls": 1, "input_tokens": response.usage.input_tokens,
             "output_tokens": response.usage.output_tokens}
    base["usage"][stage] = value
    if base.get("model") and base["model"] != response.model:
        raise ValueError("JEV model changed within a pool")
    base["model"] = response.model


def _domain_score(checks: Mapping[str, float]) -> float:
    return sum(float(checks[name]) for name in CHECKS if name != "overall") / 8


def _anchored_scores(row: Mapping[str, Any], base: Mapping[str, Any]) -> dict[int, float]:
    candidates = row["rounds"][0]["candidates"]
    return {i: RAW_WEIGHT * float(candidates[i]["candidate_raw"])
            + (1 - RAW_WEIGHT) * _domain_score(base["checks"][str(i)])
            for i in _valid(row)}


def _collect_one(client: JevClient, row: dict[str, Any], path: Path,
                 execution: Mapping[str, Any] | None = None,
                 pairwise: bool = True) -> dict[str, Any]:
    indices = _valid(row)
    base = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {
        "db_id": row["db_id"], "question_id": row["question_id"],
        "pool_sha256": pool_identity(row), "prompt_sha256": PROMPT_SHA256,
        "valid_indices": indices, "checks": {}, "locations": {}, "pairwise": {}, "usage": {}}
    if base["pool_sha256"] != pool_identity(row) or base["prompt_sha256"] != PROMPT_SHA256:
        raise ValueError(f"Frozen pool or prompt changed: {path}")
    if not indices:
        base["location_stage_done"] = True
        base["pairwise_stage_done"] = True
        return base
    state = _state(row, indices)
    if not base["checks"]:
        questions = {f"{name}_{i}": NoulQuestion(template.format(i=i))
                     for i in indices for name, template in CHECKS.items()}
        response = client.ask(state, questions)
        base["checks"] = {str(i): {name: response.require(f"{name}_{i}", NoulResult).noul
                                    for name in CHECKS} for i in indices}
        _usage(base, response, "checks")
        _save(path, base)
    if not base["locations"]:
        questions = {}
        locations: dict[str, dict[str, Any]] = {}
        for i in indices:
            low = [domain for domain in DOMAINS if min(
                base["checks"][str(i)][domain + "_missing"],
                base["checks"][str(i)][domain + "_extra"]) < LOCATION_THRESHOLD]
            if not low:
                continue
            clauses = state["candidates"][indices.index(i)]["sql_clauses"]
            locations[str(i)] = {}
            for domain in low:
                locations[str(i)][domain] = {}
                questions[f"q_{domain}_{i}"] = ChoiceQuestion(
                    {**state["question_fragments"], "none": "No applicable fragment in the question"},
                    f"For SQL {i}, choose the question fragment most relevant to the suspected {domain} error. If the SQL has unsupported extra logic, choose the fragment it falsely appears to implement, or none.")
                questions[f"s_{domain}_{i}"] = ChoiceQuestion(
                    {**clauses, "missing": "Required SQL clause is absent"},
                    f"For SQL {i}, choose the exact SQL clause most relevant to the suspected {domain} error; choose missing when the required clause is absent.")
        if questions:
            response = client.ask(state, questions)
            for i, groups in locations.items():
                for domain in groups:
                    q = response.require(f"q_{domain}_{i}", ChoiceResult).choice
                    s = response.require(f"s_{domain}_{i}", ChoiceResult).choice
                    clauses = state["candidates"][indices.index(int(i))]["sql_clauses"]
                    groups[domain] = {"question_fragment_id": q,
                                      "question_fragment": state["question_fragments"].get(q),
                                      "sql_clause_id": s, "sql_clause": clauses.get(s),
                                      "jev_claim_only": True}
            _usage(base, response, "locations")
        base["locations"] = locations
        base["location_stage_done"] = True
        _save(path, base)
    if pairwise and len(indices) > 1 and not base.get("pairwise_stage_done"):
        scores = _anchored_scores(row, base)
        ordered = sorted(indices, key=lambda i: (-scores[i], i))
        a, b = ordered[:2]
        raw_top = max(indices, key=lambda i: (float(row["rounds"][0]["candidates"][i]["candidate_raw"]), -i))
        different_result = execution is None or (
            execution["candidates"][a].get("result_hash") !=
            execution["candidates"][b].get("result_hash"))
        triggered = different_result and (scores[a] - scores[b] <= PAIR_MARGIN or a != raw_top)
        if triggered:
            questions = {
                "a_over_b": NoulQuestion(f"Comparing SQL {a} against SQL {b}, is SQL {a} more likely to fully answer the question? Inspect the concrete difference in schema, conditions, output and ordering, not just whether either query executes."),
                "b_over_a": NoulQuestion(f"Comparing SQL {b} against SQL {a}, is SQL {b} more likely to fully answer the question? Inspect the concrete difference in schema, conditions, output and ordering, not just whether either query executes."),
            }
            response = client.ask(state, questions)
            p = (response.require("a_over_b", NoulResult).noul +
                 1 - response.require("b_over_a", NoulResult).noul) / 2
            base["pairwise"] = {"first": a, "second": b, "first_preferred": p,
                                "trigger": "close_or_raw_conflict", "different_result": True}
            _usage(base, response, "pairwise")
        base["pairwise_stage_done"] = True
        _save(path, base)
    return base


def collect(args: argparse.Namespace) -> None:
    configure_environment()
    client = JevClient(JevSettings.from_environment())
    rows = _read(args.input)
    if args.split_file:
        protocol = json.loads(Path(args.split_file).read_text(encoding="utf-8"))
        allowed = set(protocol["question_ids"][args.split])
        rows = [row for row in rows if int(row["question_id"]) in allowed]
    output = Path(args.output_dir)
    failures = []
    done = attempted = 0
    for row in rows:
        if int(row["question_id"]) % args.shard_count != args.shard_index:
            continue
        path = output / f'{row["db_id"]}_{row["question_id"]}.json'
        execution = None
        if args.execution_dir:
            source = Path(args.execution_dir) / path.name
            execution = json.loads(source.read_text(encoding="utf-8"))
            if execution["pool_sha256"] != pool_identity(row):
                raise ValueError(f"Execution fingerprint mismatch: {source}")
        if path.is_file():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved["pool_sha256"] != pool_identity(row) or saved["prompt_sha256"] != PROMPT_SHA256:
                raise ValueError(f"Frozen pool or prompt changed: {path}")
            if saved.get("location_stage_done") and (not args.pairwise or saved.get("pairwise_stage_done")):
                continue
        if args.limit and attempted >= args.limit:
            break
        attempted += 1
        for attempt in range(args.attempts):
            try:
                result = _collect_one(client, row, path, execution, args.pairwise)
                _save(path, result)
                print(json.dumps({"question_id": row["question_id"], "usage": result["usage"]}), flush=True)
                done += 1
                break
            except JevTimeoutError as exc:
                if attempt + 1 < args.attempts:
                    time.sleep(min(8, 2 ** attempt))
                    continue
                failures.append({"question_id": row["question_id"], "error": str(exc)})
            except Exception as exc:
                failures.append({"question_id": row["question_id"],
                                 "error": f"{type(exc).__name__}: {exc}"})
                break
    _save(output / "failures.json", {"failures": failures})
    print(json.dumps({"attempted_this_run": attempted, "completed_this_run": done,
                      "failures": len(failures)}))


def _select(scores: Mapping[int, float], raw: Mapping[int, float]) -> int | None:
    return max(scores, key=lambda i: (scores[i], raw[i], -i)) if scores else None


def _self_consistency(indices: list[int], execution: Mapping[str, Any],
                      raw: Mapping[int, float], route_diverse: bool = False) -> int | None:
    if not indices:
        return None
    clusters = []
    for cluster in execution["clusters"]:
        members = [i for i in cluster["indices"] if i in indices]
        if members:
            support = len(set(execution["candidates"][i]["route"] for i in members)) if route_diverse else len(members)
            representative = max(members, key=lambda i: (raw[i], -i))
            clusters.append((support, raw[representative], -representative, representative))
    if not clusters:
        raise ValueError("Valid candidates have no result clusters")
    return max(clusters)[-1]


def _decisions(row: Mapping[str, Any], result: Mapping[str, Any],
               execution: Mapping[str, Any]) -> dict[str, int | None]:
    indices = _valid(row)
    cs = row["rounds"][0]["candidates"]
    raw = {i: float(cs[i]["candidate_raw"]) for i in indices}
    domain = {i: _domain_score(result["checks"][str(i)]) for i in indices}
    anchor = {i: RAW_WEIGHT * raw[i] + (1 - RAW_WEIGHT) * domain[i] for i in indices}
    selected = {"raw_jev": _select(raw, raw),
                "domain_mean": _select(domain, raw),
                "domain_anchored": _select(anchor, raw),
                "self_consistency": _self_consistency(indices, execution, raw),
                "route_consistency": _self_consistency(indices, execution, raw, True)}
    pair = result.get("pairwise") or {}
    selected["domain_pairwise"] = (pair["second"] if pair.get("first_preferred", .5) < .4
                                   else selected["domain_anchored"])
    return selected


def _baseline_decisions(row: Mapping[str, Any], execution: Mapping[str, Any]) -> dict[str, int | None]:
    indices = _valid(row)
    cs = row["rounds"][0]["candidates"]
    raw = {i: float(cs[i]["candidate_raw"]) for i in indices}
    return {"raw_jev": _select(raw, raw),
            "self_consistency": _self_consistency(indices, execution, raw),
            "route_consistency": _self_consistency(indices, execution, raw, True)}


def evaluate(args: argparse.Namespace) -> None:
    rows = _read(args.input)
    if args.split_file:
        protocol = json.loads(Path(args.split_file).read_text(encoding="utf-8"))
        allowed = set(protocol["question_ids"][args.split])
        rows = [row for row in rows if int(row["question_id"]) in allowed]
    records = []
    usage = Counter()
    missing = []
    for row in rows:
        stem = f'{row["db_id"]}_{row["question_id"]}.json'
        path = Path(args.collection_dir) / stem if args.collection_dir else None
        if not args.baseline_only and (path is None or not path.is_file()):
            missing.append(stem)
            continue
        result = json.loads(path.read_text(encoding="utf-8")) if path and not args.baseline_only else None
        execution = json.loads((Path(args.execution_dir) / stem).read_text(encoding="utf-8"))
        expected = pool_identity(row)
        if execution["pool_sha256"] != expected or (result and result["pool_sha256"] != expected):
            raise ValueError(f"Fingerprint mismatch: {stem}")
        if result and result["prompt_sha256"] != PROMPT_SHA256:
            raise ValueError(f"Prompt version mismatch: {stem}")
        if result and not result.get("location_stage_done"):
            missing.append(stem)
            continue
        if result:
            for stage, value in result["usage"].items():
                for key in ("calls", "input_tokens", "output_tokens"):
                    usage[f"{stage}_{key}"] += value[key]
        cs = row["rounds"][0]["candidates"]
        indices = _valid(row)
        if any(cs[i].get("execution_label") is None for i in indices):
            raise ValueError("Offline labels are required only for evaluation")
        correct = {i for i in indices if cs[i]["execution_label"]}
        choices = (_decisions(row, result, execution) if result else
                   _baseline_decisions(row, execution))
        records.append({"db_id": row["db_id"], "question_id": row["question_id"],
                        "covered": bool(correct), "mixed": bool(correct and len(correct) < len(indices)),
                        "choices": choices, "correct": {name: int(index in correct)
                                                   for name, index in choices.items()},
                        "pairwise_triggered": bool(result and result.get("pairwise")),
                        "locations": result["locations"] if result else {}})
    if missing and not args.allow_partial:
        raise ValueError(f"Collection incomplete: {len(missing)} missing; first {missing[:3]}")
    methods = records[0]["choices"] if records else {}
    report = {"split": args.split, "questions_requested": len(rows),
              "questions_evaluated": len(records), "missing": missing,
              "covered": sum(r["covered"] for r in records),
              "mixed": sum(r["mixed"] for r in records),
              "pairwise_triggered": sum(r["pairwise_triggered"] for r in records),
              "usage": dict(usage), "methods": {}}
    for name in methods:
        report["methods"][name] = {
            "all_top1": sum(r["correct"][name] for r in records),
            "covered_top1": sum(r["covered"] and r["correct"][name] for r in records),
            "mixed_top1": sum(r["mixed"] and r["correct"][name] for r in records),
            "rescues_vs_raw": sum(r["correct"][name] and not r["correct"]["raw_jev"] for r in records),
            "regressions_vs_raw": sum(r["correct"]["raw_jev"] and not r["correct"][name] for r in records),
        }
    _save(args.output, report)
    Path(args.output).with_suffix(".jsonl").write_text("".join(
        json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"missing"}}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--input", required=True); c.add_argument("--output-dir", required=True)
    c.add_argument("--execution-dir"); c.add_argument("--split-file")
    c.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    c.add_argument("--limit", type=int, default=0); c.add_argument("--attempts", type=int, default=2)
    c.add_argument("--shard-count", type=int, default=1); c.add_argument("--shard-index", type=int, default=0)
    c.add_argument("--pairwise", action="store_true")
    c.set_defaults(func=collect)
    e = sub.add_parser("evaluate")
    e.add_argument("--input", required=True); e.add_argument("--collection-dir")
    e.add_argument("--execution-dir", required=True); e.add_argument("--split-file")
    e.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    e.add_argument("--allow-partial", action="store_true"); e.add_argument("--baseline-only", action="store_true")
    e.add_argument("--output", required=True)
    e.set_defaults(func=evaluate)
    args = parser.parse_args()
    if args.command == "collect" and args.shard_count < 1:
        parser.error("shard-count must be positive")
    if args.command == "collect" and not 0 <= args.shard_index < args.shard_count:
        parser.error("shard-index out of range")
    if args.command == "evaluate" and not args.baseline_only and not args.collection_dir:
        parser.error("collection-dir is required unless baseline-only is set")
    args.func(args)


if __name__ == "__main__":
    main()
