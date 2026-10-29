"""Frozen-pool multi-angle JEV selection experiment.

Collection never sends labels or reference SQL to JEV. Fit uses train labels;
threshold choice uses validation labels; evaluation only reads the held-out split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize

from alphasql.jev.client import JevClient, JevTimeoutError
from alphasql.jev.models import NoulQuestion, NoulResult
from alphasql.jev.runtime import JevSettings
from alphasql.llm_call.runtime import configure_environment
from alphasql.algorithm.selection.training_data import label_trace


VIEWS = {
    "forward_entities": "Does SQL {i} retrieve the entities and relationships requested by every part of the question? A successful execution alone is insufficient.",
    "forward_output": "Does SQL {i} return exactly the requested attributes or measure, with the requested granularity and multiplicity?",
    "forward_conditions": "Does SQL {i} implement every restriction and boundary in the question and evidence, including filters that may be missing from this SQL?",
    "reverse_sources": "Are every table and JOIN in SQL {i} justified by the question, evidence and schema, with correct keys and no unintended row multiplication?",
    "reverse_predicates": "Are every predicate and literal in SQL {i} supported by the question or evidence, with no extra or contradictory restriction?",
    "reverse_operations": "Are every aggregation, grouping, ordering, LIMIT, DISTINCT and set operation in SQL {i} justified, with no extra or contradictory operation?",
    "critical": "Considering projection, filters, joins, aggregation, ordering, LIMIT, DISTINCT, NULLs and ties where relevant, is SQL {i} fully correct?",
}
FEATURES = ("candidate_raw", *VIEWS)


def _rows(path: str) -> list[dict]:
    source = Path(path)
    if source.is_dir():
        return [json.loads(file.read_text(encoding="utf-8"))
                for file in sorted(source.glob("[0-9]*.json"), key=lambda p: int(p.stem))]
    return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]


def _valid(row: dict) -> list[int]:
    return [i for i, c in enumerate(row["rounds"][0]["candidates"])
            if c.get("execution_rechecked_valid", c.get("execution", {}).get("valid", False))]


def _identity(row: dict) -> str:
    state = row["rounds"][0]
    data = {"db_id": row["db_id"], "question_id": row["question_id"],
            "sql": [c["sql"] for c in state["candidates"]]}
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _collect_one(client: JevClient, row: dict, pairwise: bool,
                 checkpoint: Path | None = None, existing: dict | None = None) -> dict:
    indices = _valid(row)
    candidates = row["rounds"][0]["candidates"]
    base = existing or {"question_id": row["question_id"], "db_id": row["db_id"],
            "pool_sha256": _identity(row), "valid_indices": indices,
            "views": {}, "pairs": {}, "calls": 0,
            "input_tokens": 0, "output_tokens": 0,
            "view_input_tokens": 0, "view_output_tokens": 0,
            "pair_input_tokens": 0, "pair_output_tokens": 0}
    if not indices:
        return base
    state = {"task": "Judge existing SQL only; do not generate or revise SQL.",
             "question": row["question"], "evidence": row.get("evidence") or "",
             "schema": row["full_schema"],
             "candidates": [{"index": i, "sql": candidates[i]["sql"],
                              "execution": candidates[i]["execution"]} for i in indices]}
    questions = {f"{view}_{i}": NoulQuestion(template.format(i=i))
                 for i in indices for view, template in VIEWS.items()}
    if not base["views"]:
        response = client.ask(state, questions)
        base["views"] = {str(i): {view: response.require(f"{view}_{i}", NoulResult).noul
                                   for view in VIEWS} for i in indices}
        base["calls"] = 1
        base["input_tokens"] = response.usage.input_tokens
        base["output_tokens"] = response.usage.output_tokens
        base["view_input_tokens"] = response.usage.input_tokens
        base["view_output_tokens"] = response.usage.output_tokens
        base["model"] = response.model
        if checkpoint:
            _save(checkpoint, base)
    if pairwise and len(indices) > 1 and not base["pairs"]:
        # Compare all first-pool candidates in both orders to expose position bias.
        pairs = [(i, j) for p, i in enumerate(indices) for j in indices[p + 1:]]
        questions = {}
        for i, j in pairs:
            questions[f"pair_{i}_{j}"] = NoulQuestion(
                f"Which better answers the question: SQL {i} or SQL {j}? "
                f"Is SQL {i} more likely to be fully correct than SQL {j}?")
            questions[f"pair_{j}_{i}"] = NoulQuestion(
                f"Which better answers the question: SQL {j} or SQL {i}? "
                f"Is SQL {j} more likely to be fully correct than SQL {i}?")
        response = client.ask(state, questions)
        base["pairs"] = {f"{i}_{j}": [response.require(f"pair_{i}_{j}", NoulResult).noul,
                                           response.require(f"pair_{j}_{i}", NoulResult).noul]
                         for i, j in pairs}
        base["calls"] += 1
        base["input_tokens"] += response.usage.input_tokens
        base["output_tokens"] += response.usage.output_tokens
        base["pair_input_tokens"] = response.usage.input_tokens
        base["pair_output_tokens"] = response.usage.output_tokens
        if base["model"] != response.model:
            raise ValueError("JEV model changed within a pool")
    return base


def collect(args: argparse.Namespace) -> None:
    configure_environment()
    client = JevClient(JevSettings.from_environment())
    output = Path(args.output_dir)
    failures = []
    rows = _rows(args.input)
    for row in rows[:args.limit] if args.limit else rows:
        path = output / f'{row["db_id"]}_{row["question_id"]}.json'
        existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        if existing is not None:
            if existing["pool_sha256"] != _identity(row):
                raise ValueError(f"Frozen pool changed: {path}")
            if not args.pairwise or existing["pairs"] or len(_valid(row)) < 2:
                continue
        for attempt in range(args.attempts):
            try:
                existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
                result = _collect_one(client, row, args.pairwise, path, existing)
                _save(path, result)
                print(json.dumps({"question_id": row["question_id"], "calls": result["calls"]}), flush=True)
                break
            except JevTimeoutError as exc:
                if attempt + 1 < args.attempts:
                    time.sleep(min(8, 2 ** attempt))
                    continue
                failures.append({"db_id": row["db_id"], "question_id": row["question_id"],
                                 "type": type(exc).__name__, "message": str(exc)})
                print(json.dumps(failures[-1]), flush=True)
            except Exception as exc:
                failures.append({"db_id": row["db_id"], "question_id": row["question_id"],
                                 "type": type(exc).__name__, "message": str(exc)})
                print(json.dumps(failures[-1]), flush=True)
                break
    _save(output / f"failures_{Path(args.input).stem}.json", {"failures": failures})


def _features(row: dict, evidence: dict, i: int) -> np.ndarray:
    item = row["rounds"][0]["candidates"][i]
    views = evidence["views"][str(i)]
    return np.array([float(item["candidate_raw"])] + [float(views[v]) for v in VIEWS])


def _rule(x: np.ndarray) -> float:
    # Predeclared conservative rule; one absent requirement cannot be hidden by averages.
    forward = min(x[1], x[2], x[3])
    reverse = min(x[4], x[5], x[6])
    return float(.25 * x[0] + .30 * forward + .25 * reverse + .20 * x[7])


def _tournament(evidence: dict, indices: list[int], base_scores: dict[int, float]) -> dict[int, float]:
    scores = {i: 0.0 for i in indices}
    for key, pair in evidence["pairs"].items():
        i, j = map(int, key.split("_"))
        # p(i beats j) combines both display orders; a contradictory pair is 0.5.
        p = (float(pair[0]) + 1 - float(pair[1])) / 2
        scores[i] += p
        scores[j] += 1 - p
    n = max(1, len(indices) - 1)
    return {i: .65 * base_scores[i] + .35 * scores[i] / n for i in indices}


def _prepare(rows: list[dict], folder: Path) -> list[tuple[dict, dict]]:
    out = []
    for row in rows:
        path = folder / f'{row["db_id"]}_{row["question_id"]}.json'
        evidence = json.loads(path.read_text(encoding="utf-8"))
        if evidence["pool_sha256"] != _identity(row):
            raise ValueError(f"Pool changed: {path}")
        if len(_valid(row)) > 1 and not evidence["pairs"]:
            raise ValueError(f"Pairwise judgement missing: {path}")
        out.append((row, evidence))
    return out


def _fit(pools: list[tuple[dict, dict]], regularization: float = 1.0) -> np.ndarray:
    x, y, pairs = [], [], []
    for row, evidence in pools:
        start = len(x)
        for i in _valid(row):
            x.append(_features(row, evidence, i))
            y.append(int(row["rounds"][0]["candidates"][i]["execution_label"]))
        local = list(range(start, len(x)))
        pairs.extend((a, b) for a in local for b in local if y[a] == 1 and y[b] == 0)
    if not pairs:
        raise ValueError("No correct-versus-incorrect training pairs")
    x = np.asarray(x)
    delta = np.asarray([x[a] - x[b] for a, b in pairs])
    def objective(w):
        z = delta @ w
        return float(np.logaddexp(0, -z).mean() + regularization * np.dot(w, w) / 2)
    result = minimize(objective, np.zeros(x.shape[1]), method="L-BFGS-B")
    if not result.success:
        raise ValueError(result.message)
    return result.x


def _evaluate(pools: list[tuple[dict, dict]], weights: np.ndarray) -> tuple[dict, list[dict]]:
    names = ("jev", "rule", "learned", "tournament")
    records = []
    for row, evidence in pools:
        indices = _valid(row)
        cs = row["rounds"][0]["candidates"]
        covered = any(cs[i]["execution_label"] for i in indices)
        xs = {i: _features(row, evidence, i) for i in indices}
        scores = {
            "jev": {i: float(cs[i]["candidate_raw"]) for i in indices},
            "rule": {i: _rule(xs[i]) for i in indices},
            "learned": {i: float(xs[i] @ weights) for i in indices},
        }
        scores["tournament"] = _tournament(evidence, indices, scores["rule"])
        chosen = {name: max(indices, key=lambda i: (score[i], -i)) if indices else None
                  for name, score in scores.items()}
        records.append({"db_id": row["db_id"], "question_id": row["question_id"],
                        "covered": bool(covered),
                        "hard_for_jev": bool(covered and chosen["jev"] is not None
                                             and not cs[chosen["jev"]]["execution_label"]),
                        "correct": {name: int(chosen[name] is not None and
                                                      cs[chosen[name]]["execution_label"]) for name in names},
                        "selected": chosen,
                        "selected_score": {name: scores[name][chosen[name]] if chosen[name] is not None
                                           else None for name in names},
                        "calls": evidence["calls"],
                        "input_tokens": evidence["input_tokens"],
                        "output_tokens": evidence["output_tokens"]})
    summary = {name: {"total_correct": sum(r["correct"][name] for r in records),
                      "conditional_correct": sum(r["correct"][name] for r in records if r["covered"]),
                      "covered": sum(r["covered"] for r in records),
                      "hard_correct": sum(r["correct"][name] for r in records if r["hard_for_jev"]),
                      "hard_count": sum(r["hard_for_jev"] for r in records)} for name in names}
    separated = all("view_input_tokens" in e for _, e in pools)
    return {"questions": len(records), "methods": summary,
            "multi_angle_jev_calls": sum(1 for _, e in pools if e["calls"] > 0),
            "tournament_extra_calls": sum(1 for _, e in pools if e["pairs"]),
            "total_extra_jev_calls": sum(e["calls"] for _, e in pools),
            "extra_jev_input_tokens": sum(e["input_tokens"] for _, e in pools),
            "extra_jev_output_tokens": sum(e["output_tokens"] for _, e in pools),
            "view_tokens": ({"input": sum(e["view_input_tokens"] for _, e in pools),
                             "output": sum(e["view_output_tokens"] for _, e in pools)}
                            if separated else None),
            "pair_tokens": ({"input": sum(e["pair_input_tokens"] for _, e in pools),
                             "output": sum(e["pair_output_tokens"] for _, e in pools)}
                            if separated else None)}, records


def analyze(args: argparse.Namespace) -> None:
    train, val, test = [_prepare(_rows(path), Path(args.evidence_dir))
                        for path in (args.train, args.validation, args.test)]
    dbs = [{row["db_id"] for row, _ in split} for split in (train, val, test)]
    if dbs[0] & dbs[1] or dbs[0] & dbs[2] or dbs[1] & dbs[2]:
        raise ValueError("Train, validation and test databases must be disjoint")
    weights = _fit(train, args.regularization)
    validation, val_rows = _evaluate(val, weights)
    # Thresholds chosen solely on validation; no threshold retuning on test.
    thresholds = {}
    for name in validation["methods"]:
        options = sorted({r["selected_score"][name] for r in val_rows
                          if r["selected_score"][name] is not None}, reverse=True)
        eligible = []
        for threshold in options:
            accepted = [r for r in val_rows if r["selected_score"][name] is not None
                        and r["selected_score"][name] >= threshold]
            errors = sum(1 - r["correct"][name] for r in accepted)
            if accepted and errors / len(accepted) <= args.max_error:
                eligible.append((len(accepted), threshold))
        thresholds[name] = max(eligible)[1] if eligible else None
    test_summary, test_rows = _evaluate(test, weights)
    for name, threshold in thresholds.items():
        accepted = [r for r in test_rows if threshold is not None
                    and r["selected_score"][name] is not None
                    and r["selected_score"][name] >= threshold]
        test_summary["methods"][name]["stop"] = {
            "validation_threshold": threshold, "accepted": len(accepted),
            "wrong": sum(1 - r["correct"][name] for r in accepted),
            "wrong_fraction": (sum(1 - r["correct"][name] for r in accepted) / len(accepted)
                               if accepted else None)}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _save(output / "model.json", {"features": FEATURES, "weights": weights.tolist(),
                                  "training_databases": sorted(dbs[0]),
                                  "validation_databases": sorted(dbs[1]),
                                  "thresholds": thresholds})
    _save(output / "report.json", {"validation": validation, "test": test_summary,
                                   "test_databases": sorted(dbs[2]),
                                   "label_caveat": "Single-database execution agreement is a proxy"})
    (output / "test_per_question.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in test_rows), encoding="utf-8")
    print(json.dumps({"validation": validation, "test": test_summary}, ensure_ascii=False))


def label_holdout(args: argparse.Namespace) -> None:
    """Run only after online collection; reference SQL never enters JEV prompts."""
    manifest = json.loads(Path(args.protocol_dir, "manifest.json").read_text(encoding="utf-8"))
    dataset_path = Path(args.dataset)
    if hashlib.sha256(dataset_path.read_bytes()).hexdigest() != manifest["dataset_sha256"]:
        raise ValueError("Dataset changed after holdout was frozen")
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    labelled = []
    for qid in manifest["question_ids"]:
        if int(qid) % args.shard_count != args.shard_index:
            continue
        raw = json.loads(Path(args.raw_dir, f"{qid}.json").read_text(encoding="utf-8"))
        if args.require_evidence_dir and not Path(args.require_evidence_dir,
                f'{raw["db_id"]}_{qid}.json').is_file():
            raise ValueError(f"Online JEV evidence incomplete for {qid}")
        if raw["db_id"] != dataset[qid]["db_id"]:
            raise ValueError(f"Database mismatch for {qid}")
        db = raw["db_id"]
        db_path = Path(args.database_root_dir, db, f"{db}.sqlite")
        labelled.append(label_trace(raw, dataset[qid]["SQL"], str(db_path)))
    output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in labelled),
                      encoding="utf-8")
    print(json.dumps({"questions": len(labelled), "databases": manifest["selected_databases"]}))


def evaluate_holdout(args: argparse.Namespace) -> None:
    model = json.loads(Path(args.model).read_text(encoding="utf-8"))
    if tuple(model["features"]) != FEATURES:
        raise ValueError("Feature definition changed")
    pools = _prepare(_rows(args.input), Path(args.evidence_dir))
    dbs = {r["db_id"] for r, _ in pools}
    if dbs & set(model["training_databases"]) or dbs & set(model["validation_databases"]):
        raise ValueError("Holdout databases overlap model fitting databases")
    summary, records = _evaluate(pools, np.asarray(model["weights"]))
    for name, threshold in model["thresholds"].items():
        accepted = [r for r in records if threshold is not None
                    and r["selected_score"][name] is not None
                    and r["selected_score"][name] >= threshold]
        wrong = sum(1 - r["correct"][name] for r in accepted)
        summary["methods"][name]["stop"] = {"validation_threshold": threshold,
                                               "accepted": len(accepted), "wrong": wrong,
                                               "wrong_fraction": wrong / len(accepted) if accepted else None}
    summary["databases"] = sorted(dbs)
    summary["paired_vs_jev"] = {
        name: {
            "rescued": sum(r["correct"][name] and not r["correct"]["jev"] for r in records),
            "regressed": sum(r["correct"]["jev"] and not r["correct"][name] for r in records),
        }
        for name in ("rule", "learned", "tournament")
    }
    summary["by_database"] = {
        db: {"questions": sum(r["db_id"] == db for r in records),
             "covered": sum(r["db_id"] == db and r["covered"] for r in records),
             "correct": {name: sum(r["db_id"] == db and r["correct"][name]
                              for r in records) for name in summary["methods"]}}
        for db in sorted(dbs)
    }
    summary["label_caveat"] = "Single-database execution agreement is a proxy for semantic correctness"
    if args.raw_dir:
        raw = [json.loads(Path(args.raw_dir, f'{r["question_id"]}.json').read_text(encoding="utf-8"))
               for r in records]
        summary["baseline_usage"] = {
            "llm_calls": sum(x["llm_usage"]["calls"] for x in raw),
            "llm_input_tokens": sum(x["llm_usage"]["input_tokens"] for x in raw),
            "llm_output_tokens": sum(x["llm_usage"]["output_tokens"] for x in raw),
            "jev_calls": sum(x["jev_calls_total"] for x in raw),
            "jev_input_tokens": sum(x["jev_usage_total"]["input_tokens"] for x in raw),
            "jev_output_tokens": sum(x["jev_usage_total"]["output_tokens"] for x in raw),
        }
    _save(Path(args.output), summary)
    per_question = (Path(args.per_question_output) if args.per_question_output
                    else Path(args.output).with_name(f"{Path(args.output).stem}_per_question.jsonl"))
    per_question.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--input", required=True)
    c.add_argument("--output-dir", required=True)
    c.add_argument("--limit", type=int)
    c.add_argument("--pairwise", action="store_true")
    c.add_argument("--attempts", type=int, default=3)
    c.set_defaults(func=collect)
    a = sub.add_parser("analyze")
    a.add_argument("--train", required=True)
    a.add_argument("--validation", required=True)
    a.add_argument("--test", required=True)
    a.add_argument("--evidence-dir", required=True)
    a.add_argument("--output-dir", required=True)
    a.add_argument("--regularization", type=float, default=1.0)
    a.add_argument("--max-error", type=float, default=.1)
    a.set_defaults(func=analyze)
    h = sub.add_parser("label-holdout")
    h.add_argument("--protocol-dir", required=True)
    h.add_argument("--raw-dir", required=True)
    h.add_argument("--dataset", default="data/bird/train/train.json")
    h.add_argument("--database-root-dir", default="data/bird/train/train_databases")
    h.add_argument("--output", required=True)
    h.add_argument("--require-evidence-dir")
    h.add_argument("--shard-count", type=int, default=1)
    h.add_argument("--shard-index", type=int, default=0)
    h.set_defaults(func=label_holdout)
    e = sub.add_parser("evaluate-holdout")
    e.add_argument("--input", required=True)
    e.add_argument("--evidence-dir", required=True)
    e.add_argument("--model", required=True)
    e.add_argument("--output", required=True)
    e.add_argument("--raw-dir", help="Frozen raw traces for baseline request and token totals")
    e.add_argument("--per-question-output", help="Optional path for per-question selections")
    e.set_defaults(func=evaluate_holdout)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
