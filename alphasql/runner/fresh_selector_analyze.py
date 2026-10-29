"""Label frozen paired branches offline and compare accuracy, risk and cost."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from alphasql.algorithm.compiler.validation import validate_sql
from alphasql.algorithm.selection.content_verifier import ContentVerifier, content_features
from alphasql.algorithm.selection.selective_stop import StopPolicy
from alphasql.algorithm.selection.training_data import _execution_signature, _gold_requires_order
from alphasql.runner.fresh_selector_branch import decision


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _score_verifier(model: ContentVerifier, row: dict, candidates: list[dict]) -> int | None:
    valid = [i for i, item in enumerate(candidates) if item["execution"]["valid"]]
    if not valid:
        return None
    records = [{"features": content_features(row["question"], row.get("evidence") or "",
                                         row["full_schema"], candidates[i]["sql"],
                                         candidates[i]["execution"])} for i in valid]
    scores = model.predict(records)
    return valid[max(range(len(valid)), key=lambda j: (scores[j], -valid[j]))]


def _best_first(candidates: list[dict]) -> int | None:
    valid = [i for i, item in enumerate(candidates) if item["execution"]["valid"]]
    return max(valid, key=lambda i: (float(candidates[i]["candidate_raw"]), -i)) if valid else None


def _sum_usage(*items: dict) -> dict[str, int]:
    return {"input_tokens": sum(int(item.get("input_tokens") or 0) for item in items),
            "output_tokens": sum(int(item.get("output_tokens") or 0) for item in items)}


def _summary(rows: list[dict], arm: str) -> dict:
    selected = sum(item[arm]["correct"] for item in rows)
    accepted = [item for item in rows if item[arm]["direct_accepted"]]
    wrong = sum(1 - item[arm]["correct"] for item in accepted)
    llm_calls = sum(item[arm]["llm_calls"] for item in rows)
    jev_calls = sum(item[arm]["jev_calls"] for item in rows)
    return {"questions": len(rows), "selected_correct": selected,
            "selected_accuracy": selected / len(rows),
            "direct_accepted": len(accepted), "direct_wrong": wrong,
            "direct_error_fraction": wrong / len(accepted) if accepted else None,
            "direct_acceptance_fraction": len(accepted) / len(rows),
            "final_pool_covered": sum(item[arm]["covered"] for item in rows),
            "llm_calls": llm_calls, "jev_calls": jev_calls,
            "llm_calls_per_question": llm_calls / len(rows),
            "jev_calls_per_question": jev_calls / len(rows),
            "llm_tokens": _sum_usage(*(item[arm]["llm_usage"] for item in rows)),
            "jev_tokens": _sum_usage(*(item[arm]["jev_usage"] for item in rows))}


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float] | None:
    if not total:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    middle = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [middle - radius, middle + radius]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-dir", required=True)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--branch-dir", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--verifier-model", required=True)
    parser.add_argument("--data-path", default="data/bird/train/train.json")
    parser.add_argument("--database-root-dir", default="data/bird/train/train_databases")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    manifest = _read(Path(args.protocol_dir) / "manifest.json")
    policy = StopPolicy.load(args.policy)
    model = ContentVerifier.load(args.verifier_model)
    data_path = Path(args.data_path)
    if hashlib.sha256(data_path.read_bytes()).hexdigest() != manifest["dataset_sha256"]:
        raise ValueError("Reference dataset changed after the protocol was frozen")
    dataset = json.loads(data_path.read_text(encoding="utf-8"))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    evaluated = []
    for qid in manifest["question_ids"]:
        baseline = _read(Path(args.baseline_dir) / f"{qid}.json")
        if baseline["db_id"] not in manifest["selected_databases"]:
            raise ValueError("Database outside frozen protocol")
        if dataset[qid]["db_id"] != baseline["db_id"]:
            raise ValueError("Reference database differs from the collected question")
        first = baseline["rounds"][0]
        first_candidates = first["candidates"]
        action = decision(first, policy)
        gold_sql = dataset[qid].get("SQL") or dataset[qid].get("sql")
        db_id = baseline["db_id"]
        db_path = Path(args.database_root_dir) / db_id / f"{db_id}.sqlite"
        gold = validate_sql(str(db_path), gold_sql, 60)
        if not gold.valid:
            raise ValueError(f"Reference SQL became invalid for question {qid}")
        ordered = _gold_requires_order(gold_sql)
        reference = _execution_signature(gold.rows, ordered)
        label_cache = {}

        def correct(sql: str | None) -> int:
            if not sql:
                return 0
            if sql not in label_cache:
                result = validate_sql(str(db_path), sql, 60)
                label_cache[sql] = int(result.valid and
                                       _execution_signature(result.rows, ordered) == reference)
            return label_cache[sql]

        first_best = _best_first(first_candidates)
        first_sql = first_candidates[first_best]["sql"] if first_best is not None else None
        baseline_candidates = baseline["candidates"]
        baseline_sql = baseline.get("selected_sql")
        baseline_direct = (first_best is not None
                           and first["jev"]["pool_covered"] >= .8
                           and first_candidates[first_best]["candidate_raw"] >= .8)
        first_llm = baseline["llm_usage"]["rounds"][0]
        first_jev = first["jev"]["usage"]
        baseline_jev_calls = sum(int(state["jev"]["called"]) for state in baseline["rounds"])
        old = {"correct": correct(baseline_sql), "covered": int(any(correct(c["sql"])
               for c in baseline_candidates)), "direct_accepted": baseline_direct,
               "llm_calls": int(baseline["llm_usage"]["calls"]),
               "jev_calls": baseline_jev_calls,
               "llm_usage": baseline["llm_usage"], "jev_usage": baseline["jev_usage_total"]}
        review = None
        if action == "adjudicate":
            review = _read(Path(args.branch_dir) / "review" / f"{qid}.json")
            new_candidates, new_sql = first_candidates, first_sql
            llm_calls, jev_calls = int(first_llm["calls"]), int(first["jev"]["called"]) + int(review["called"])
            llm_usage, jev_usage = first_llm, _sum_usage(first_jev, review["usage"])
        elif action == "stop":
            new_candidates, new_sql = first_candidates, first_sql
            llm_calls, jev_calls = int(first_llm["calls"]), int(first["jev"]["called"])
            llm_usage, jev_usage = first_llm, first_jev
        elif len(baseline["rounds"]) > 1 or first["pool_size"] >= 16:
            new_candidates, new_sql = baseline_candidates, baseline_sql
            llm_calls, jev_calls = old["llm_calls"], baseline_jev_calls
            llm_usage, jev_usage = old["llm_usage"], old["jev_usage"]
        else:
            supplemental = _read(Path(args.branch_dir) / "supplement" / f"{qid}.json")
            new_candidates = supplemental["candidates"]
            index = supplemental["selected_index"]
            new_sql = new_candidates[index]["sql"] if index is not None else None
            llm_calls = int(first_llm["calls"]) + int(supplemental["llm_usage"]["calls"])
            jev_calls = int(first["jev"]["called"]) + int(bool(new_candidates))
            llm_usage = _sum_usage(first_llm, supplemental["llm_usage"])
            jev_usage = _sum_usage(first_jev, supplemental["jev_usage"])
        if llm_calls > 16 or jev_calls > 2:
            raise ValueError(f"Budget exceeded for {qid}: {llm_calls} LLM, {jev_calls} JEV")
        new = {"correct": correct(new_sql), "covered": int(any(correct(c["sql"])
               for c in new_candidates)), "direct_accepted": action == "stop",
               "llm_calls": llm_calls, "jev_calls": jev_calls,
               "llm_usage": llm_usage, "jev_usage": jev_usage}
        applied = dict(new)
        if review and review["recommended_index"] is not None:
            applied["correct"] = correct(first_candidates[review["recommended_index"]]["sql"])
        verifier_first = _score_verifier(model, baseline, first_candidates)
        verifier_baseline_final = _score_verifier(model, baseline, baseline_candidates)
        record = {"question_id": qid, "db_id": db_id, "action": action,
                  "first_pool_covered": int(any(correct(c["sql"]) for c in first_candidates)),
                  "first_selected_correct": correct(first_sql),
                  "baseline": old, "selective": new,
                  "adjudication_applied": applied,
                  "verifier_first_correct": correct(first_candidates[verifier_first]["sql"])
                    if verifier_first is not None else 0,
                  "verifier_baseline_final_correct": correct(
                      baseline_candidates[verifier_baseline_final]["sql"])
                    if verifier_baseline_final is not None else 0,
                  "adjudication_recommended_correct":
                    correct(first_candidates[review["recommended_index"]]["sql"])
                    if review and review["recommended_index"] is not None else None}
        evaluated.append(record)
    per_db = {db_id: {"baseline": _summary(items, "baseline"),
                       "selective": _summary(items, "selective"),
                       "adjudication_applied": _summary(items, "adjudication_applied")}
              for db_id, items in sorted(((db, [item for item in evaluated if item["db_id"] == db])
                                           for db in manifest["selected_databases"]), key=lambda pair: pair[0])}
    reviews = [item for item in evaluated if item["action"] == "adjudicate"]
    report = {"protocol": manifest["protocol"], "databases": manifest["selected_databases"],
              "questions": len(evaluated), "action_counts": dict(Counter(item["action"] for item in evaluated)),
              "first_pool_covered": sum(item["first_pool_covered"] for item in evaluated),
              "first_jev_selected_correct": sum(item["first_selected_correct"] for item in evaluated),
              "verifier_first_correct": sum(item["verifier_first_correct"] for item in evaluated),
              "verifier_baseline_final_correct": sum(item["verifier_baseline_final_correct"] for item in evaluated),
              "baseline": _summary(evaluated, "baseline"),
              "selective": _summary(evaluated, "selective"),
              "adjudication_applied": _summary(evaluated, "adjudication_applied"),
              "paired_improvements": sum(item["selective"]["correct"] > item["baseline"]["correct"]
                                         for item in evaluated),
              "paired_regressions": sum(item["selective"]["correct"] < item["baseline"]["correct"]
                                        for item in evaluated),
              "adjudication": {"questions": len(reviews),
                               "original_correct": sum(item["first_selected_correct"] for item in reviews),
                               "recommended_correct": sum(item["adjudication_recommended_correct"]
                                                          for item in reviews)},
              "per_database": per_db,
              "label_caveat": "Single-database execution agreement is only a proxy for semantic correctness"}
    for arm in ("baseline", "selective", "adjudication_applied"):
        summary = report[arm]
        summary["selected_accuracy_wilson95"] = _wilson_interval(
            summary["selected_correct"], summary["questions"])
        summary["direct_error_wilson95"] = _wilson_interval(
            summary["direct_wrong"], summary["direct_accepted"])
    (output / "per_question.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                                  for row in evaluated), encoding="utf-8")
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False,
                                            indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
