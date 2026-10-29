"""Export covered first-round pools whose raw JEV Top-1 is execution-labelled wrong.

Reference SQL is not sent to any model. The label is same-database execution
equivalence, so manual semantic review is still needed for difficult cases.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alphasql.algorithm.compiler.validation import validate_sql
from alphasql.algorithm.selection.execution_evidence import pool_identity
from alphasql.runner.selection_v2 import _read, _save


def audit(split_dir: Path, execution_dir: Path, gold_path: Path | None = None,
          database_root_dir: Path | None = None) -> tuple[dict, list[dict]]:
    cases = []
    totals = {}
    gold_rows = json.loads(gold_path.read_text(encoding="utf-8")) if gold_path else None
    for split in ("train", "validation", "test"):
        rows = _read(split_dir / f"{split}.jsonl")
        covered = wrong = 0
        for row in rows:
            candidates = row["rounds"][0]["candidates"]
            valid = [i for i, item in enumerate(candidates)
                     if item.get("execution_rechecked_valid", item["execution"]["valid"])]
            ranked = sorted(valid, key=lambda i: (-float(candidates[i]["candidate_raw"]), i))
            positive = [i for i in ranked if candidates[i]["execution_label"] == 1]
            if not positive:
                continue
            covered += 1
            top = ranked[0]
            if candidates[top]["execution_label"] == 1:
                continue
            wrong += 1
            stem = f'{row["db_id"]}_{row["question_id"]}.json'
            execution = json.loads((execution_dir / stem).read_text(encoding="utf-8"))
            if execution["pool_sha256"] != pool_identity(row):
                raise ValueError(f"Pool fingerprint mismatch: {stem}")
            best_correct = positive[0]
            case = {
                "split": split, "db_id": row["db_id"], "question_id": row["question_id"],
                "question": row["question"], "evidence": row.get("evidence") or "",
                "pool_sha256": pool_identity(row), "valid_candidates": len(ranked),
                "top_index": top, "top_raw": float(candidates[top]["candidate_raw"]),
                "best_correct_index": best_correct,
                "best_correct_rank": ranked.index(best_correct) + 1,
                "best_correct_raw": float(candidates[best_correct]["candidate_raw"]),
                "raw_gap": float(candidates[top]["candidate_raw"] - candidates[best_correct]["candidate_raw"]),
                "all_correct_ranks": [ranked.index(i) + 1 for i in positive],
                "ranked_candidates": [{
                    "rank": rank, "index": i, "raw": float(candidates[i]["candidate_raw"]),
                    "execution_label": int(candidates[i]["execution_label"]),
                    "label_source": candidates[i].get("label_source"),
                    "sql": candidates[i]["sql"], "route": candidates[i].get("route"),
                    "result_hash": execution["candidates"][i].get("result_hash"),
                    "row_count": execution["candidates"][i].get("row_count"),
                    "column_count": execution["candidates"][i].get("column_count"),
                    "original_execution": candidates[i]["execution"],
                    "near_miss": candidates[i].get("near_miss"),
                    "near_miss_ast_edits": candidates[i].get("near_miss_ast_edits"),
                } for rank, i in enumerate(ranked, start=1)],
            }
            if gold_rows is not None:
                reference = gold_rows[int(row["question_id"])]
                if reference["db_id"] != row["db_id"]:
                    raise ValueError(f"Gold question/database mismatch: {stem}")
                case["gold_sql"] = reference["SQL"]
                if database_root_dir is not None:
                    db = database_root_dir / row["db_id"] / f'{row["db_id"]}.sqlite'
                    gold_result = validate_sql(str(db), reference["SQL"], 60)
                    case["gold_execution"] = gold_result.summary()
            cases.append(case)
        totals[split] = {"questions": len(rows), "covered": covered,
                         "covered_top1_wrong": wrong}
    return {"scope": "selection_v2_300", "totals": totals,
            "covered_top1_wrong": len(cases),
            "best_correct_rank_histogram": {str(k): sum(c["best_correct_rank"] == k for c in cases)
                                            for k in range(2, 6)}}, cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--execution-dir", required=True)
    parser.add_argument("--gold-path")
    parser.add_argument("--database-root-dir")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    summary, cases = audit(Path(args.split_dir), Path(args.execution_dir),
                           Path(args.gold_path) if args.gold_path else None,
                           Path(args.database_root_dir) if args.database_root_dir else None)
    _save(output / "summary.json", summary)
    output.mkdir(parents=True, exist_ok=True)
    (output / "cases.jsonl").write_text("".join(
        json.dumps(case, ensure_ascii=False) + "\n" for case in cases), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
