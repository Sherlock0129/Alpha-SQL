"""Collect only the new actions needed to replay a policy from shared first pools."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import socket

from alphasql.algorithm.compiler.validation import validate_sql
from alphasql.algorithm.selection.dual_confidence import JevPoolJudge, PoolJudgement
from alphasql.algorithm.selection.generation_feedback import choose_feedback
from alphasql.algorithm.selection.selective_stop import StopPolicy
from alphasql.jev.client import JevClient
from alphasql.jev.models import Usage
from alphasql.jev.runtime import JevSettings
from alphasql.llm_call.runtime import configure_environment
from alphasql.runner.fresh_selector_run import CachedLoggedGenerator, _question_usage
from alphasql.runner.new_selector import _canonical


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def decision(first: dict, policy: StopPolicy) -> str:
    valid = [(i, item) for i, item in enumerate(first["candidates"])
             if item["execution"]["valid"]]
    if not valid:
        return "continue"
    score = max(float(item[policy.score_key]) for _, item in valid)
    pool = float(first["jev"]["pool_covered"])
    if policy.accepts(score, pool):
        return "stop"
    if policy.pool_review_below is not None and pool < policy.pool_review_below:
        return "adjudicate"
    return "continue"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-dir", required=True)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--database-root-dir", default="data/bird/train/train_databases")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    configure_environment()
    policy = StopPolicy.load(args.policy)
    if policy.score_key != "candidate_raw":
        raise ValueError("This paired branch collector uses the frozen JEV raw policy")
    protocol = Path(args.protocol_dir)
    manifest = json.loads((protocol / "manifest.json").read_text(encoding="utf-8"))
    with (protocol / "tasks.pkl").open("rb") as handle:
        tasks = pickle.load(handle)
    if [task.question_id for task in tasks] != manifest["question_ids"]:
        raise ValueError("Frozen question IDs changed")
    baseline_dir = Path(args.baseline_dir)
    missing = [task.question_id for task in tasks
               if not (baseline_dir / f"{task.question_id}.json").is_file()]
    if missing:
        raise ValueError(f"Baseline must complete before branching; missing {len(missing)} results")
    if args.limit is not None:
        tasks = tasks[:args.limit]
    for host in ("aigc-api.hkust-gz.edu.cn", "api.typesafe.ai"):
        try:
            socket.getaddrinfo(host, 443)
        except OSError as exc:
            raise RuntimeError(f"Cannot resolve {host}; restore the authorized VPN before running") from exc
    settings = JevSettings.from_environment()
    settings.require_api_key()
    judge = JevPoolJudge(JevClient(settings), 16)
    output = Path(args.output_dir)
    generator = CachedLoggedGenerator(args.database_root_dir, output / "generated",
                                "Qwen3.8-27B", 2, .8,
                                ("skeleton", "query_plan"), 1,
                                "https://aigc-api.hkust-gz.edu.cn/api/v2",
                                "AIGC_API_KEY",
                                {"chat_template_kwargs": {"enable_thinking": False}}, True)
    old_usage, old_limit = os.environ.get("ALPHASQL_USAGE_PATH"), os.environ.get("LLM_MAX_REQUESTS")
    failures, counts = [], {"stop": 0, "adjudicate": 0, "continue_shared": 0,
                            "continue_supplement": 0}
    try:
        os.environ["LLM_MAX_REQUESTS"] = str(manifest["budget_per_question"]["max_llm_requests_per_round"])
        for task in tasks:
            baseline = json.loads((baseline_dir / f"{task.question_id}.json").read_text(encoding="utf-8"))
            first = baseline["rounds"][0]
            if policy.generator_version and baseline["generator"].get("version") != policy.generator_version:
                raise ValueError("Baseline generator version differs from frozen stop policy")
            if (policy.judge_model and first["jev"].get("called")
                    and first["jev"].get("model") != policy.judge_model):
                raise ValueError("Baseline JEV model differs from frozen stop policy")
            action = decision(first, policy)
            if action == "stop":
                counts["stop"] += 1
                continue
            if action == "continue" and len(baseline["rounds"]) > 1:
                counts["continue_shared"] += 1
                continue
            if action == "continue" and first["pool_size"] >= 16:
                counts["continue_shared"] += 1
                continue
            if action == "adjudicate":
                path = output / "review" / f"{task.question_id}.json"
                if path.exists():
                    counts["adjudicate"] += 1
                    continue
                try:
                    response = judge.adjudicate(task.question, task.evidence,
                                                baseline["full_schema"],
                                                first["candidates"],
                                                [float(item["candidate_raw"])
                                                 for item in first["candidates"]])
                    usage = response.pop("usage")
                    if policy.judge_model and response["model"] != policy.judge_model:
                        raise ValueError("Actual JEV review model changed")
                    response["usage"] = {"input_tokens": usage.input_tokens,
                                         "output_tokens": usage.output_tokens}
                    response.update(question_id=task.question_id, db_id=task.db_id,
                                    initial_action="adjudicate")
                    _write(path, response)
                    counts["adjudicate"] += 1
                except Exception as exc:
                    failures.append({"question_id": task.question_id, "action": action,
                                     "error_type": type(exc).__name__, "error": str(exc)})
                continue
            path = output / "supplement" / f"{task.question_id}.json"
            if path.exists():
                counts["continue_supplement"] += 1
                continue
            usage_path = output / "usage" / f"{task.question_id}.jsonl"
            usage_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path = (generator.work_dir / str(task.question_id) /
                          "round_01" / f"{task.question_id}.pkl")
            if not cache_path.is_file():
                usage_path.with_name(f"{usage_path.stem}_round1{usage_path.suffix}").unlink(missing_ok=True)
            os.environ["ALPHASQL_USAGE_PATH"] = str(usage_path.resolve())
            try:
                jev = first["jev"]
                judgement = PoolJudgement(float(jev["pool_covered"]),
                    tuple(float(item["candidate_raw"]) for item in first["candidates"]),
                    Usage(), {name: float(jev.get(f"gap_{name}") or 0.0)
                              for name in ("schema", "projection", "filters", "logic")},
                    jev.get("model") or "")
                feedback = choose_feedback(judgement, first["candidates"],
                                           candidate_scores=judgement.candidate_raw)
                generated = generator.generate(task, 1, feedback)
                combined = [dict(item) for item in first["candidates"]]
                seen = {item.get("canonical_sql") or _canonical(item["sql"]) for item in combined}
                db_path = Path(args.database_root_dir) / task.db_id / f"{task.db_id}.sqlite"
                new_indices = []
                for item in generated:
                    canonical = _canonical(item.sql)
                    if not canonical or canonical in seen or len(combined) >= 16:
                        continue
                    seen.add(canonical)
                    execution = validate_sql(str(db_path), item.sql, 60)
                    new_indices.append(len(combined))
                    combined.append({"sql": item.sql, "canonical_sql": canonical,
                                     "route": item.route, "source_round": 1,
                                     "selected_tables": list(item.selected_tables),
                                     "selected_columns": list(item.selected_columns),
                                     "execution": execution.summary()})
                second = judge.assess(task.question, task.evidence,
                                      baseline["full_schema"], combined)
                if policy.judge_model and second.model != policy.judge_model:
                    raise ValueError("Actual JEV response model changed")
                for i, item in enumerate(combined):
                    item["candidate_raw"] = second.candidate_raw[i]
                valid = [i for i, item in enumerate(combined) if item["execution"]["valid"]]
                selected = max(valid, key=lambda i: (second.candidate_raw[i], -i)) if valid else None
                llm_usage = _question_usage(usage_path)["rounds"][1]
                if int(llm_usage["calls"]) > 8:
                    raise ValueError("Supplement exceeded LLM request budget")
                _write(path, {"question_id": task.question_id, "db_id": task.db_id,
                              "feedback": feedback.to_dict(), "new_candidate_indices": new_indices,
                              "candidates": combined, "selected_index": selected,
                              "pool_covered": second.coverage_raw,
                              "jev_model": second.model,
                              "jev_usage": {"input_tokens": second.usage.input_tokens,
                                            "output_tokens": second.usage.output_tokens},
                              "llm_usage": llm_usage})
                counts["continue_supplement"] += 1
            except Exception as exc:
                failures.append({"question_id": task.question_id, "action": action,
                                 "error_type": type(exc).__name__, "error": str(exc)})
    finally:
        if old_usage is None:
            os.environ.pop("ALPHASQL_USAGE_PATH", None)
        else:
            os.environ["ALPHASQL_USAGE_PATH"] = old_usage
        if old_limit is None:
            os.environ.pop("LLM_MAX_REQUESTS", None)
        else:
            os.environ["LLM_MAX_REQUESTS"] = old_limit
    _write(output / "failures.json", {"failures": failures, "counts": counts})
    print(json.dumps({"counts": counts, "failures": len(failures)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
