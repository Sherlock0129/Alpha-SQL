"""Run the frozen fresh-database baseline with bounded, logged API calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import socket

from alphasql.algorithm.selection.dual_confidence import JevPoolJudge
from alphasql.jev.client import JevClient
from alphasql.jev.runtime import JevSettings
from alphasql.llm_call.runtime import configure_environment
from alphasql.runner.new_selector import (
    AdaptiveSQLSelector, DirectLLMGenerator, SelectorSettings, _path_candidates,
)


def _usage(path: Path) -> dict[str, int | bool]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    return {"calls": len(rows), "input_tokens": sum(int(row.get("prompt_tokens") or 0) for row in rows),
            "output_tokens": sum(int(row.get("completion_tokens") or 0) for row in rows),
            "tokens_complete": all(row.get("prompt_tokens") is not None and
                                   row.get("completion_tokens") is not None for row in rows)}


class LoggedGenerator(DirectLLMGenerator):
    """Keep first- and second-round usage separate for paired replay."""

    def generate(self, task, round_index, feedback=None):
        base = os.environ.get("ALPHASQL_USAGE_PATH")
        if not base:
            raise ValueError("Per-question usage path is required")
        path = Path(base)
        round_path = path.with_name(f"{path.stem}_round{round_index}{path.suffix}")
        os.environ["ALPHASQL_USAGE_PATH"] = str(round_path)
        try:
            return super().generate(task, round_index, feedback)
        finally:
            os.environ["ALPHASQL_USAGE_PATH"] = base


class CachedLoggedGenerator(LoggedGenerator):
    """Replay already generated SQL paths after a JEV transport failure."""

    def generate(self, task, round_index, feedback=None):
        path = (self.work_dir / str(task.question_id) /
                f"round_{round_index:02d}" / f"{task.question_id}.pkl")
        if path.is_file():
            feedback_path = path.with_name("feedback.json")
            if round_index and feedback_path.is_file():
                saved = json.loads(feedback_path.read_text(encoding="utf-8"))
                if feedback is None or saved != feedback.to_dict():
                    raise ValueError("Cached SQL paths were generated from different feedback")
            with path.open("rb") as handle:
                return _path_candidates(pickle.load(handle))
        return super().generate(task, round_index, feedback)


def _question_usage(base: Path) -> dict[str, object]:
    rounds = [_usage(base.with_name(f"{base.stem}_round{i}{base.suffix}"))
              for i in range(2)]
    return {"calls": sum(int(item["calls"]) for item in rounds),
            "input_tokens": sum(int(item["input_tokens"]) for item in rounds),
            "output_tokens": sum(int(item["output_tokens"]) for item in rounds),
            "tokens_complete": all(bool(item["tokens_complete"]) for item in rounds),
            "rounds": rounds}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-dir", required=True)
    parser.add_argument("--database-root-dir", default="data/bird/train/train_databases")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume-cached", action="store_true",
                        help="Replay saved generated paths rather than sampling them again")
    parser.add_argument("--max-rounds", type=int, choices=(1, 2), default=2,
                        help="Use one round for a frozen-pool selection study")
    args = parser.parse_args()
    configure_environment()
    protocol = Path(args.protocol_dir)
    manifest = json.loads((protocol / "manifest.json").read_text(encoding="utf-8"))
    task_path = protocol / "tasks.pkl"
    if hashlib.sha256(task_path.read_bytes()).hexdigest() != manifest["tasks_sha256"]:
        raise ValueError("Frozen task file changed")
    with task_path.open("rb") as handle:
        tasks = pickle.load(handle)
    if [task.question_id for task in tasks] != manifest["question_ids"]:
        raise ValueError("Frozen question IDs changed")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Invalid shard index or count")
    tasks = [task for index, task in enumerate(tasks)
             if index % args.shard_count == args.shard_index]
    if args.limit is not None:
        tasks = tasks[:args.limit]
    for host in ("aigc-api.hkust-gz.edu.cn", "api.typesafe.ai"):
        try:
            socket.getaddrinfo(host, 443)
        except OSError as exc:
            raise RuntimeError(f"Cannot resolve {host}; restore the authorized VPN before running") from exc
    settings = JevSettings.from_environment()
    settings.require_api_key()
    if not os.environ.get("AIGC_API_KEY"):
        raise ValueError("AIGC_API_KEY is required")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    budget = manifest["budget_per_question"]
    generator_class = CachedLoggedGenerator if args.resume_cached else LoggedGenerator
    generator = generator_class(args.database_root_dir, output / "generated",
                                   "Qwen3.8-27B", 2, .8,
                                   ("skeleton", "query_plan"), 1,
                                   "https://aigc-api.hkust-gz.edu.cn/api/v2",
                                   "AIGC_API_KEY",
                                   {"chat_template_kwargs": {"enable_thinking": False}},
                                   True)
    selector = AdaptiveSQLSelector(generator, JevPoolJudge(JevClient(settings), 16),
        SelectorSettings(max_rounds=args.max_rounds, max_candidates=16,
                         coverage_stop=.8, candidate_stop=.8, execution_timeout=60),
        generator_version="hkustgz-qwen38-selector-v3-direct")
    old_usage_path, old_limit = os.environ.get("ALPHASQL_USAGE_PATH"), os.environ.get("LLM_MAX_REQUESTS")
    failures, completed = [], 0
    try:
        os.environ["LLM_MAX_REQUESTS"] = str(budget["max_llm_requests_per_round"])
        for task in tasks:
            result_path = output / f"{task.question_id}.json"
            if result_path.exists():
                completed += 1
                continue
            usage_path = output / "usage" / f"{task.question_id}.jsonl"
            usage_path.parent.mkdir(parents=True, exist_ok=True)
            if not args.resume_cached:
                for round_index in range(2):
                    usage_path.with_name(f"{usage_path.stem}_round{round_index}{usage_path.suffix}").unlink(missing_ok=True)
            os.environ["ALPHASQL_USAGE_PATH"] = str(usage_path.resolve())
            try:
                result = selector.run(task, args.database_root_dir)
                result["llm_usage"] = _question_usage(usage_path)
                if result["llm_usage"]["calls"] > 16 or len(result["rounds"]) > 2:
                    raise ValueError("Call or round budget exceeded")
                if sum(int(round_["jev"]["called"]) for round_ in result["rounds"]) > 2:
                    raise ValueError("JEV request budget exceeded")
                temporary = result_path.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(result, ensure_ascii=False,
                                                indent=2, allow_nan=False, default=str) + "\n",
                                     encoding="utf-8")
                temporary.replace(result_path)
                completed += 1
                print(json.dumps({"question_id": task.question_id, "completed": completed,
                                  "llm_calls": result["llm_usage"]["calls"],
                                  "jev_calls": sum(int(r["jev"]["called"]) for r in result["rounds"])},
                                 ensure_ascii=False), flush=True)
            except Exception as exc:
                failures.append({"question_id": task.question_id,
                                 "error_type": type(exc).__name__, "error": str(exc),
                                 "llm_usage": _question_usage(usage_path)})
                print(json.dumps(failures[-1], ensure_ascii=False), flush=True)
    finally:
        if old_usage_path is None:
            os.environ.pop("ALPHASQL_USAGE_PATH", None)
        else:
            os.environ["ALPHASQL_USAGE_PATH"] = old_usage_path
        if old_limit is None:
            os.environ.pop("LLM_MAX_REQUESTS", None)
        else:
            os.environ["LLM_MAX_REQUESTS"] = old_limit
    failure_name = ("failures.json" if args.shard_count == 1 else
                    f"failures_shard{args.shard_index}.json")
    (output / failure_name).write_text(json.dumps(failures, ensure_ascii=False,
                                                 indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"completed": completed, "requested": len(tasks),
                      "failures": len(failures)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
