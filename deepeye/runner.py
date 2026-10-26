"""Windows/Linux launcher: python -m deepeye.runner --help.

Each upstream stage runs in its own process to preserve upstream config and resume
semantics. Files contain environment placeholders; credentials stay in memory.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "external" / "DeepEye-SQL"
STAGES = {
    "preprocess": "preprocess_dataset.py",
    "index-values": "create_vector_db_parallel.py",
    "retrieve-values": "run_value_retrieval.py",
    "index-few-shot": "build_few_shot_index.py",
    "prepare-few-shot": "run_few_shot_preparation.py",
    "link": "run_schema_linking.py",
    "generate": "run_sql_generation.py",
    "revise": "run_sql_revision.py",
    "select": "run_sql_selection.py",
}
KEYS = ("DEEPEYE_API_KEY", "DEEPEYE_EMBEDDING_API_KEY", "AIGC_API_KEY", "OPENAI_API_KEY",
        "DASHSCOPE_API_KEY", "EMBEDDING_API_KEY", "TYPESAFE_API_KEY")


def configure_environment():
    from dotenv import dotenv_values
    # Shell > .env.local > .env. Do not log the loaded mapping.
    for filename in (".env.local", ".env"):
        for key, value in dotenv_values(ROOT / filename).items():
            if value is not None and value != "":
                os.environ.setdefault(key, value)
    env = os.environ
    base_url = env.get("DEEPEYE_BASE_URL") or env.get("OPENAI_BASE_URL") or env.get("DASHSCOPE_BASE_URL") or \
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    school_api = base_url.rstrip("/") == "https://aigc-api.hkust-gz.edu.cn/api/v2"
    defaults = {
        "DEEPEYE_MODEL": env.get("LLM_MODEL") or env.get("PREPROCESS_MODEL") or "qwen3.5-35b-a3b",
        "DEEPEYE_BASE_URL": base_url,
        "DEEPEYE_API_KEY": (env.get("AIGC_API_KEY") if school_api else
                             env.get("OPENAI_API_KEY") or env.get("DASHSCOPE_API_KEY")) or "",
        "DEEPEYE_EMBEDDING_MODEL": env.get("EMBEDDING_MODEL") or "text-embedding-v4",
        "DEEPEYE_EMBEDDING_BASE_URL": env.get("EMBEDDING_BASE_URL") or
                                     env.get("DASHSCOPE_BASE_URL") or
                                     env.get("OPENAI_BASE_URL") or
                                     "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "DEEPEYE_EMBEDDING_API_KEY": env.get("EMBEDDING_API_KEY") or
                                    env.get("DASHSCOPE_API_KEY") or
                                    env.get("OPENAI_API_KEY") or "",
        "ANONYMIZED_TELEMETRY": "False", "PYTHONUTF8": "1",
    }
    for key, value in defaults.items():
        if not env.get(key):
            env[key] = value


def redact(text):
    for key in KEYS:
        secret = os.environ.get(key)
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return text


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def toml_dump(data):
    """Serialize the scalar/table/list subset used by upstream templates."""
    lines = []
    def emit(table, prefix):
        if prefix:
            lines.append("[" + ".".join(prefix) + "]")
        for key, value in table.items():
            if value is not None and not isinstance(value, dict):
                lines.append(f"{key} = {json.dumps(value, ensure_ascii=False)}")
        lines.append("")
        for key, value in table.items():
            if isinstance(value, dict):
                emit(value, prefix + [key])
    emit(data, [])
    text = "\n".join(lines)
    tomllib.loads(text)  # Validate before writing.
    return text


def run_dir(name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or name in {".", ".."}:
        raise ValueError("run-name must contain letters, digits, dots, underscores or hyphens")
    return ROOT / "results" / "deepeye" / name


def prepare(args):
    if not (UPSTREAM / "app" / "config" / "config.py").is_file():
        raise FileNotFoundError(f"Official source missing: {UPSTREAM}")
    data_root = Path(args.data_root).resolve()
    labels = data_root / args.split / f"{args.split}.json"
    examples = [dict(item, question_id=item.get("question_id", index))
                for index, item in enumerate(json.loads(labels.read_text(encoding="utf-8")))]
    ids = None
    if args.question_ids:
        ids = [int(value.strip()) for value in args.question_ids.split(",") if value.strip()]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("question-ids must be nonempty and unique")
    elif args.ids_file:
        source = json.loads(Path(args.ids_file).read_text(encoding="utf-8"))
        ids = source if isinstance(source, list) else source.get("evaluation_ids")
        if not isinstance(ids, list):
            raise ValueError("ids-file must be a list or contain evaluation_ids")
        ids = [int(value) for value in ids]
    if ids is not None:
        requested = set(ids)
        missing = requested - {item["question_id"] for item in examples}
        if missing:
            raise ValueError(f"Missing question IDs: {sorted(missing)}")
        examples = [item for item in examples if item["question_id"] in requested]
    limit = args.max_samples if args.max_samples is not None else (3 if args.profile == "smoke" else None)
    if limit is not None:
        if limit < 1:
            raise ValueError("max-samples must be positive")
        examples = examples[:limit]
    if not examples:
        raise ValueError("No dataset examples selected")
    for item in examples:
        db = data_root / args.split / f"{args.split}_databases" / item["db_id"] / f"{item['db_id']}.sqlite"
        if not db.is_file():
            raise FileNotFoundError(f"Database missing: {db}")
    few_shot = args.few_shot if args.few_shot is not None else args.profile == "full"
    if args.split == "train" and few_shot:
        raise ValueError("Train-target runs require --no-few-shot to avoid indexing target gold SQL")
    if few_shot and not (data_root / "train" / "train.json").is_file():
        raise FileNotFoundError("Dynamic few-shot retrieval needs data/bird/train/train.json")
    template = UPSTREAM / "config/template/qwen3.5-35B-a3b/config-bird-dev.toml"
    cfg = tomllib.loads(template.read_text(encoding="utf-8"))
    out = run_dir(args.run_name)
    cfg["run"].update(save_root=out.parent.as_posix(), exp_name=args.run_name,
                      shared_dir=(out / "shared").as_posix(), parallelism=args.workers,
                      embedding_batch_size=10, llm_timeout=60, checkpoint_interval=1)
    cfg["run"]["default_llm_profile"] = "local"
    cfg["llm_profiles"] = {"local": {
        "model": "${DEEPEYE_MODEL}", "base_url": "${DEEPEYE_BASE_URL}",
        "api_key": "${DEEPEYE_API_KEY}", "api_type": "openai", "temperature":0.7,
        "max_tokens":8192 if args.profile == "smoke" else 16384,
        "max_model_len":65536, "n_call_strategy":"split",
    }}
    if os.environ.get("DEEPEYE_MODEL", "").lower().startswith(("qwen3.5", "qwen3.6")):
        cfg["llm_profiles"]["local"]["extra_body"] = {"enable_thinking": False}
        if os.environ.get("DEEPEYE_BASE_URL", "").rstrip("/") == "https://aigc-api.hkust-gz.edu.cn/api/v2":
            cfg["llm_profiles"]["local"]["extra_body"]["chat_template_kwargs"] = {"enable_thinking": False}
    cfg["embedding"] = {
        "api_type":"openai", "embedding_model_name_or_path":"${DEEPEYE_EMBEDDING_MODEL}",
        "base_url":"${DEEPEYE_EMBEDDING_BASE_URL}", "api_key":"${DEEPEYE_EMBEDDING_API_KEY}",
    }
    cfg["dataset"].update(root_path=data_root.as_posix(), split=args.split,
                          question_ids=[item["question_id"] for item in examples],
                          sql_execution_timeout=30)
    cfg["few_shot_index"]["similarity_device"] = "auto"
    cfg["value_retrieval"]["local_index_device"] = "auto"
    for section in ("few_shot_index", "value_retrieval", "schema_linking", "sql_generation",
                    "sql_revision", "sql_selection"):
        cfg[section]["llm_profile"] = "local"
    cfg["few_shot_index"]["preliminary_sql"]["llm_profile"] = "local"
    if args.profile == "smoke":
        cfg["schema_linking"].update(direct_linking_sampling_budget=1, reversed_linking_sampling_budget=1)
        cfg["sql_generation"].update(dc_sampling_budget=1, skeleton_sampling_budget=1, icl_sampling_budget=1)
        cfg["sql_revision"]["checker_sampling_budget"] = 1
        cfg["sql_selection"]["evaluator_sampling_budget"] = 1
        cfg["few_shot_index"].update(max_samples=16, num_examples=3)
        cfg["few_shot_index"]["preliminary_sql"].update(dc_sampling_budget=1,skeleton_sampling_budget=1)
        cfg["llm_extractor"]["max_retry"] = 2
    if not few_shot:
        cfg["sql_generation"]["icl_sampling_budget"] = 0
        cfg["few_shot_index"]["preliminary_sql"]["enabled"] = False
    text = toml_dump(cfg)
    manifest = {
        "schema_version":1, "profile":args.profile, "run_name":args.run_name,
        "question_ids":[item["question_id"] for item in examples], "dataset_split":args.split,
        "dataset_root":data_root.as_posix(), "few_shot":few_shot,
        "model":os.environ.get("DEEPEYE_MODEL"),
        "base_url":os.environ.get("DEEPEYE_BASE_URL"),
        "embedding_model":os.environ.get("DEEPEYE_EMBEDDING_MODEL"),
        "embedding_base_url":os.environ.get("DEEPEYE_EMBEDDING_BASE_URL"),
        "input_sha256":fingerprint(examples), "config_sha256":hashlib.sha256(text.encode()).hexdigest(),
        "upstream_archive_sha256":"c8f78edce13fb69e13369961fe634a8d17e504a78f1c59da93f77188b44e7d17",
    }
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old != manifest:
            raise ValueError("Run already exists with different inputs/config; choose a new --run-name")
    else:
        write_json(manifest_path, manifest)
        (out / "config.toml").write_text(text, encoding="utf-8", newline="\n")
    print(f"Prepared {len(examples)} BIRD {args.split} questions: {out}")
    return out


def load_manifest(args, verify_inputs=False):
    out = run_dir(args.run_name)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    config = out / "config.toml"
    if hashlib.sha256(config.read_text(encoding="utf-8").encode()).hexdigest() != manifest["config_sha256"]:
        raise ValueError("Frozen config was modified; prepare a new run name")
    if verify_inputs:
        for field, env_key in (("model","DEEPEYE_MODEL"),("base_url","DEEPEYE_BASE_URL"),
                               ("embedding_model","DEEPEYE_EMBEDDING_MODEL"),
                               ("embedding_base_url","DEEPEYE_EMBEDDING_BASE_URL")):
            if manifest.get(field) != os.environ.get(env_key):
                raise ValueError(f"{env_key} changed since prepare; use a new run name")
        split = manifest["dataset_split"]
        labels = Path(manifest["dataset_root"])/split/f"{split}.json"
        ids = set(manifest["question_ids"])
        examples = [dict(item, question_id=item.get("question_id", index))
                    for index, item in enumerate(json.loads(labels.read_text(encoding="utf-8")))
                    if item.get("question_id", index) in ids]
        if fingerprint(examples) != manifest["input_sha256"]:
            raise ValueError("Dataset labels changed since prepare; use a new run name")
    os.environ["CONFIG_PATH"] = str(config)
    return out, manifest


def add_upstream():
    sys.path.insert(0, str(UPSTREAM))


def check_credentials():
    missing = [key for key in ("DEEPEYE_API_KEY", "DEEPEYE_EMBEDDING_API_KEY") if not os.environ.get(key)]
    if missing:
        raise ValueError("Set credentials in .env.local: " + ", ".join(missing))


def execute_stage(args, stage):
    out, manifest = load_manifest(args,verify_inputs=True)
    if stage in {"index-few-shot", "prepare-few-shot"} and not manifest["few_shot"]:
        raise ValueError("This run has few-shot disabled")
    if stage != "preprocess":
        check_credentials()
    log_path = out / "logs" / f"{stage}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    command = [sys.executable, "-u", "-m", "deepeye.runner", "_stage",
               "--run-name", args.run_name, "--stage", stage]
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(), stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in process.stdout:
            safe = redact(line)
            print(safe, end="", flush=True)
            log.write(safe)
        code = process.wait()
    with (out / "stages.jsonl").open("a", encoding="utf-8") as log:
        log.write(json.dumps({"stage":stage,"exit_code":code,"elapsed_seconds":time.perf_counter()-start,
                              "completed_at":datetime.now(timezone.utc).isoformat()}) + "\n")
    if code:
        raise RuntimeError(f"Stage {stage} failed (exit {code}); see {log_path}")


def internal_stage(args):
    out, _ = load_manifest(args,verify_inputs=True)
    os.environ["DEEPEYE_USAGE_PATH"] = str(out / "api_usage.jsonl")
    os.environ["DEEPEYE_STAGE"] = args.stage
    add_upstream()
    os.chdir(UPSTREAM)
    sys.argv = [str(UPSTREAM / "runner" / STAGES[args.stage])]
    runpy.run_path(sys.argv[0], run_name="__main__")
    # Some upstream runners report incomplete items without a failing exit code.
    if args.stage in {"retrieve-values", "link", "generate", "revise", "select"}:
        from app.config import get_config
        from app.dataset import load_dataset
        cfg = get_config()
        field = {"retrieve-values":"value_retrieval", "link":"schema_linking",
                 "generate":"sql_generation", "revise":"sql_revision", "select":"sql_selection"}[args.stage]
        dataset = load_dataset(getattr(cfg, field + "_config").save_path)
        failed = [item.question_id for item in dataset if not item.is_stage_complete(field)]
        if failed:
            raise RuntimeError(f"Incomplete {field} items: {failed}; rerun the stage to resume")


def export_candidates(out):
    from app.dataset import load_dataset
    from app.config import get_config
    cfg = get_config()
    dataset = load_dataset(cfg.sql_selection_config.save_path)
    predictions = {}
    with (out / "candidates.jsonl").open("w", encoding="utf-8") as file:
        for item in dataset:
            raw_candidates = item.sql_candidates_after_revision or item.sql_candidates or []
            candidates = list(dict.fromkeys(raw_candidates))
            predictions[str(item.question_id)] = item.final_selected_sql
            record = {"question_id":item.question_id,"db_id":item.database_id,"question":item.question,
                      "candidate_sqls":candidates,"selected_sql":item.final_selected_sql,
                      "candidate_sqls_with_votes":raw_candidates,
                      "schema_linking":item.final_linked_tables_and_columns,
                      "metrics":item.get_metrics_record().model_dump(),
                      "stage_metrics":{stage:{"seconds":getattr(item,stage+"_time"),
                                              "usage":getattr(item,stage+"_llm_cost")}
                                       for stage in ("value_retrieval","schema_linking","sql_generation",
                                                     "sql_revision","sql_selection")}}
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_json(out / "predictions.json", predictions)
    print(f"Exported {len(predictions)} questions: {out / 'candidates.jsonl'}")


def evaluate(out):
    from app.config import get_config
    from app.dataset import load_dataset
    from app.db_utils import execute_sql_without_cache
    cfg = get_config()
    dataset = load_dataset(cfg.sql_selection_config.save_path)
    rows = []
    for item in dataset:
        if not item.gold_sql:
            raise ValueError("No public gold SQL: export predictions instead")
        gold = execute_sql_without_cache(item.database_path,item.gold_sql,timeout=cfg.dataset_config.sql_execution_timeout)
        candidates = list(dict.fromkeys(item.sql_candidates_after_revision or item.sql_candidates or []))
        results = [execute_sql_without_cache(item.database_path,sql,timeout=cfg.dataset_config.sql_execution_timeout)
                   for sql in candidates]
        by_sql = dict(zip(candidates, results))
        before_candidates = list(dict.fromkeys(item.sql_candidates or []))
        before_results = [by_sql[sql] if sql in by_sql else
                          execute_sql_without_cache(item.database_path,sql,timeout=cfg.dataset_config.sql_execution_timeout)
                          for sql in before_candidates]
        selected = execute_sql_without_cache(item.database_path,item.final_selected_sql,timeout=cfg.dataset_config.sql_execution_timeout) if item.final_selected_sql else None
        valid_gold = gold.result_rows is not None
        def equal(result, multiset=False):
            if not valid_gold or result is None or result.result_rows is None:
                return False
            constructor = Counter if multiset else set
            return constructor(result.result_rows) == constructor(gold.result_rows)
        def self_consistency(multiset=False):
            # Preserve generation/revision multiplicity. Empty results remain valid.
            # Ties use first occurrence, never gold correctness or runtime noise.
            votes, representatives = Counter(), {}
            for sql in (item.sql_candidates_after_revision or item.sql_candidates or []):
                result = by_sql[sql]
                if result.result_rows is None:
                    continue
                signature = (frozenset(Counter(result.result_rows).items()) if multiset
                             else frozenset(result.result_rows))
                votes[signature] += 1
                representatives.setdefault(signature, sql)
            if not votes:
                return None, 0.0
            winner, count = votes.most_common(1)[0]
            return representatives[winner], count / votes.total()
        sc_set, sc_set_share = self_consistency()
        sc_multi, sc_multi_share = self_consistency(True)
        rows.append({"question_id":item.question_id,"db_id":item.database_id,
                     "difficulty":item.difficulty,
                     "gold_valid":valid_gold,"coverage_set":any(equal(r) for r in results),
                     "coverage_multiset":any(equal(r,True) for r in results),
                     "coverage_before_revision_multiset":any(equal(r,True) for r in before_results),
                     "before_revision_candidate_count":len(before_results),
                     "before_revision_executable_count":sum(r.result_rows is not None for r in before_results),
                     "correct_before_revision_examples":[sql for sql,r in zip(before_candidates,before_results) if equal(r,True)][:3],
                     "selected_ex_set":equal(selected),"selected_ex_multiset":equal(selected,True),
                     "sc_ex_set":equal(by_sql.get(sc_set)),
                     "sc_ex_multiset":equal(by_sql.get(sc_multi),True),
                     "sc_sql_set":sc_set,"sc_sql_multiset":sc_multi,
                     "sc_vote_share_set":sc_set_share,"sc_vote_share_multiset":sc_multi_share,
                     "selected_status":selected.result_type if selected else "missing",
                     "selected_error":selected.error_message if selected else None,
                     "gold_status":gold.result_type,"gold_row_count":len(gold.result_rows) if valid_gold else None,
                     "candidate_count":len(results),"executable_count":sum(r.result_rows is not None for r in results),
                     "candidates":[{"sql":sql,"status":r.result_type,"error":r.error_message,
                                    "row_count":len(r.result_rows) if r.result_rows is not None else None,
                                    "ex_set":equal(r),"ex_multiset":equal(r,True)}
                                   for sql,r in zip(candidates,results)]})
    valid = [r for r in rows if r["gold_valid"]]
    summary = {"questions":len(rows),"evaluable":len(valid),"invalid_gold":len(rows)-len(valid),
               "evaluation":"local order-insensitive rows; set matches upstream; multiset preserves duplicates; not official BIRD evaluator"}
    for field in ("coverage_set","coverage_multiset","coverage_before_revision_multiset","selected_ex_set","selected_ex_multiset",
                  "sc_ex_set","sc_ex_multiset"):
        summary[field] = sum(row[field] for row in valid)/len(valid) if valid else None
    count = sum(r["candidate_count"] for r in rows)
    summary["executable_rate"] = sum(r["executable_count"] for r in rows)/count if count else None
    write_json(out / "metrics.json", {"summary":summary,"per_question":rows})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def doctor(args):
    modules = ("pydantic","openai","chromadb","torch","pandas","tiktoken","dotenv")
    status = {}
    for name in modules:
        try:
            module = importlib.import_module(name)
            status[name] = getattr(module,"__version__","installed")
        except Exception as exc:
            status[name] = redact(f"ERROR: {type(exc).__name__}: {exc}")
    result = {"python":sys.version.split()[0],"source_exists":(UPSTREAM/"README.md").is_file(),
              "dependencies":status,"llm_credentials_configured":bool(os.environ.get("DEEPEYE_API_KEY")),
              "embedding_credentials_configured":bool(os.environ.get("DEEPEYE_EMBEDDING_API_KEY")),
              "bird_dev_exists":(Path(args.data_root)/"dev/dev.json").is_file()}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if any(str(value).startswith("ERROR") for value in status.values()):
        raise RuntimeError("Dependencies are incomplete; install requirements-deepeye.txt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["doctor","prepare","run","stage","inspect","export","evaluate","_stage"])
    parser.add_argument("--run-name",default="bird-smoke-qwen35")
    parser.add_argument("--profile",choices=["smoke","full"],default="smoke")
    parser.add_argument("--data-root",default=str(ROOT/"data/bird"))
    parser.add_argument("--split",choices=["dev","train","test"],default="dev")
    parser.add_argument("--max-samples",type=int)
    parser.add_argument("--workers",type=int,default=2)
    ids = parser.add_mutually_exclusive_group()
    ids.add_argument("--question-ids")
    ids.add_argument("--ids-file")
    parser.add_argument("--few-shot",action=argparse.BooleanOptionalAction,default=None)
    parser.add_argument("--stage",choices=list(STAGES))
    parser.add_argument("--dry-run",action="store_true")
    args = parser.parse_args()
    configure_environment()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.command == "doctor":
        return doctor(args)
    if args.command == "prepare":
        return prepare(args)
    if args.command == "_stage":
        if not args.stage:
            raise ValueError("--stage is required")
        return internal_stage(args)
    out, manifest = load_manifest(args,verify_inputs=args.command in {"run","stage"})
    if args.command == "run":
        stages = [name for name in STAGES if manifest["few_shot"] or "few-shot" not in name]
        print(f"Run {args.run_name}: {len(manifest['question_ids'])} questions; stages: {', '.join(stages)}")
        if args.dry_run:
            return
        check_credentials()
        for stage in stages:
            execute_stage(args,stage)
        add_upstream()
        export_candidates(out)
    elif args.command == "stage":
        if not args.stage:
            raise ValueError("--stage is required")
        execute_stage(args,args.stage)
    elif args.command == "inspect":
        print(json.dumps(manifest,ensure_ascii=False,indent=2))
        if (out/"stages.jsonl").exists():
            print((out/"stages.jsonl").read_text(encoding="utf-8"))
    else:
        add_upstream()
        if args.command == "export":
            export_candidates(out)
        elif args.command == "evaluate":
            evaluate(out)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(redact(f"{type(exc).__name__}: {exc}"),file=sys.stderr)
        sys.exit(1)
