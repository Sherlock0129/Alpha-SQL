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
from pathlib import Path
import pickle
import random
from typing import Any, Mapping, Protocol, Sequence

import sqlglot

from alphasql.algorithm.compiler.validation import validate_sql
from alphasql.algorithm.selection.dual_calibration import DualCalibrator, fit_dual
from alphasql.algorithm.selection.dual_confidence import JevPoolJudge, PoolJudgement
from alphasql.algorithm.selection.schema_signal import result_signature
from alphasql.database.database_manager import DatabaseManager
from alphasql.jev.client import JevClient
from alphasql.jev.models import Usage
from alphasql.jev.runtime import JevSettings
from alphasql.runner.task import Task


class CandidateGenerator(Protocol):
    def generate(self, task: Task, round_index: int) -> Sequence[str]: ...


def _path_sqls(paths: Any) -> list[str]:
    sqls = []
    for path in paths:
        if not path:
            continue
        node = path[-1]
        sql = (getattr(node, "final_sql_query", None)
               or getattr(node, "revised_sql_query", None)
               or getattr(node, "sql_query", None))
        if isinstance(sql, str) and sql.strip():
            sqls.append(sql.strip())
    return sqls


class FrozenPoolGenerator:
    """Reveal a fixed, previously generated candidate pool in bounded rounds."""

    def __init__(self, candidate_dir: str | Path, per_round: int = 8) -> None:
        if per_round < 1:
            raise ValueError("per_round must be positive")
        self.candidate_dir = Path(candidate_dir)
        self.per_round = per_round

    def generate(self, task: Task, round_index: int) -> Sequence[str]:
        path = self.candidate_dir / f"{task.question_id}.pkl"
        with path.open("rb") as handle:
            sqls = _path_sqls(pickle.load(handle))
        start = round_index * self.per_round
        return sqls[start:start + self.per_round]


class DirectLLMGenerator:
    """Reuse the repository's full-schema plus linked-schema SQL generator."""

    def __init__(self, db_root: str, work_dir: str | Path, model: str,
                 samples: int = 3, temperature: float = 0.8) -> None:
        if samples < 1:
            raise ValueError("samples must be positive")
        self.db_root, self.work_dir, self.model = db_root, Path(work_dir), model
        self.samples, self.temperature = samples, temperature

    def generate(self, task: Task, round_index: int) -> Sequence[str]:
        if not task.table_schema_dict:
            raise ValueError("Direct generation needs a preprocessed Task with table_schema_dict")
        from alphasql.runner.direct_candidate_runner import generate_one
        round_dir = self.work_dir / str(task.question_id) / f"round_{round_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        kwargs = {"model": self.model, "n": 1, "top_p": 0.95,
                  "max_tokens": 4096, "temperature": self.temperature,
                  "n_strategy": "multiple"}
        summary = generate_one((task.model_copy(update={"sql": None}), self.db_root,
                                str(round_dir), kwargs, self.samples))
        path = round_dir / f"{task.question_id}.pkl"
        if not path.exists():
            raise RuntimeError(f"SQL generation failed: {summary['warnings']}")
        with path.open("rb") as handle:
            return _path_sqls(pickle.load(handle))


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


@dataclass(frozen=True)
class SelectorSettings:
    max_rounds: int = 3
    max_candidates: int = 32
    coverage_stop: float = 0.8
    candidate_stop: float = 0.8
    execution_timeout: int = 60

    def __post_init__(self) -> None:
        if self.max_rounds < 1 or self.max_candidates < 1 or self.execution_timeout < 1:
            raise ValueError("Round, candidate and timeout limits must be positive")
        if not 0 <= self.coverage_stop <= 1 or not 0 <= self.candidate_stop <= 1:
            raise ValueError("Stop thresholds must be probabilities")


class AdaptiveSQLSelector:
    def __init__(self, generator: CandidateGenerator, judge: JevPoolJudge,
                 settings: SelectorSettings | None = None,
                 calibrator: DualCalibrator | None = None) -> None:
        self.generator, self.judge = generator, judge
        self.settings = settings or SelectorSettings()
        self.calibrator = calibrator
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
        usage_total = {"input_tokens": 0, "output_tokens": 0}
        generation_task = task.model_copy(update={"sql": None})
        for round_index in range(self.settings.max_rounds):
            generated = self.generator.generate(generation_task, round_index)
            new_count = 0
            for sql in generated:
                canonical = _canonical(sql)
                if not canonical or canonical in seen:
                    continue
                if len(candidates) >= self.settings.max_candidates:
                    break
                seen.add(canonical)
                execution = validate_sql(str(db_path), sql, self.settings.execution_timeout)
                candidates.append({"sql": sql, "canonical_sql": canonical,
                                   "source_round": round_index,
                                   "execution": execution.summary()})
                new_count += 1
            if candidates:
                last = self.judge.assess(task.question, task.evidence, schema, candidates)
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
            rounds.append({"round": round_index, "new_candidates": new_count,
                           "pool_size": len(candidates), "coverage": coverage,
                           "best_candidate_probability": best})
            if (coverage >= self.settings.coverage_stop
                    and best >= self.settings.candidate_stop):
                break
            if len(candidates) >= self.settings.max_candidates:
                break
        if last is None:
            last = PoolJudgement(0.0, (), Usage())
        valid = [i for i, item in enumerate(candidates) if item["execution"]["valid"]]
        for i, item in enumerate(candidates):
            raw = last.candidate_raw[i]
            item["candidate_raw"] = raw
            item["candidate_probability"] = (
                self.calibrator.candidate.predict(raw) if self.calibrator and i in valid
                else raw if i in valid else 0.0)
        selected = max(valid, key=lambda i: (candidates[i]["candidate_probability"], -i)) if valid else None
        return {
            "schema_version": 1, "question_id": task.question_id, "db_id": task.db_id,
            "selected_index": selected,
            "selected_sql": candidates[selected]["sql"] if selected is not None else None,
            "coverage_raw": last.coverage_raw,
            "coverage_probability": (self.calibrator.coverage.predict(last.coverage_raw)
                                     if self.calibrator and valid else last.coverage_raw),
            "confidence_kind": "calibrated" if self.calibrator else "raw_jev_judgement",
            "uncertain": (selected is None or rounds[-1]["coverage"] < self.settings.coverage_stop
                          or rounds[-1]["best_candidate_probability"] < self.settings.candidate_stop),
            "candidates": candidates, "rounds": rounds,
            "jev_usage_total": usage_total,
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


def _run(args: argparse.Namespace) -> None:
    settings = JevSettings.from_environment()
    settings.require_api_key()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with Path(args.tasks_file_path).open("rb") as handle:
        tasks = pickle.load(handle)
    if args.question_id is not None:
        tasks = [task for task in tasks if str(task.question_id) == args.question_id]
    if args.limit is not None:
        tasks = tasks[:args.limit]
    generator: CandidateGenerator
    if args.generator == "pool":
        if not args.candidate_dir:
            raise ValueError("--candidate-dir is required for pool generation")
        generator = FrozenPoolGenerator(args.candidate_dir, args.per_round)
    else:
        generator = DirectLLMGenerator(args.database_root_dir, output / "generated",
                                       args.model, args.samples, args.temperature)
    judge = JevPoolJudge(JevClient(settings), args.max_candidates)
    calibrator = DualCalibrator.load(args.calibrator) if args.calibrator else None
    pipeline = AdaptiveSQLSelector(generator, judge,
        SelectorSettings(args.max_rounds, args.max_candidates,
                         args.coverage_stop, args.candidate_stop, args.execution_timeout),
        calibrator)
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
    run.add_argument("--model", default="qwen3-coder-flash")
    run.add_argument("--samples", type=int, default=3)
    run.add_argument("--temperature", type=float, default=0.8)
    run.add_argument("--per-round", type=int, default=8)
    run.add_argument("--max-rounds", type=int, default=3)
    run.add_argument("--max-candidates", type=int, default=32)
    run.add_argument("--coverage-stop", type=float, default=0.8)
    run.add_argument("--candidate-stop", type=float, default=0.8)
    run.add_argument("--execution-timeout", type=int, default=60)
    run.add_argument("--calibrator")
    run.add_argument("--question-id")
    run.add_argument("--limit", type=int)
    run.add_argument("--overwrite", action="store_true")
    run.set_defaults(func=_run)
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
