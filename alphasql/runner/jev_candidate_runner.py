"""Independent Jev schema-link -> constrained beam -> SQLite -> rerank runner."""

from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import pickle
import re
import sqlite3
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

from tqdm import tqdm

from alphasql.algorithm.compiler.beam_search import QueryPlanBeamSearch
from alphasql.algorithm.compiler.query_plan import SchemaGraph
from alphasql.algorithm.compiler.sql_compiler import SQLCompilationError, SQLCompiler
from alphasql.algorithm.compiler.validation import CandidateExecution, validate_sql
from alphasql.algorithm.schema_linking.jev_linker import JevSchemaLinker, SchemaLinkingResult
from alphasql.algorithm.selection.jev_reranker import (
    JevSQLReranker,
    RerankCandidate,
    RerankResult,
    RerankWeights,
)
from alphasql.database.database_manager import DatabaseManager
from alphasql.database.schema import DatabaseSchema
from alphasql.jev.client import JevClient
from alphasql.jev.runtime import JevSettings
from alphasql.runner.task import Task


_VALUE_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_&.'-]*")
_STOP_WORDS = {
    "a", "all", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "list", "of", "on", "or", "the", "their", "there",
    "to", "what", "which", "who", "with",
}


def retrieve_exact_question_values(
    db_path: str | Path,
    schema: DatabaseSchema,
    question: str,
    evidence: str = "",
    max_phrases: int = 64,
    max_values: int = 32,
) -> Dict[str, Dict[str, List[Any]]]:
    """Retrieve exact DB values mentioned in the question without model generation."""

    tokens = _VALUE_TOKEN.findall(f"{question} {evidence}".strip())
    phrases: List[str] = []
    seen = set()
    quoted = re.findall(r"['\"]([^'\"]+)['\"]", f"{question} {evidence}")
    for phrase in quoted + [
        " ".join(tokens[start:start + size])
        for size in range(min(5, len(tokens)), 0, -1)
        for start in range(0, len(tokens) - size + 1)
    ]:
        normalized = phrase.strip().strip(".,?!;:")
        key = normalized.lower()
        if not normalized or key in seen or (len(normalized.split()) == 1 and key in _STOP_WORDS):
            continue
        seen.add(key)
        phrases.append(normalized)
        if len(phrases) >= max_phrases:
            break
    if not phrases:
        return {}
    placeholders = ",".join("?" for _ in phrases)
    parameters = [phrase.lower() for phrase in phrases]
    found: Dict[str, Dict[str, List[Any]]] = {}
    db_uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(db_uri, uri=True)) as connection:
        for table_key, table in schema.tables.items():
            table_name = table.table_name or table_key
            for column_key, column in table.columns.items():
                if column.column_type.upper() == "BLOB":
                    continue
                column_name = column.original_column_name or column_key
                quoted_table = '"' + table_name.replace('"', '""') + '"'
                quoted_column = '"' + column_name.replace('"', '""') + '"'
                query = (
                    f"SELECT DISTINCT {quoted_column} FROM {quoted_table} "
                    f"WHERE lower(CAST({quoted_column} AS TEXT)) IN ({placeholders}) LIMIT 8"
                )
                try:
                    values = [row[0] for row in connection.execute(query, parameters).fetchall()]
                except sqlite3.Error:
                    continue
                if values:
                    found.setdefault(table_name, {})[column_name] = values
                    if sum(len(items) for columns in found.values() for items in columns.values()) >= max_values:
                        return found
    return found


@dataclass
class JevSQLCandidate:
    canonical_sql: str
    plan: Dict[str, Any]
    beam_probability: float
    decision_probabilities: Dict[str, float]
    execution: CandidateExecution
    beam_uncertain: bool = False
    rerank: Optional[RerankResult] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "canonical_sql": self.canonical_sql,
            "plan": self.plan,
            "beam_probability": self.beam_probability,
            "decision_probabilities": self.decision_probabilities,
            "execution": self.execution.summary(),
            "beam_uncertain": self.beam_uncertain,
            "rerank": self.rerank.to_dict() if self.rerank else None,
        }


@dataclass
class JevRunResult:
    question_id: Any
    db_id: str
    schema_linking: SchemaLinkingResult
    candidates: List[JevSQLCandidate]
    selected_index: Optional[int]
    uncertain: bool
    timings_ms: Dict[str, float]

    @property
    def selected_sql(self) -> Optional[str]:
        if self.selected_index is None:
            return None
        return self.candidates[self.selected_index].canonical_sql

    def to_selection_payload(self) -> Dict[str, Any]:
        """JSON adapter for downstream candidate/selection evaluation."""

        return {
            "question_id": self.question_id,
            "db_id": self.db_id,
            "selected_sql": self.selected_sql,
            "selected_index": self.selected_index,
            "uncertain": self.uncertain,
            "timings_ms": self.timings_ms,
            "candidate_sqls": [candidate.canonical_sql for candidate in self.candidates],
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "schema_linking": self.schema_linking.to_dict(),
        }


class JevCandidateRunner:
    def __init__(
        self,
        client: JevClient | None = None,
        settings: JevSettings | None = None,
        rerank_weights: RerankWeights | None = None,
        execution_timeout: int = 60,
        max_literal_questions: int = 64,
    ) -> None:
        self.settings = settings or JevSettings.from_environment()
        self.client = client or JevClient(self.settings)
        self.linker = JevSchemaLinker(
            self.client, self.settings, max_literal_questions=max_literal_questions
        )
        self.reranker = JevSQLReranker(self.client, self.settings, rerank_weights)
        self.execution_timeout = execution_timeout

    def run(
        self,
        question: str,
        evidence: str,
        schema: DatabaseSchema,
        db_path: str | Path,
        question_id: Any = None,
        value_examples: Optional[Mapping[str, Mapping[str, Sequence[Any]]]] = None,
    ) -> JevRunResult:
        started = time.perf_counter()
        retrieved = retrieve_exact_question_values(db_path, schema, question, evidence)
        if value_examples:
            for table, columns in value_examples.items():
                for column, values in columns.items():
                    target = retrieved.setdefault(table, {}).setdefault(column, [])
                    target.extend(value for value in values if value not in target)
        linked = self.linker.link(question, evidence, schema, retrieved)
        linked_at = time.perf_counter()
        graph = SchemaGraph(schema)
        beam = QueryPlanBeamSearch(
            graph, self.settings.beam_width, self.settings.min_confidence
        ).generate(linked)
        compiler = SQLCompiler(graph)
        candidates: List[JevSQLCandidate] = []
        seen = set()
        for plan_candidate in beam:
            try:
                sql = compiler.compile(plan_candidate.plan)
            except SQLCompilationError:
                continue
            if sql in seen:
                continue
            seen.add(sql)
            execution = validate_sql(str(db_path), sql, self.execution_timeout)
            candidates.append(JevSQLCandidate(
                canonical_sql=sql,
                plan=plan_candidate.plan.to_dict(),
                beam_probability=plan_candidate.probability,
                decision_probabilities=plan_candidate.decision_probabilities,
                execution=execution,
                beam_uncertain=plan_candidate.uncertain,
            ))
        executed_at = time.perf_counter()
        valid_indices = [index for index, item in enumerate(candidates) if item.execution.valid]
        selected_index = None
        uncertain = True
        if valid_indices:
            selected_schema = {
                "tables": linked.selected_tables,
                "columns": [asdict(column) for column in linked.selected_columns],
            }
            rerank_input = [
                RerankCandidate(
                    candidate.canonical_sql,
                    candidate.plan,
                    candidate.execution.summary(),
                    selected_schema,
                )
                for index, candidate in enumerate(candidates) if index in valid_indices
            ]
            ranking = self.reranker.rerank(question, evidence, rerank_input)
            for result in ranking:
                original_index = valid_indices[result.candidate_index]
                result.candidate_index = original_index
                candidates[original_index].rerank = result
            if ranking:
                selected_index = ranking[0].candidate_index
                uncertain = ranking[0].uncertain or not ranking[0].above_threshold
        finished = time.perf_counter()
        return JevRunResult(
            question_id=question_id,
            db_id=schema.db_id,
            schema_linking=linked,
            candidates=candidates,
            selected_index=selected_index,
            uncertain=uncertain,
            timings_ms={
                "schema_link": round((linked_at - started) * 1000, 3),
                "beam_compile_execute": round((executed_at - linked_at) * 1000, 3),
                "rerank": round((finished - executed_at) * 1000, 3),
                "total": round((finished - started) * 1000, 3),
            },
        )


def run_task(
    runner: JevCandidateRunner,
    task: Task,
    database_root_dir: str,
) -> JevRunResult:
    schema = DatabaseManager.get_database_schema(task.db_id, database_root_dir)
    db_path = Path(database_root_dir) / task.db_id / f"{task.db_id}.sqlite"
    return runner.run(
        question=task.question,
        evidence=task.evidence,
        schema=schema,
        db_path=db_path,
        question_id=task.question_id,
    )


def main(args: argparse.Namespace) -> None:
    settings = JevSettings.from_environment()
    # Fail before loading a large task file and, critically, before any SDK/network work.
    settings.require_api_key()
    runner = JevCandidateRunner(
        settings=settings,
        execution_timeout=args.execution_timeout,
        max_literal_questions=args.max_literal_questions,
    )
    with Path(args.tasks_file_path).open("rb") as handle:
        tasks = pickle.load(handle)
    if args.question_id is not None:
        tasks = [task for task in tasks if str(task.question_id) == str(args.question_id)]
    if args.question_ids:
        requested = {item.strip() for item in args.question_ids.split(",") if item.strip()}
        tasks = [task for task in tasks if str(task.question_id) in requested]
    if args.limit is not None:
        tasks = tasks[:args.limit]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    for task in tqdm(tasks, desc="Jev constrained candidates"):
        try:
            result = run_task(runner, task, args.database_root_dir)
            path = output_dir / f"{task.question_id}.json"
            path.write_text(
                json.dumps(result.to_selection_payload(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception as exc:
            failures.append({
                "question_id": task.question_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
    (output_dir / "failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print({"completed": len(tasks) - len(failures), "failed": len(failures), "output": str(output_dir)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-file-path", required=True)
    parser.add_argument("--database-root-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--question-id")
    parser.add_argument("--question-ids", help="Comma-separated question ids for a small experiment")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--execution-timeout", type=int, default=60)
    parser.add_argument("--max-literal-questions", type=int, default=64)
    main(parser.parse_args())
