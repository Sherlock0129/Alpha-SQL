"""Generate diverse SQL candidates without running the full MCTS search."""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pickle
import json
from typing import Any, Dict, Iterable, List

from tqdm import tqdm

import alphasql.algorithm.mcts.mcts_action as actions
from alphasql.algorithm.mcts.mcts_action import (
    EndAction,
    MCTSNodeType,
    SQLGenerationAction,
    SchemaSelectionAction,
)
from alphasql.algorithm.mcts.mcts_node import MCTSNode
from alphasql.database.utils import build_table_ddl_statement
from alphasql.llm_call.openai_llm import reset_request_counter
from alphasql.llm_call.runtime import configure_environment
from alphasql.runner.task import Task


def terminalize(node: MCTSNode) -> List[MCTSNode]:
    terminal = copy.copy(node)
    terminal.node_type = MCTSNodeType.END
    terminal.parent_node = node
    terminal.parent_action = EndAction()
    terminal.depth = node.depth + 1
    terminal.children = []
    terminal.final_sql_query = (
        node.sql_query if node.node_type == MCTSNodeType.SQL_GENERATION
        else node.revised_sql_query
    )
    terminal.path_nodes = node.path_nodes + [terminal]
    return terminal.path_nodes


def deduplicate_paths(paths: Iterable[List[MCTSNode]]) -> List[List[MCTSNode]]:
    unique = []
    seen = set()
    for path in paths:
        sql = getattr(path[-1], "final_sql_query", None)
        key = " ".join((sql or "").split()).lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _root_node(task: Task, db_root_dir: str) -> MCTSNode:
    schema_context = "\n".join([
        build_table_ddl_statement(
            task.table_schema_dict[table_name].to_dict(),
            add_value_description=True,
            add_column_description=True,
            add_value_examples=True,
            add_expanded_column_name=True,
        )
        for table_name in task.table_schema_dict
    ])
    root = MCTSNode(
        MCTSNodeType.ROOT,
        db_id=task.db_id,
        db_root_dir=db_root_dir,
        original_question=task.question,
        hint=task.evidence,
        schema_context=schema_context,
        table_schema_dict=task.table_schema_dict,
    )
    root.path_nodes = [root]
    return root


def generate_one(payload) -> Dict[str, Any]:
    task, db_root_dir, save_root_dir, llm_kwargs, samples = payload
    configure_environment()
    reset_request_counter()
    actions.SQL_GENERATION_LLM_KWARGS_N = samples
    task = task.model_copy(update={"sql": None})
    root = _root_node(task, db_root_dir)
    generation = SQLGenerationAction()
    paths = []
    errors = []

    try:
        paths.extend(terminalize(node) for node in generation.create_children_nodes(root, llm_kwargs))
    except Exception as error:
        errors.append(f"full_schema: {error}")

    try:
        linked_nodes = SchemaSelectionAction().create_children_nodes(root, llm_kwargs)
        for linked_node in linked_nodes:
            try:
                paths.extend(
                    terminalize(node)
                    for node in generation.create_children_nodes(linked_node, llm_kwargs)
                )
            except Exception as error:
                errors.append(f"linked_schema_generation: {error}")
    except Exception as error:
        errors.append(f"schema_selection: {error}")

    paths = deduplicate_paths(paths)
    if not paths:
        return {
            "question_id": task.question_id,
            "candidate_count": 0,
            "warnings": errors,
            "failed": True,
        }
    save_path = Path(save_root_dir) / f"{task.question_id}.pkl"
    temporary_path = save_path.with_suffix(".pkl.tmp")
    with temporary_path.open("wb") as handle:
        pickle.dump(paths, handle)
    temporary_path.replace(save_path)
    return {
        "question_id": task.question_id,
        "candidate_count": len(paths),
        "warnings": errors,
        "failed": False,
    }


def safe_generate_one(payload) -> Dict[str, Any]:
    try:
        return generate_one(payload)
    except Exception as error:
        return {
            "question_id": payload[0].question_id,
            "candidate_count": 0,
            "warnings": [f"unexpected: {error}"],
            "failed": True,
        }


def main(args):
    configure_environment()
    save_root = Path(args.save_root_dir)
    save_root.mkdir(parents=True, exist_ok=True)
    with Path(args.tasks_file_path).open("rb") as handle:
        tasks = pickle.load(handle)
    done = {int(path.stem) for path in save_root.glob("*.pkl")}
    tasks = [task for task in tasks if task.question_id not in done]
    if args.limit is not None:
        tasks = tasks[:args.limit]
    llm_kwargs = {
        "model": args.model,
        "n": 1,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "n_strategy": "multiple",
    }
    payloads = [
        (task, args.db_root_dir, str(save_root), llm_kwargs, args.samples)
        for task in tasks
    ]
    summaries = []
    with ProcessPoolExecutor(max_workers=args.n_processes) as executor:
        for summary in tqdm(
            executor.map(safe_generate_one, payloads), total=len(payloads), desc="Direct candidates"
        ):
            summaries.append(summary)
    payload_by_id = {payload[0].question_id: payload for payload in payloads}
    for attempt in range(args.retries):
        failed = [summary for summary in summaries if summary["failed"]]
        if not failed:
            break
        retry_payloads = [payload_by_id[summary["question_id"]] for summary in failed]
        print(f"Retrying {len(retry_payloads)} failed questions, attempt {attempt + 1}/{args.retries}")
        with ProcessPoolExecutor(max_workers=min(args.n_processes, len(retry_payloads))) as executor:
            retry_summaries = list(executor.map(safe_generate_one, retry_payloads))
        replacements = {summary["question_id"]: summary for summary in retry_summaries}
        summaries = [replacements.get(summary["question_id"], summary) for summary in summaries]
    warning_count = sum(bool(summary["warnings"]) for summary in summaries)
    candidate_count = sum(summary["candidate_count"] for summary in summaries)
    failures = [summary for summary in summaries if summary["failed"]]
    failure_path = save_root / "failures.json"
    failure_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print({
        "completed_questions": len(summaries),
        "new_candidates": candidate_count,
        "questions_with_partial_warnings": warning_count,
        "already_complete": len(done),
        "failed_questions": len(failures),
        "failure_report": str(failure_path),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-file-path", required=True)
    parser.add_argument("--db-root-dir", required=True)
    parser.add_argument("--save-root-dir", required=True)
    parser.add_argument("--model", default="qwen3-coder-flash")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--n-processes", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retries", type=int, default=2)
    main(parser.parse_args())
