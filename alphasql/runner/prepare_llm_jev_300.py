"""Freeze exactly 300 executable-gold BIRD train tasks for the LLM/Jev study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
from typing import Any, Callable, Sequence

from alphasql.database.sql_execution import (
    SQLExecutionResultType,
    execute_sql_with_timeout,
)


def select_exact_tasks(
    primary: Sequence[Any],
    fallback: Sequence[Any],
    is_executable: Callable[[Any], bool],
    size: int,
) -> tuple[list[Any], list[int], list[int]]:
    """Keep executable primary tasks, then deterministically fill from fallback."""

    selected = []
    selected_ids = set()
    excluded = []
    replacements = []
    for task in primary:
        if len(selected) >= size:
            break
        if is_executable(task):
            selected.append(task)
            selected_ids.add(int(task.question_id))
        else:
            excluded.append(int(task.question_id))
    for task in sorted(fallback, key=lambda item: int(item.question_id)):
        question_id = int(task.question_id)
        if len(selected) >= size:
            break
        if question_id in selected_ids or not is_executable(task):
            continue
        selected.append(task)
        selected_ids.add(question_id)
        replacements.append(question_id)
    if len(selected) < size:
        raise ValueError(
            f"Only {len(selected)} executable unique tasks are available; {size} requested"
        )
    if len(selected_ids) != size:
        raise ValueError("Prepared task set contains duplicate question ids")
    return selected, excluded, replacements


def main(args: argparse.Namespace) -> None:
    with Path(args.primary_tasks).open("rb") as handle:
        primary = pickle.load(handle)
    with Path(args.fallback_tasks).open("rb") as handle:
        fallback = pickle.load(handle)
    db_root = Path(args.db_root)
    execution_cache: dict[int, bool] = {}

    def is_executable(task: Any) -> bool:
        question_id = int(task.question_id)
        if question_id not in execution_cache:
            db_path = db_root / task.db_id / f"{task.db_id}.sqlite"
            result = execute_sql_with_timeout(
                str(db_path), task.sql, args.execution_timeout
            )
            execution_cache[question_id] = (
                result.result_type is SQLExecutionResultType.SUCCESS
            )
        return execution_cache[question_id]

    tasks, excluded, replacements = select_exact_tasks(
        primary, fallback, is_executable, args.size
    )
    task_path = Path(args.output_tasks)
    task_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = task_path.with_suffix(task_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(tasks, handle)
    temporary.replace(task_path)

    ids = [int(task.question_id) for task in tasks]
    subset_path = Path(args.output_subset)
    subset_path.parent.mkdir(parents=True, exist_ok=True)
    subset_path.write_text(
        json.dumps([{"question_id": value} for value in ids], indent=2) + "\n",
        encoding="utf-8",
    )
    split = {
        "definition": {
            "name": "exactly 300 executable-gold BIRD train questions",
            "source": str(Path(args.primary_tasks)),
            "fallback_source": str(Path(args.fallback_tasks)),
            "selection": "retain executable primary tasks, fill by ascending fallback question id",
        },
        "calibration_databases": [],
        "calibration_ids": [],
        "evaluation_ids": ids,
        "calibration_count": 0,
        "evaluation_count": len(ids),
        "excluded_non_executable_gold": excluded,
        "replacement_ids": replacements,
        "frozen_before_api_runs": True,
    }
    split_path = Path(args.output_split)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "evaluation_count": len(ids),
        "excluded_primary": excluded,
        "replacement_ids": replacements,
        "tasks": str(task_path),
        "subset": str(subset_path),
        "split": str(split_path),
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-tasks", required=True)
    parser.add_argument("--fallback-tasks", required=True)
    parser.add_argument("--db-root", required=True)
    parser.add_argument("--output-tasks", required=True)
    parser.add_argument("--output-subset", required=True)
    parser.add_argument("--output-split", required=True)
    parser.add_argument("--size", type=int, default=300)
    parser.add_argument("--execution-timeout", type=int, default=15)
    main(parser.parse_args())
