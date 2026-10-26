"""Distinct, single-call SQL prompting paths that preserve MCTS path provenance."""

from __future__ import annotations

import copy
from typing import Any, Sequence

from alphasql.algorithm.mcts.mcts_action import (
    MCTSNodeType, SQLGenerationAction, SQLRevisionAction, extract_sql_from_response,
)
from alphasql.algorithm.mcts.mcts_node import MCTSNode
from alphasql.llm_call.openai_llm import call_openai
from alphasql.llm_call.prompt_factory import get_prompt
from alphasql.runner.direct_candidate_runner import terminalize


PROMPT_ROUTES = ("skeleton", "query_plan")
REVISION_ROUTE = "contrastive_revision"


def generate_prompt_paths(
    parent: MCTSNode,
    llm_kwargs: dict[str, Any],
    routes: Sequence[str],
    samples: int = 1,
    anchor_sql: str | None = None,
) -> tuple[list[list[MCTSNode]], list[str]]:
    """Generate independent prompt paths; one route failing does not discard others."""
    if samples < 1:
        raise ValueError("Prompt path samples must be positive")
    unknown = set(routes) - set(PROMPT_ROUTES) - {REVISION_ROUTE}
    if unknown:
        raise ValueError(f"Unknown prompt routes: {sorted(unknown)}")
    if REVISION_ROUTE in routes and not anchor_sql:
        raise ValueError("Contrastive revision requires an anchor SQL")
    schema = parent.selected_schema_context or parent.schema_context
    arguments = {"SCHEMA_CONTEXT": schema, "QUESTION": parent.original_question,
                 "HINT": parent.hint, "ANCHOR_SQL": anchor_sql or ""}
    paths: list[list[MCTSNode]] = []
    warnings: list[str] = []
    for route in routes:
        prompt = get_prompt(f"sql_{route}", arguments)
        try:
            responses = call_openai(prompt, **dict(llm_kwargs, n=samples))
            route_count = 0
            for response in responses:
                sql = extract_sql_from_response(response)
                if not sql:
                    continue
                child = copy.copy(parent)
                revision = route == REVISION_ROUTE
                child.node_type = (MCTSNodeType.SQL_REVISION if revision
                                   else MCTSNodeType.SQL_GENERATION)
                child.parent_node = parent
                child.parent_action = SQLRevisionAction() if revision else SQLGenerationAction()
                child.depth = parent.depth + 1
                child.children = []
                child.path_nodes = parent.path_nodes + [child]
                child.final_sql_query = None
                if revision:
                    child.revised_sql_query = sql
                else:
                    child.sql_query = sql
                child.generation_route = route + (
                    "_linked" if child.selected_schema_dict else "_full")
                paths.append(terminalize(child))
                route_count += 1
            if not route_count:
                warnings.append(f"{route}: no parseable SQL")
        except Exception as error:
            warnings.append(f"{route}: {type(error).__name__}: {error}")
    return paths, warnings
