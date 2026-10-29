import json

from alphasql.algorithm.selection.execution_evidence import pool_identity
from alphasql.jev.client import JevClient
from alphasql.jev.models import ChoiceQuestion, NoulQuestion
from alphasql.jev.runtime import JevSettings
from alphasql.runner.contract_evidence_experiment import (
    _blind_state, _collect_one, _slot_differences,
)


def _row():
    return {
        "db_id": "toy", "question_id": 1,
        "question": "How many sales in 2018?", "evidence": "2018 means year = 2018",
        "full_schema": {"db_id": "toy", "tables": {
            "sales": {"columns": {"year": {"type": "INT", "description": "sale year"},
                                  "amount": {"type": "REAL", "description": "sale amount"}}}}},
        "rounds": [{"candidates": [
            {"sql": "SELECT COUNT(*) FROM sales WHERE year = 2018",
             "candidate_raw": .3, "execution_label": 1,
             "execution": {"valid": True}},
            {"sql": "SELECT SUM(amount) FROM sales WHERE year = 2018",
             "candidate_raw": .8, "execution_label": 0,
             "execution": {"valid": True}},
        ]}],
    }


def test_candidate_blind_intent_state():
    state = _blind_state(_row())
    payload = json.dumps(state)
    assert "SELECT" not in payload
    assert "candidate_raw" not in payload
    assert "execution_label" not in payload
    assert "sales" in payload


def test_only_disputed_nonambiguous_slots_trigger():
    facts = {0: {"aggregates": ["count"], "group_by": False,
                 "distinct": False, "result_rows": 1},
             1: {"aggregates": ["sum"], "group_by": False,
                 "distinct": False, "result_rows": 1}}
    contract = {"metric": "count", "multiplicity": "unspecified",
                "output_shape": "unspecified", "population": "unspecified",
                "missing_values": "unspecified"}
    assert _slot_differences(facts, contract) == ["metric"]
    contract["metric"] = "ambiguous"
    assert _slot_differences(facts, contract) == []


def test_collection_keeps_gold_and_raw_scores_out_of_jev_states(tmp_path):
    row = _row()
    execution = {"pool_sha256": pool_identity(row), "candidates": [
        {"valid": True, "row_count": 1, "column_count": 1,
         "result_hash": "a", "features": {}},
        {"valid": True, "row_count": 1, "column_count": 1,
         "result_hash": "b", "features": {}},
    ]}
    states = []

    def transport(state, questions):
        states.append(state)
        answers = {}
        for name, question in questions.items():
            if isinstance(question, ChoiceQuestion):
                choice = {"metric": "count", "multiplicity": "unspecified",
                          "output_shape": "single", "population": "restricted",
                          "missing_values": "unspecified"}[name]
                answers[name] = {"choice": choice,
                                 "probabilities": {key: float(key == choice)
                                                   for key in question.criteria}}
            elif isinstance(question, NoulQuestion):
                answers[name] = {"noul": .8}
        return {"answers": answers, "usage": {"input_tokens": 10,
                                              "output_tokens": 5}, "model": "fake"}

    client = JevClient(JevSettings(), transport=transport)
    result = _collect_one(client, row, execution, tmp_path / "one.json")
    assert result["obligations_done"]
    assert "metric" in result["triggered_slots"]
    assert len(states) == 2
    assert "candidates" not in states[0]
    for state in states:
        serialized = json.dumps(state)
        assert "candidate_raw" not in serialized
        assert "execution_label" not in serialized
