"""Train-only, question-balanced curriculum controller for frozen SQL pairs.

SQL contrasts below are deliberately weak structural labels, not semantic truth.
No labels produced here are model input features.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from sqlglot import exp, parse_one


GROUPS = ("projection_multiplicity", "filter_literal", "relation",
          "computation_ordering", "unknown_mixed")


def group_records(records):
    grouped = defaultdict(list)
    for i, row in enumerate(records):
        grouped[str(row["question_id"])].append(i)
    return dict(grouped)


def sc_index(records, indices):
    return max(indices, key=lambda i: (
        records[i]["features"].get("exec_result_group_size", 0),
        -int(records[i]["candidate_index"])))


def weak_membership(positive, negative):
    """Multi-label normalized AST differences against an existing positive SQL.

    Alias and equivalent-expression changes can be false positives; parse failures
    and execution failures go to unknown. No gold SQL is consulted here.
    """
    labels = []
    if not negative["features"].get("exec_success", 0):
        labels = [4]
    else:
        try:
            a, b = (parse_one(r["sql"], read="sqlite") for r in (positive, negative))

            def clauses(tree, classes):
                return tuple(sorted(node.sql(dialect="sqlite", normalize=True)
                                    for kind in classes for node in tree.find_all(kind)))

            def projection(tree):
                return tuple(tuple(e.sql(dialect="sqlite", normalize=True)
                                   for e in s.expressions) for s in tree.find_all(exp.Select))

            if (projection(a), clauses(a, (exp.Distinct,))) != (
                    projection(b), clauses(b, (exp.Distinct,))):
                labels.append(0)
            if clauses(a, (exp.Where,)) != clauses(b, (exp.Where,)):
                labels.append(1)
            if clauses(a, (exp.Table, exp.Join)) != clauses(b, (exp.Table, exp.Join)):
                labels.append(2)
            kinds = (exp.AggFunc, exp.Group, exp.Having, exp.Order, exp.Limit,
                     exp.Union, exp.Intersect, exp.Except)
            if clauses(a, kinds) != clauses(b, kinds):
                labels.append(3)
        except Exception:
            labels = [4]
    labels = labels or [4]
    result = np.zeros(len(GROUPS), dtype=np.float64)
    result[labels] = 1.0 / len(labels)
    return result


@dataclass
class PairPool:
    positive: np.ndarray
    negative: np.ndarray
    question_ids: list[str]
    membership: np.ndarray
    q0: np.ndarray
    keep: np.ndarray

    @classmethod
    def from_records(cls, records):
        positive, negative, questions, membership, keep_questions = [], [], [], [], set()
        for qid, indices in group_records(records).items():
            pos = [i for i in indices if records[i]["label"] == 1]
            neg = [i for i in indices if records[i]["label"] == 0]
            if not pos or not neg:
                continue
            if records[sc_index(records, indices)]["label"]:
                keep_questions.add(qid)
            for p in pos:
                for n in neg:
                    positive.append(p)
                    negative.append(n)
                    questions.append(qid)
                    membership.append(weak_membership(records[p], records[n]))
        if not positive:
            raise ValueError("No train positive-negative pairs in frozen pool")
        by_question = defaultdict(list)
        for i, qid in enumerate(questions):
            by_question[qid].append(i)
        q0 = np.zeros(len(positive))
        keep = np.zeros(len(positive))
        if not keep_questions:
            raise ValueError("R1 cannot be evaluated: no SC-correct pairwise train questions")
        for qid, indices in by_question.items():
            q0[indices] = 1.0 / len(by_question) / len(indices)
            if qid in keep_questions:
                keep[indices] = 1.0 / len(keep_questions) / len(indices)
        return cls(np.asarray(positive), np.asarray(negative), questions,
                   np.asarray(membership), q0, keep)

    def support(self):
        return {name: {
            "pairs": int((self.membership[:, g] > 0).sum()),
            "distinct_questions": len({q for q, a in zip(self.question_ids, self.membership[:, g]) if a > 0}),
            "question_equivalents": float(len(set(self.question_ids)) * (self.q0 * self.membership[:, g]).sum()),
        } for g, name in enumerate(GROUPS)}


class CurriculumController:
    def __init__(self, pool, strategy, eta=1.0, decay=0.9, rho=0.25, kappa=10.0):
        if strategy not in ("uniform", "loss_progress", "error_aware"):
            raise ValueError(f"Unsupported strategy: {strategy}")
        self.pool, self.strategy, self.eta = pool, strategy, eta
        self.decay, self.rho, self.kappa = decay, rho, kappa
        self.ema = None
        self.probabilities = pool.q0.copy()
        self.diagnostics = {}

    def probe(self, losses, margins):
        losses, margins = np.asarray(losses), np.asarray(margins)
        if losses.shape != self.pool.q0.shape or not np.isfinite(losses).all():
            raise ValueError("Probe needs one finite loss per frozen train pair")
        previous = self.ema
        self.ema = losses.copy() if previous is None else self.decay * previous + (1 - self.decay) * losses
        progress = np.zeros_like(losses) if previous is None else np.clip(
            (previous - self.ema) / np.maximum(previous, 1e-8), 0, 1)
        dynamic = self.ema / (1 + self.ema) + progress
        error = (margins <= 0).astype(float)
        q0, a = self.pool.q0, self.pool.membership
        mass = q0[:, None] * a * len(set(self.pool.question_ids))
        support = mass.sum(axis=0)
        global_error, global_progress = q0 @ error, q0 @ progress
        group_error = (mass.T @ error + self.kappa * global_error) / (support + self.kappa)
        group_progress = (mass.T @ progress + self.kappa * global_progress) / (support + self.kappa)
        group_demand = group_error + group_progress
        # eta==0 takes exactly LP's arithmetic path and consumes no RNG.
        if self.strategy == "uniform":
            weights = np.ones_like(dynamic)
        else:
            exponent = dynamic
            if self.strategy == "error_aware" and self.eta != 0:
                exponent = exponent + self.eta * (a @ group_demand)
            weights = np.clip(np.exp(exponent), 0.1, 10.0)
        weighted = q0 * weights
        self.probabilities = self.rho * q0 + (1 - self.rho) * weighted / weighted.sum()
        self.probabilities /= self.probabilities.sum()
        self.diagnostics = {"ema": self.ema.tolist(), "progress": progress.tolist(),
                            "group_demand": group_demand.tolist(),
                            "probabilities": self.probabilities.tolist()}
        return self.probabilities.copy()


class SharedDrawStreams:
    """Arm-independent seeds; sampler, retention and anchors never share state."""
    def __init__(self, seed, records):
        self.base = np.random.default_rng(seed + 1000)
        self.keep = np.random.default_rng(seed + 2000)
        self.anchor = np.random.default_rng(seed + 3000)
        self.questions = list(group_records(records).values())

    def draw(self, pool, probabilities, warmup=False, pair_batch=8, anchor_batch=16):
        if pair_batch % 4:
            raise ValueError("R1 fixed keep slots requires pair batch divisible by four")
        base = self.base.choice(len(pool.q0), size=3 * pair_batch // 4,
                                p=pool.q0 if warmup else probabilities)
        keep = self.keep.choice(len(pool.q0), size=pair_batch // 4, p=pool.keep)
        anchors = []
        for _ in range(anchor_batch):
            indices = self.questions[int(self.anchor.integers(len(self.questions)))]
            anchors.append(int(indices[int(self.anchor.integers(len(indices)))]))
        return base.tolist(), keep.tolist(), anchors
