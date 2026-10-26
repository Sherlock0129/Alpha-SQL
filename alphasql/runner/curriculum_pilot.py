"""Offline P0a: uniform/replay, LP/replay, EA/replay and eta-zero parity.

This is a mechanism sanity check, not a time-to-target effectiveness experiment.
Only the frozen train/validation IDs are processed; historical test is excluded.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import time
from unittest.mock import patch

# Must precede importing torch / creating a CUDA context.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch.nn import functional as F
import yaml

from alphasql.algorithm.selection.curriculum import (
    GROUPS, CurriculumController, PairPool, SharedDrawStreams, group_records, sc_index,
)
from alphasql.algorithm.selection.neural_reranker import SchemaAwareMLP
from alphasql.algorithm.selection.schema_signal import Standardizer


ARMS = {"U1": ("uniform", 0.0), "LP1": ("loss_progress", 0.0),
        "EA1": ("error_aware", 1.0), "NoGroupEA1": ("error_aware", 0.0)}


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def write_json(path, value):
    Path(path).write_bytes(encoded(value))


def write_jsonl(path, rows):
    Path(path).write_bytes(b"".join(encoded(row) + b"\n" for row in rows))


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@contextmanager
def offline_only():
    def denied(*args, **kwargs):
        raise RuntimeError("Network access forbidden in offline curriculum pilot")
    with ExitStack() as stack:
        for target in ("socket.socket.connect", "socket.socket.connect_ex",
                       "socket.create_connection", "socket.socket.sendto"):
            stack.enter_context(patch(target, denied))
        yield


def resolve_settings(protocol):
    p, s, o, m = (protocol[k] for k in ("pilot", "sampling", "optimizer", "model"))
    return {"updates": p["max_updates"], "warmup": p["uniform_warmup_updates"],
            "probe_every": p["probe_every_updates"], "pair_batch": p["pair_batch_size"],
            "anchor_batch": p["anchor_batch_size"], "hidden_sizes": m["hidden_sizes"],
            "dropout": m["dropout"], "lr": o["learning_rate"],
            "weight_decay": o["weight_decay"], "clip": o["gradient_clip_norm"],
            "pointwise_weight": protocol["objective"]["pointwise_weight"],
            "decay": s["ema_decay"], "rho": s["uniform_floor_rho"],
            "kappa": s["shrinkage_question_equivalents"],
            "eta": s["semantic_coefficient_eta"]}


def audit_scope(records, split):
    train_ids, val_ids, test_ids = (set(map(str, split[k]))
                                   for k in ("train_ids", "validation_ids", "test_ids"))
    if train_ids & val_ids or (train_ids | val_ids) & test_ids:
        raise ValueError("Frozen split overlaps")
    actual = {str(r["question_id"]) for r in records}
    if not actual <= train_ids | val_ids:
        raise ValueError("Candidate records outside train/validation scope")
    if any(r["candidate_source"] != "mcts" for r in records):
        raise ValueError("Synthetic/gold candidates forbidden")
    train = [r for r in records if str(r["question_id"]) in train_ids]
    val = [r for r in records if str(r["question_id"]) in val_ids]
    if {str(r["question_id"]) for r in train} != train_ids:
        raise ValueError("Frozen train question missing after execution labeling")
    if {str(r["question_id"]) for r in val} != val_ids:
        raise ValueError("Frozen validation question missing after execution labeling")
    return train, val


def selection_evaluation(records, logits):
    rows = []
    for qid, indices in group_records(records).items():
        baseline = sc_index(records, indices)
        valid = [i for i in indices if records[i]["features"]["exec_success"]]
        selected = max(valid, key=lambda i: (float(logits[i]), -int(records[i]["candidate_index"]))) if valid else None
        correct = bool(selected is not None and records[selected]["label"])
        sc_correct = bool(records[baseline]["label"])
        rows.append({"question_id": qid, "selected_candidate_index": None if selected is None else records[selected]["candidate_index"],
                     "sc_candidate_index": records[baseline]["candidate_index"],
                     "correct": correct, "sc_correct": sc_correct,
                     "covered": any(records[i]["label"] for i in indices),
                     "logits": [float(logits[i]) for i in indices]})
    n = len(rows)
    summary = {"questions": n, "correct": sum(r["correct"] for r in rows),
               "sc_correct": sum(r["sc_correct"] for r in rows),
               "covered": sum(r["covered"] for r in rows),
               "fixes": sum(r["correct"] and not r["sc_correct"] for r in rows),
               "regressions": sum(r["sc_correct"] and not r["correct"] for r in rows)}
    summary["ex"] = summary["correct"] / n
    summary["regression_rate"] = summary["regressions"] / n
    summary["conditional_regression"] = summary["regressions"] / max(1, summary["sc_correct"])
    return summary, rows


def state_digest(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_arm(name, train, val, feature_names, pool, settings, seed, device, output):
    start = time.perf_counter()
    output.mkdir()
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    matrix = lambda rows: np.asarray([[r["features"].get(f, 0.0) for f in feature_names]
                                     for r in rows], dtype=np.float32)
    scaler = Standardizer().fit(matrix(train))
    x = torch.as_tensor(scaler.transform(matrix(train)), dtype=torch.float32, device=device)
    vx = torch.as_tensor(scaler.transform(matrix(val)), dtype=torch.float32, device=device)
    y = torch.tensor([r["label"] for r in train], dtype=torch.float32, device=device)
    positive_weight = (len(train) - float(y.sum())) / float(y.sum())
    model = SchemaAwareMLP(len(feature_names), settings["hidden_sizes"], settings["dropout"]).to(device)
    initial_hash = state_digest(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    strategy, eta = ARMS[name]
    if name == "EA1":
        eta = settings["eta"]
    controller = CurriculumController(pool, strategy, eta, settings["decay"], settings["rho"], settings["kappa"])
    streams = SharedDrawStreams(seed, train)
    exposures, trace, probes, predictions = [], [], [], []
    costs = {"training_seconds": 0.0, "probe_seconds": 0.0, "sampling_seconds": 0.0,
             "checkpoint_seconds": 0.0}
    sync(device)
    costs["initialization_seconds"] = time.perf_counter() - start

    def probe(step):
        sync(device)
        tick = time.perf_counter()
        model.eval()
        with torch.no_grad():
            logits = model(x).cpu().numpy()
            validation_logits = model(vx).cpu().numpy()
        margins = logits[pool.positive] - logits[pool.negative]
        losses = np.logaddexp(0.0, -margins.astype(np.float64))
        controller.probe(losses, margins)
        train_metrics, _ = selection_evaluation(train, logits)
        metrics, per_question = selection_evaluation(val, validation_logits)
        sync(device)
        costs["probe_seconds"] += time.perf_counter() - tick
        probes.append({"step": step, "train": train_metrics, "validation": metrics,
                       "train_pair_loss_mean": float(losses.mean()),
                       "model_sha256": state_digest(model.state_dict()), **controller.diagnostics})
        predictions.append({"step": step, "questions": per_question})
        tick = time.perf_counter()
        torch.save({"model_state": model.state_dict(), "feature_names": feature_names,
                    "scaler_mean": scaler.mean, "scaler_scale": scaler.scale,
                    "step": step, "seed": seed}, output / f"checkpoint_{step:04d}.pt")
        costs["checkpoint_seconds"] += time.perf_counter() - tick

    probe(0)
    for step in range(1, settings["updates"] + 1):
        tick = time.perf_counter()
        base, keep, anchors = streams.draw(pool, controller.probabilities,
            step <= settings["warmup"], settings["pair_batch"], settings["anchor_batch"])
        pairs = np.asarray(base + keep)
        # Forward only unique candidates in this batch, not the entire train pool.
        unique, inverse = np.unique(np.concatenate((pool.positive[pairs], pool.negative[pairs], anchors)), return_inverse=True)
        exposures.append({"step": step, "base_pairs": base, "keep_pairs": keep, "anchors": anchors,
                          "unique_candidates_forwarded": len(unique)})
        costs["sampling_seconds"] += time.perf_counter() - tick
        sync(device)
        tick = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        scores = model(x[torch.as_tensor(unique, device=device)])
        local = torch.as_tensor(inverse, dtype=torch.long, device=device)
        b = len(pairs)
        rank_loss = F.softplus(-(scores[local[:b]] - scores[local[b:2*b]])).mean()
        point_loss = F.binary_cross_entropy_with_logits(
            scores[local[2*b:]], y[torch.as_tensor(anchors, device=device)],
            pos_weight=torch.tensor(positive_weight, device=device))
        loss = rank_loss + settings["pointwise_weight"] * point_loss
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["clip"])
        optimizer.step()
        values = {"step": step, "loss": float(loss.detach()), "rank_loss": float(rank_loss.detach()),
                  "point_loss": float(point_loss.detach()), "gradient_norm": float(norm)}
        sync(device)
        costs["training_seconds"] += time.perf_counter() - tick
        trace.append(values)
        if step % settings["probe_every"] == 0 or step == settings["updates"]:
            probe(step)
    sync(device)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    costs["wall_seconds_before_log_serialization"] = time.perf_counter() - start
    tick = time.perf_counter()
    write_jsonl(output / "exposure.jsonl", exposures)
    write_jsonl(output / "train_trace.jsonl", trace)
    write_jsonl(output / "probe_trace.jsonl", probes)
    write_jsonl(output / "checkpoint_predictions.jsonl", predictions)
    costs["log_serialization_seconds"] = time.perf_counter() - tick
    costs["wall_seconds"] = time.perf_counter() - start
    write_json(output / "cost_breakdown.json", costs)
    exposure_counts = np.bincount([p for row in exposures for p in row["base_pairs"] + row["keep_pairs"]], minlength=len(pool.q0))
    summary = {"arm": name, "seed": seed, "initial_sha256": initial_hash,
               "final_sha256": state_digest(state), "initial_validation": probes[0]["validation"],
               "final_validation": probes[-1]["validation"], "final_train": probes[-1]["train"],
               "pair_draws": int(exposure_counts.sum()), "unique_pairs_exposed": int((exposure_counts > 0).sum()),
               "group_fractional_exposures": dict(zip(GROUPS, (exposure_counts @ pool.membership).tolist())),
               "cost": costs, "eta": eta, "strategy": strategy}
    write_json(output / "summary.json", summary)
    print(json.dumps({"arm": name, "final_validation": summary["final_validation"],
                      "wall_seconds": costs["wall_seconds"]}), flush=True)
    return {"summary": summary, "state": state, "exposures": exposures,
            "trace": trace, "probes": probes, "predictions": predictions}


def equivalence_report(runs, warmup):
    left, right = runs["LP1"], runs["NoGroupEA1"]
    differences = {name: float(torch.max(torch.abs(left["state"][name] - right["state"][name])))
                   for name in left["state"]}
    checks = {"initial_state_equal_all_arms": len({r["summary"]["initial_sha256"] for r in runs.values()}) == 1,
              "warmup_draws_equal_all_arms": all(r["exposures"][:warmup] == left["exposures"][:warmup] for r in runs.values()),
              "warmup_losses_equal_all_arms": all(r["trace"][:warmup] == left["trace"][:warmup] for r in runs.values()),
              "keep_and_anchor_streams_equal_all_arms": all(
                  [(r["keep_pairs"], r["anchors"]) for r in run["exposures"]] ==
                  [(r["keep_pairs"], r["anchors"]) for r in left["exposures"]] for run in runs.values()),
              "lp_eta0_exposure_exact": left["exposures"] == right["exposures"],
              "lp_eta0_train_trace_exact": left["trace"] == right["trace"],
              "lp_eta0_probabilities_exact": all(a["probabilities"] == b["probabilities"] for a, b in zip(left["probes"], right["probes"])),
              "lp_eta0_predictions_exact": left["predictions"] == right["predictions"],
              "lp_eta0_final_tensors_bit_exact": all(torch.equal(v, right["state"][k]) for k, v in left["state"].items())}
    return {"passed": all(checks.values()), "checks": checks,
            "max_parameter_absolute_difference": max(differences.values()),
            "scope": "P0a one-seed finite-trajectory parity; not mathematical proof for all inputs",
            "ea_vs_lp_base_draws_differ": any(a["base_pairs"] != b["base_pairs"] for a, b in zip(left["exposures"], runs["EA1"]["exposures"]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="config/sql_selector_curriculum_protocol.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    protocol = yaml.safe_load(Path(args.protocol).read_text(encoding="utf-8"))
    settings = resolve_settings(protocol)
    data = protocol["frozen_data"]
    split = json.loads(Path(data["pilot_split"]).read_text(encoding="utf-8"))
    allowed_ids = set(map(str, split["train_ids"] + split["validation_ids"]))
    with offline_only():
        tick = time.perf_counter()
        from alphasql.runner.schema_signal_experiment import build_records
        print(f"Preparing only {len(allowed_ids)} frozen train/validation questions, network blocked", flush=True)
        records = build_records(data["pilot_candidate_dir"], data["pilot_data_path"], data["pilot_db_root"],
                                question_ids=sorted(allowed_ids))
        train, val = audit_scope(records, split)
        pool = PairPool.from_records(train)
        feature_names = sorted({name for r in train for name in r["features"]})
        if any(not f.startswith(("sql_", "schema_", "exec_", "mcts_")) for f in feature_names):
            raise ValueError("Unexpected input feature; audit for label leakage")
        if "exec_result_group_share" not in feature_names:
            raise ValueError("Self-consistency feature must remain enabled")
        pair_rows = [{"pair_id": i, "question_id": pool.question_ids[i],
                      "positive_record_index": int(p), "negative_record_index": int(n),
                      "membership": pool.membership[i].tolist(), "q0": float(pool.q0[i]),
                      "q_keep": float(pool.keep[i])} for i, (p, n) in enumerate(zip(pool.positive, pool.negative))]
        write_json(output / "frozen_records.json", {"train": train, "validation": val})
        write_json(output / "weak_pair_labels.json", {"groups": GROUPS, "pairs": pair_rows, "support": pool.support()})
        code_files = [Path(__file__), Path("alphasql/algorithm/selection/curriculum.py"),
                      Path("alphasql/algorithm/selection/neural_reranker.py"),
                      Path("alphasql/algorithm/selection/schema_signal.py"),
                      Path("alphasql/runner/schema_signal_experiment.py"),
                      Path("alphasql/database/sql_execution.py"), Path("alphasql/database/database_manager.py")]
        source_files = {str(p): file_hash(p) for p in sorted(Path(data["pilot_candidate_dir"]).glob("*.pkl")) if p.stem in allowed_ids}
        manifest = {"protocol_sha256": file_hash(args.protocol), "split_sha256": file_hash(data["pilot_split"]),
                    "train_sha256": digest(train), "validation_sha256": digest(val),
                    "features": feature_names, "feature_sha256": digest(feature_names),
                    "error_label_sha256": digest(pair_rows), "code_sha256": {str(p): file_hash(p) for p in code_files},
                    "source_pickle_sha256": source_files, "train_questions": len(group_records(train)),
                    "validation_questions": len(group_records(val)), "train_candidates": len(train),
                    "validation_candidates": len(val), "train_pairwise_questions": len(set(pool.question_ids)),
                    "train_pairs": len(pool.q0), "keep_questions": len({q for q, k in zip(pool.question_ids, pool.keep) if k > 0}),
                    "test_ids_processed": [], "test_candidate_files_opened": 0,
                    "source_note": "direct cache, legacy builder source tag mcts retained; no MCTS run here",
                    "label_note": "weak alias-sensitive AST contrasts; no manual audit; sparse groups not confirmatory",
                    "support": pool.support(), "preparation_seconds": time.perf_counter() - tick,
                    "execution_metric": "existing order-insensitive multiset execution signature, not official BIRD evaluator",
                    "candidate_generation_api_calls": 0}
        write_json(output / "frozen_manifest.json", manifest)
        write_json(output / "threshold_manifest.json", {"quality_tau": None, "regression_r_max": None,
                                                       "time_to_target_claim_allowed": False})
        print(json.dumps({k: manifest[k] for k in ("train_questions", "validation_questions", "train_candidates", "train_pairs", "support")}), flush=True)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
        # Common untimed backend startup is reported separately, not attributed to U1.
        tick = time.perf_counter()
        torch.manual_seed(args.seed)
        warm_model = SchemaAwareMLP(len(feature_names), settings["hidden_sizes"], settings["dropout"]).to(device)
        warm_model(torch.zeros((32, len(feature_names)), device=device)).sum().backward()
        sync(device)
        startup_seconds = time.perf_counter() - tick
        del warm_model
        write_json(output / "run_config.json", {"settings": settings, "seed": args.seed,
            "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
            "torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version(),
            "deterministic": True, "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "arms": ARMS, "checkpoint_every": settings["probe_every"],
            "backend_startup_seconds": startup_seconds, "thresholds_frozen": False,
            "training_label_source": "existing candidate execution equality only",
            "validation_early_stopping": False, "legacy_pair_weights": False,
            "inference": "raw valid-candidate logit top1; no gate/fusion; empty valid",
            "pointwise_positive_weight": (len(train) - sum(r["label"] for r in train)) / sum(r["label"] for r in train)})
        runs = {name: train_arm(name, train, val, feature_names, pool, settings, args.seed, device, output / name)
                for name in ARMS}
        parity = equivalence_report(runs, settings["warmup"])
        write_json(output / "equivalence_report.json", parity)
        write_json(output / "paired_summary.json", {name: r["summary"] for name, r in runs.items()})
        lines = ["# SQL curriculum P0a", "", "One seed, validation only; no target-time or semantic effectiveness claim.", "",
                 "| Arm | Final correct | Fixes | Regressions | Wall seconds |", "|---|---:|---:|---:|---:|"]
        for name, r in runs.items():
            s = r["summary"]
            v = s["final_validation"]
            lines.append(f"| {name} | {v['correct']}/{v['questions']} | {v['fixes']} | {v['regressions']} | {s['cost']['wall_seconds']:.3f} |")
        lines += ["", f"LP vs eta=0 parity passed: {parity['passed']}", "", json.dumps(parity, indent=2)]
        (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
        print(json.dumps(parity), flush=True)
        if not parity["passed"]:
            raise SystemExit("Equivalence check FAILED; do not interpret arm performance")


if __name__ == "__main__":
    main()
