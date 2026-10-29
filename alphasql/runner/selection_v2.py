"""Cost-aware, execution-grounded selection on frozen first-round SQL pools.

View audit and ranker fitting use labelled training databases. Execution
enrichment and optional JEV collection are label-free. Validation chooses
ranking hyperparameters and stop thresholds; test is read only once.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata, spearmanr

from alphasql.algorithm.selection.execution_evidence import enrich_pool, pool_identity
from alphasql.runner.multi_angle_selector import VIEWS, _rule


EXEC_FEATURES = (
    "ast_tables", "ast_columns", "ast_joins", "ast_where", "ast_having",
    "ast_group", "ast_order", "ast_limit", "ast_distinct", "ast_count",
    "ast_avg", "ast_sum", "ast_subquery", "question_order_mismatch",
    "question_number_missing", "question_number_coverage",
    "question_count_mismatch", "question_avg_mismatch", "exec_empty",
    "exec_rows_log", "exec_columns", "exec_sample_has_null",
    "cluster_fraction", "cluster_route_fraction", "cluster_raw_mean",
    "cluster_raw_max", "cluster_entropy", "cluster_margin", "cluster_count",
)


def _read(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if source.is_dir():
        return [json.loads(file.read_text(encoding="utf-8"))
                for file in sorted(source.glob("[0-9]*.json"), key=lambda p: int(p.stem))]
    return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]


def _save(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def enrich(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    failures = []
    for row in _read(args.input):
        if int(row["question_id"]) % args.shard_count != args.shard_index:
            continue
        path = output / f'{row["db_id"]}_{row["question_id"]}.json'
        if path.is_file():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved["pool_sha256"] != pool_identity(row):
                raise ValueError(f"Frozen pool changed: {path}")
            continue
        try:
            result = enrich_pool(row, args.database_root_dir, args.timeout)
            _save(path, result)
            print(json.dumps({"question_id": row["question_id"],
                              "cluster_sizes": result["cluster_sizes"]}), flush=True)
        except Exception as exc:
            failures.append({"question_id": row["question_id"], "db_id": row["db_id"],
                             "type": type(exc).__name__, "error": str(exc)})
            print(json.dumps(failures[-1]), flush=True)
    _save(output / f"failures_{Path(args.input).stem}_{args.shard_index}.json", {"failures": failures})


def _joined(rows: Sequence[Mapping[str, Any]], view_dir: str | Path,
            execution_dir: str | Path,
            probe_dir: str | Path | None = None) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        stem = f'{row["db_id"]}_{row["question_id"]}.json'
        view = json.loads((Path(view_dir) / stem).read_text(encoding="utf-8"))
        execution = json.loads((Path(execution_dir) / stem).read_text(encoding="utf-8"))
        expected = pool_identity(row)
        if view["pool_sha256"] != expected or execution["pool_sha256"] != expected:
            raise ValueError(f"Candidate pool fingerprint mismatch: {stem}")
        probe_path = Path(probe_dir) / stem if probe_dir else None
        probe = json.loads(probe_path.read_text(encoding="utf-8")) if probe_path and probe_path.is_file() else None
        if probe and probe["pool_sha256"] != expected:
            raise ValueError(f"Counterfactual probe fingerprint mismatch: {stem}")
        candidates = row["rounds"][0]["candidates"]
        items = []
        for i, item in enumerate(candidates):
            if not item.get("execution_rechecked_valid", item["execution"]["valid"]):
                continue
            if not execution["candidates"][i]["valid"]:
                raise ValueError(f"Execution validity mismatch: {stem}/{i}")
            if str(i) not in view["views"]:
                raise ValueError(f"View evidence missing: {stem}/{i}")
            items.append({"index": i, "sql": item["sql"],
                          "raw": float(item["candidate_raw"]),
                          "views": {name: float(view["views"][str(i)][name]) for name in VIEWS},
                          "execution": execution["candidates"][i],
                          "label": item.get("execution_label"),
                          "route": item.get("route")})
        out.append({"question_id": row["question_id"], "db_id": row["db_id"],
                    "items": items, "question": row["question"],
                    "evidence": row.get("evidence") or "",
                    "schema": row["full_schema"],
                    "pool_sha256": expected,
                    "probe": probe,
                    "view_usage": {k: view.get(k, 0) for k in
                                   ("calls", "input_tokens", "output_tokens")}})
    return out


def _view_audit(pools: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pairs = [(p, a, b) for p in pools for a in p["items"] for b in p["items"]
             if a["label"] == 1 and b["label"] == 0]
    candidates = [item for p in pools for item in p["items"]]
    baseline = {p["question_id"]: max(p["items"], key=lambda i:(i["raw"],-i["index"]))
                if p["items"] else None for p in pools}
    scores = np.array([[item["raw"]] + [item["views"][v] for v in VIEWS]
                       for item in candidates], dtype=float)
    labels = np.array([item["label"] for item in candidates], dtype=int)
    report = {"questions": len(pools), "candidates": len(candidates),
              "correct": int(labels.sum()), "mixed_label_pairs": len(pairs),
              "views": {}, "mean_view_spearman": None}
    if len(scores) > 1:
        correlations = [spearmanr(scores[:,i], scores[:,j]).statistic
                        for i in range(1, 8) for j in range(i+1, 8)]
        report["mean_view_spearman"] = float(np.nanmean(correlations))
    for j, name in enumerate(("candidate_raw", *VIEWS)):
        values = scores[:,j]
        positive = values[labels == 1]
        negative = values[labels == 0]
        auc = ((rankdata(values)[labels == 1].sum()
                - len(positive) * (len(positive)+1)/2) / (len(positive)*len(negative))
               if len(positive) and len(negative) else None)
        pair_wins = sum(a["raw"] > b["raw"] if j == 0
                        else a["views"][name] > b["views"][name] for _,a,b in pairs)
        pair_ties = sum(a["raw"] == b["raw"] if j == 0
                        else a["views"][name] == b["views"][name] for _,a,b in pairs)
        report["views"][name] = {
            "candidate_auc": float(auc) if auc is not None else None,
            "within_question_pair_concordance": ((pair_wins + .5*pair_ties)/len(pairs)
                                                 if pairs else None),
            "correct_below_0_2": int(sum(v < .2 for v in positive)),
            "correct_below_0_5": int(sum(v < .5 for v in positive)),
            "wrong_above_0_8": int(sum(v >= .8 for v in negative)),
            "spearman_with_raw": (float(spearmanr(values, scores[:,0]).statistic)
                                  if len(scores)>1 else None),
        }
    # Leave one view out of the original min rule. A view is retained only when
    # removing it does not improve training-pool Top-1 and pair concordance
    # exceeds chance. No test labels are used for this filter.
    retained = []
    for name in VIEWS:
        rescued = regressed = 0
        for p in pools:
            if not p["items"]:
                continue
            full = max(p["items"], key=lambda item:(_rule(np.array([item["raw"]] +
                  [item["views"][v] for v in VIEWS])), -item["index"]))
            def without(item: Mapping[str, Any]) -> float:
                forward = min(item["views"][v] for v in list(VIEWS)[:3] if v != name)
                reverse = min(item["views"][v] for v in list(VIEWS)[3:6] if v != name)
                critical = item["views"]["critical"] if name != "critical" else item["raw"]
                return .25*item["raw"] + .30*forward + .25*reverse + .20*critical
            dropped = max(p["items"], key=lambda item:(without(item),-item["index"]))
            rescued += int(full["label"] == 1 and dropped["label"] == 0)
            regressed += int(full["label"] == 0 and dropped["label"] == 1)
        report["views"][name]["leave_one_out_rescues"] = rescued
        report["views"][name]["leave_one_out_regressions"] = regressed
        concordance = report["views"][name]["within_question_pair_concordance"]
        if concordance is not None and concordance > .5 and rescued >= regressed:
            retained.append(name)
    report["retained_views_train_only"] = retained
    return report


def audit(args: argparse.Namespace) -> None:
    pools = _joined(_read(args.input), args.view_dir, args.execution_dir)
    if any(item["label"] is None for p in pools for item in p["items"]):
        raise ValueError("Audit requires offline candidate labels")
    report = _view_audit(pools)
    _save(args.output, report)
    print(json.dumps({"questions": report["questions"],
                      "retained_views": report["retained_views_train_only"]}))


def _feature_names(mode: str, retained_views: Sequence[str]) -> tuple[str, ...]:
    views = tuple(f"view:{name}" for name in retained_views) if mode in {"views", "fusion"} else ()
    execution = tuple(f"exec:{name}" for name in EXEC_FEATURES) if mode in {"execution", "fusion"} else ()
    return views + execution


def _features(item: Mapping[str, Any], names: Sequence[str]) -> np.ndarray:
    result = []
    for name in names:
        if name.startswith("view:"):
            result.append(item["views"][name[5:]] - item["raw"])
        else:
            result.append(float(item["execution"]["features"].get(name[5:], 0.0)))
    return np.asarray(result, dtype=float)


def _fit_ranker(pools: Sequence[Mapping[str, Any]], names: Sequence[str],
                regularization: float) -> dict[str, Any]:
    x=[]; y=[]; raw=[]; pairs=[]
    for pool in pools:
        start=len(x)
        for item in pool["items"]:
            x.append(_features(item,names)); y.append(int(item["label"])); raw.append(item["raw"])
        local=range(start,len(x))
        pairs.extend((i,j) for i in local for j in local if y[i]==1 and y[j]==0)
    if not pairs or not names:
        raise ValueError("Need numeric features and mixed-label within-question pairs")
    x=np.vstack(x); raw=np.asarray(raw)
    mean=x.mean(axis=0); scale=x.std(axis=0); scale[scale<1e-8]=1.0
    z=(x-mean)/scale
    diff=np.asarray([z[i]-z[j] for i,j in pairs])
    base=np.asarray([raw[i]-raw[j] for i,j in pairs])
    def objective(w: np.ndarray) -> tuple[float,np.ndarray]:
        margin=base+diff@w
        loss=float(np.logaddexp(0,-margin).mean()+regularization*np.dot(w,w)/2)
        grad=np.asarray(-(diff.T@expit(-margin))/len(pairs)+regularization*w)
        return loss,grad
    result=minimize(objective,np.zeros(len(names)),jac=True,method="L-BFGS-B")
    if not result.success:
        raise ValueError(result.message)
    return {"features":list(names),"mean":mean.tolist(),"scale":scale.tolist(),
            "weights":result.x.tolist(),"regularization":regularization,
            "training_pairs":len(pairs)}


def _rank_score(item: Mapping[str, Any], model: Mapping[str, Any], cap: float) -> float:
    x=_features(item,model["features"])
    delta=float(((x-np.asarray(model["mean"]))/np.asarray(model["scale"]))
                @np.asarray(model["weights"]))
    return float(item["raw"]+np.clip(delta,-cap,cap))


def _select(pool: Mapping[str, Any], method: str, model: Mapping[str, Any] | None,
            cap: float | None) -> tuple[Mapping[str, Any] | None,float | None]:
    if not pool["items"]:
        return None,None
    def score(item: Mapping[str, Any]) -> float:
        if method=="jev": return item["raw"]
        if method=="rule": return _rule(np.array([item["raw"]]+
                                        [item["views"][v] for v in VIEWS]))
        if method=="consensus":
            f=item["execution"]["features"]
            return .75*item["raw"]+.25*f.get("cluster_route_fraction",0.0)
        assert model is not None and cap is not None
        return _rank_score(item,model,cap)
    if method=="jev_probe":
        selected=max(pool["items"],key=lambda item:(item["raw"],-item["index"]))
        probe=pool.get("probe") or {}
        recommendation=probe.get("recommended_index")
        if (probe.get("status")=="decisive" and recommendation is not None
                and selected["index"] in probe.get("pair",[])):
            selected=next((item for item in pool["items"] if item["index"]==recommendation),selected)
        return selected,selected["raw"]
    selected=max(pool["items"],key=lambda item:(score(item),-item["index"]))
    return selected,score(selected)


def _summary(pools: Sequence[Mapping[str, Any]], configs: Mapping[str, Any],
             thresholds: Mapping[str,float | None] | None = None) -> tuple[dict,list[dict]]:
    records=[]
    for p in pools:
        covered=any(item["label"] for item in p["items"])
        row={"db_id":p["db_id"],"question_id":p["question_id"],
             "covered":bool(covered),"methods":{}}
        for name, config in configs.items():
            chosen,score=_select(p,name,config.get("model"),config.get("cap"))
            row["methods"][name]={"index":chosen["index"] if chosen else None,
                                  "score":score,"correct":int(chosen["label"]) if chosen else 0}
        records.append(row)
    report={"questions":len(records),"covered":sum(r["covered"] for r in records),
            "methods":{},
            "extra_view_usage":{
                "calls":sum(p["view_usage"].get("calls",0) for p in pools),
                "input_tokens":sum(p["view_usage"].get("input_tokens",0) for p in pools),
                "output_tokens":sum(p["view_usage"].get("output_tokens",0) for p in pools)},
            "counterfactual_probe":{
                "attempted":sum(bool(p.get("probe")) for p in pools),
                "decisive":sum((p.get("probe") or {}).get("status")=="decisive" for p in pools),
                "llm_calls":sum((p.get("probe") or {}).get("llm_calls",0) for p in pools),
                "input_tokens":sum((p.get("probe") or {}).get("usage",{}).get("input_tokens",0)
                                   for p in pools),
                "output_tokens":sum((p.get("probe") or {}).get("usage",{}).get("output_tokens",0)
                                    for p in pools)}}
    for name in configs:
        correct=sum(r["methods"][name]["correct"] for r in records)
        baseline=[r for r in records if not r["methods"]["jev"]["correct"] and r["covered"]]
        entry={"top1_correct":correct,"covered_top1_correct":correct,
               "hard_pool_rescued":sum(r["methods"][name]["correct"] for r in baseline),
               "hard_pool_count":len(baseline),
               "rescues_vs_jev":sum(r["methods"][name]["correct"] and
                                    not r["methods"]["jev"]["correct"] for r in records),
               "regressions_vs_jev":sum(r["methods"]["jev"]["correct"] and
                                        not r["methods"][name]["correct"] for r in records)}
        if thresholds is not None:
            t=thresholds[name]
            accepted=[r for r in records if t is not None and
                      r["methods"][name]["score"] is not None and
                      r["methods"][name]["score"]>=t]
            wrong=sum(not r["methods"][name]["correct"] for r in accepted)
            entry["stop"]={"threshold":t,"accepted":len(accepted),"wrong":wrong,
                           "error_fraction":wrong/len(accepted) if accepted else None}
        report["methods"][name]=entry
    return report,records


def _thresholds(records: Sequence[Mapping[str, Any]], names: Sequence[str],
                max_error: float) -> tuple[dict[str,float | None],dict[str,list[dict]]]:
    thresholds={};curves={}
    for name in names:
        options=sorted({r["methods"][name]["score"] for r in records
                        if r["methods"][name]["score"] is not None},reverse=True)
        curve=[];eligible=[]
        for t in options:
            accepted=[r for r in records if r["methods"][name]["score"] is not None
                      and r["methods"][name]["score"]>=t]
            wrong=sum(not r["methods"][name]["correct"] for r in accepted)
            point={"threshold":t,"accepted":len(accepted),"wrong":wrong,
                   "error_fraction":wrong/len(accepted)}
            curve.append(point)
            if point["error_fraction"]<=max_error: eligible.append((len(accepted),t))
        thresholds[name]=max(eligible)[1] if eligible else None
        curves[name]=curve
    return thresholds,curves


def fit(args: argparse.Namespace) -> None:
    train=_joined(_read(args.train),args.view_dir,args.execution_dir,args.probe_dir)
    select_val=_joined(_read(args.selection_validation),args.view_dir,args.execution_dir,args.probe_dir)
    stop_val=_joined(_read(args.stop_validation),args.view_dir,args.execution_dir,args.probe_dir)
    dbs=[{p["db_id"] for p in group} for group in (train,select_val,stop_val)]
    if dbs[0]&dbs[1] or dbs[0]&dbs[2]:
        raise ValueError("Training databases overlap validation databases")
    for group in (train,select_val,stop_val):
        if any(item["label"] is None for p in group for item in p["items"]):
            raise ValueError("Fit requires offline candidate labels")
    audit_report=_view_audit(train)
    retained=audit_report["retained_views_train_only"]
    configs={name:{"model":None,"cap":None} for name in ("jev","rule","consensus")}
    if args.probe_dir:
        configs["jev_probe"]={"model":None,"cap":None}
    tuning={}
    for mode in ("views","execution","fusion"):
        names=_feature_names(mode,retained)
        if not names: continue
        trials=[]
        for reg in (.1,1.0,10.0):
            model=_fit_ranker(train,names,reg)
            for cap in (.05,.1,.2,.4):
                report,_=_summary(select_val,{"jev":configs["jev"],mode:{"model":model,"cap":cap}})
                trials.append((report["methods"][mode]["top1_correct"],-cap,-reg,model,cap))
        best=max(trials,key=lambda x:x[:3])
        configs[mode]={"model":best[3],"cap":best[4]}
        tuning[mode]={"selection_validation_top1":best[0],"cap":best[4],
                      "regularization":best[3]["regularization"]}
    stop_report,stop_rows=_summary(stop_val,configs)
    thresholds,curves=_thresholds(stop_rows,list(configs),args.max_error)
    output=Path(args.output_dir)
    model_record={"version":1,"configs":configs,"thresholds":thresholds,
                  "retained_views":retained,"training_databases":sorted(dbs[0]),
                  "selection_validation_databases":sorted(dbs[1]),
                  "stop_validation_databases":sorted(dbs[2]),
                  "training_question_ids":[p["question_id"] for p in train],
                  "selection_validation_question_ids":[p["question_id"] for p in select_val],
                  "stop_validation_question_ids":[p["question_id"] for p in stop_val],
                  "max_empirical_error":args.max_error}
    _save(output/"model.json",model_record)
    _save(output/"view_audit_train.json",audit_report)
    _save(output/"tuning.json",tuning)
    _save(output/"stop_validation_report.json",stop_report)
    _save(output/"risk_coverage.json",curves)
    print(json.dumps({"train":len(train),"selection_validation":len(select_val),
                      "stop_validation":len(stop_val),"tuning":tuning,
                      "thresholds":thresholds},ensure_ascii=False))


def evaluate(args: argparse.Namespace) -> None:
    model=json.loads(Path(args.model).read_text(encoding="utf-8"))
    if "jev_probe" in model["configs"] and not args.probe_dir:
        raise ValueError("Probe-enabled model requires --probe-dir")
    pools=_joined(_read(args.input),args.view_dir,args.execution_dir,args.probe_dir)
    dbs={p["db_id"] for p in pools}
    if dbs & set(model["training_databases"]) or dbs & set(model["selection_validation_databases"]) or dbs & set(model["stop_validation_databases"]):
        raise ValueError("Evaluation databases overlap fitting databases")
    report,records=_summary(pools,model["configs"],model["thresholds"])
    report["databases"]=sorted(dbs)
    report["model_sha256"]=hashlib.sha256(Path(args.model).read_bytes()).hexdigest()
    report["label_caveat"]="Single-database execution agreement is a proxy"
    _save(args.output,report)
    Path(args.output).with_suffix(".jsonl").write_text("".join(
        json.dumps(row,ensure_ascii=False)+"\n" for row in records),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False))


def export(args: argparse.Namespace) -> None:
    """Join only post-collection labels into a versioned future-training record."""
    rows=_read(args.input)
    pools=_joined(rows,args.view_dir,args.execution_dir,args.probe_dir)
    if any(item["label"] is None for pool in pools for item in pool["items"]):
        raise ValueError("Export requires offline labels")
    records=[]
    for row,pool in zip(rows,pools):
        raw=json.loads(Path(args.raw_dir,f'{row["question_id"]}.json').read_text(encoding="utf-8"))
        if pool_identity(raw)!=pool["pool_sha256"]:
            raise ValueError("Raw and labelled candidate pools differ")
        labelled=row["rounds"][0]["candidates"]
        records.append({"format":"jev-sql-pool-v2","split":args.split,
            "database_id":row["db_id"],"question_id":row["question_id"],
            "question":row["question"],"evidence":row.get("evidence") or "",
            "schema":row["full_schema"],"pool_sha256":pool["pool_sha256"],
            "pool_covered_label":int(any(item["label"] for item in pool["items"])),
            "candidate_labels_source":"single_database_execution_equivalence",
            "diagnosis_labels_status":"weak_automatic_only_require_human_review",
            "generator":raw["generator"],"judge":raw["judge"],
            "jev_six_outputs":row["rounds"][0]["jev"],
            "llm_usage":raw.get("llm_usage"),"jev_usage":raw.get("jev_usage_total"),
            "extra_view_usage":pool["view_usage"],
            "counterfactual_probe":pool.get("probe"),
            "candidates":[{"index":item["index"],"sql":item["sql"],
                           "route":item["route"],"candidate_label":item["label"],
                           "candidate_raw":item["raw"],"seven_views":item["views"],
                           "execution_summary":labelled[item["index"]]["execution"],
                           "execution_evidence":item["execution"],
                           "weak_diagnosis":labelled[item["index"]].get("weak_diagnosis")}
                          for item in pool["items"]]})
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("".join(json.dumps(record,ensure_ascii=False,default=str)+"\n"
                            for record in records),encoding="utf-8")
    print(json.dumps({"split":args.split,"questions":len(records),
                      "candidates":sum(len(r["candidates"]) for r in records)}))


def split_labelled(args: argparse.Namespace) -> None:
    """Partition offline labels using the database split frozen before collection."""
    protocol=json.loads(Path(args.split_file).read_text(encoding="utf-8"))
    manifest_path=Path(args.split_file).with_name("manifest.json")
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest()!=protocol["manifest_sha256"]:
        raise ValueError("Frozen manifest hash changed")
    rows=_read(args.input)
    by_id={int(row["question_id"]):row for row in rows}
    if len(by_id)!=len(rows):
        raise ValueError("Duplicate question ID in labels")
    expected={int(qid) for ids in protocol["question_ids"].values() for qid in ids}
    if set(by_id)!=expected:
        raise ValueError("Labelled question IDs differ from frozen protocol")
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    summary={}
    for split,ids in protocol["question_ids"].items():
        chosen=[]
        for qid in ids:
            row=by_id[int(qid)]
            if row["db_id"] not in protocol["databases"][split]:
                raise ValueError(f"Database split mismatch for {qid}")
            chosen.append(row)
        (output/f"{split}.jsonl").write_text("".join(
            json.dumps(row,ensure_ascii=False,default=str)+"\n" for row in chosen),
            encoding="utf-8")
        summary[split]={"questions":len(chosen),"databases":len(set(r["db_id"] for r in chosen))}
    _save(output/"split_status.json",summary)
    print(json.dumps(summary))


def merge_label_shards(args: argparse.Namespace) -> None:
    """Merge independently labelled shards only when every frozen ID appears once."""
    manifest=json.loads(Path(args.protocol_dir,"manifest.json").read_text(encoding="utf-8"))
    rows=[]
    for file in sorted(Path(args.shard_dir).glob("shard_*.jsonl")):
        rows.extend(_read(file))
    by_id={int(row["question_id"]):row for row in rows}
    expected=[int(qid) for qid in manifest["question_ids"]]
    if len(rows)!=len(by_id) or set(by_id)!=set(expected):
        raise ValueError(f"Shard labels incomplete or duplicated: {len(rows)} rows, "
                         f"{len(by_id)} unique, {len(expected)} expected")
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("".join(json.dumps(by_id[qid],ensure_ascii=False,default=str)+"\n"
                            for qid in expected),encoding="utf-8")
    print(json.dumps({"questions":len(expected),"shards":len(list(Path(args.shard_dir).glob('shard_*.jsonl')))}))


def costs(args: argparse.Namespace) -> None:
    protocol=json.loads(Path(args.split_file).read_text(encoding="utf-8"))
    summaries={}
    for split,ids in protocol["question_ids"].items():
        records=[]
        for qid in ids:
            raw=json.loads(Path(args.raw_dir,f"{qid}.json").read_text(encoding="utf-8"))
            stem=f'{raw["db_id"]}_{qid}.json'
            view=json.loads(Path(args.view_dir,stem).read_text(encoding="utf-8"))
            probe_path=Path(args.probe_dir,stem)
            probe=json.loads(probe_path.read_text(encoding="utf-8")) if probe_path.is_file() else {}
            records.append((raw,view,probe))
        def total(slot: int, *keys: str) -> int:
            return sum(int((record[slot].get(keys[0]) or {}).get(keys[1],0) or 0)
                       if len(keys)==2 else int(record[slot].get(keys[0],0) or 0)
                       for record in records)
        summaries[split]={"questions":len(records),
            "generator_calls":total(0,"llm_usage","calls"),
            "generator_input_tokens":total(0,"llm_usage","input_tokens"),
            "generator_output_tokens":total(0,"llm_usage","output_tokens"),
            "base_jev_calls":total(0,"jev_calls_total"),
            "base_jev_input_tokens":total(0,"jev_usage_total","input_tokens"),
            "base_jev_output_tokens":total(0,"jev_usage_total","output_tokens"),
            "extra_view_calls":total(1,"calls"),
            "extra_view_input_tokens":total(1,"input_tokens"),
            "extra_view_output_tokens":total(1,"output_tokens"),
            "probe_attempts":sum(bool(record[2]) for record in records),
            "probe_statuses":dict(Counter(record[2].get("status") for record in records if record[2])),
            "probe_recorded_calls":total(2,"llm_calls"),
            "probe_recorded_input_tokens":total(2,"usage","input_tokens"),
            "probe_recorded_output_tokens":total(2,"usage","output_tokens"),
            "probe_failed_calls_unmeasured":sum(record[2].get("status")=="probe_failed"
                                                for record in records)}
    _save(args.output,summaries)
    print(json.dumps(summaries,ensure_ascii=False))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    e=sub.add_parser("enrich")
    e.add_argument("--input",required=True); e.add_argument("--output-dir",required=True)
    e.add_argument("--database-root-dir",default="data/bird/train/train_databases")
    e.add_argument("--timeout",type=int,default=60)
    e.add_argument("--shard-count",type=int,default=1)
    e.add_argument("--shard-index",type=int,default=0)
    e.set_defaults(func=enrich)
    a=sub.add_parser("audit")
    a.add_argument("--input",required=True);a.add_argument("--view-dir",required=True)
    a.add_argument("--execution-dir",required=True);a.add_argument("--output",required=True)
    a.set_defaults(func=audit)
    f=sub.add_parser("fit")
    f.add_argument("--train",required=True);f.add_argument("--selection-validation",required=True)
    f.add_argument("--stop-validation",required=True);f.add_argument("--view-dir",required=True)
    f.add_argument("--execution-dir",required=True);f.add_argument("--output-dir",required=True)
    f.add_argument("--max-error",type=float,default=.1);f.set_defaults(func=fit)
    f.add_argument("--probe-dir")
    v=sub.add_parser("evaluate")
    v.add_argument("--input",required=True);v.add_argument("--view-dir",required=True)
    v.add_argument("--execution-dir",required=True);v.add_argument("--model",required=True)
    v.add_argument("--output",required=True);v.set_defaults(func=evaluate)
    v.add_argument("--probe-dir")
    x=sub.add_parser("export")
    x.add_argument("--input",required=True);x.add_argument("--raw-dir",required=True)
    x.add_argument("--view-dir",required=True);x.add_argument("--execution-dir",required=True)
    x.add_argument("--probe-dir");x.add_argument("--split",choices=("train","validation","test"),required=True)
    x.add_argument("--output",required=True);x.set_defaults(func=export)
    s=sub.add_parser("split-labelled")
    s.add_argument("--input",required=True);s.add_argument("--split-file",required=True)
    s.add_argument("--output-dir",required=True);s.set_defaults(func=split_labelled)
    m=sub.add_parser("merge-label-shards")
    m.add_argument("--protocol-dir",required=True);m.add_argument("--shard-dir",required=True)
    m.add_argument("--output",required=True);m.set_defaults(func=merge_label_shards)
    c=sub.add_parser("costs")
    c.add_argument("--split-file",required=True);c.add_argument("--raw-dir",required=True)
    c.add_argument("--view-dir",required=True);c.add_argument("--probe-dir",required=True)
    c.add_argument("--output",required=True);c.set_defaults(func=costs)
    args=parser.parse_args();args.func(args)


if __name__=="__main__":
    main()
