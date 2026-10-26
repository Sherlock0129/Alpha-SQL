"""Offline experiment report. Run with the project .venv (sqlglot installed).

Consumes frozen manifests, SQLite evaluation outputs, and API usage only. No API
calls or model-based error judgements; structural differences are diagnostic flags,
not claims that a different but equivalent SQL formulation is wrong.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import random
from statistics import mean, median
import tomllib

from deepeye.runner import ROOT, run_dir, write_json


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def wilson(successes, total):
    if not total:
        return None
    z = 1.95996398454
    p = successes / total
    center = (p + z*z/(2*total))/(1+z*z/total)
    half = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/(1+z*z/total)
    return [center-half, center+half]


def paired_difference(rows, first="selected_ex_multiset", second="sc_ex_multiset"):
    if not rows:
        return {}
    delta = [int(row[first])-int(row[second]) for row in rows]
    rng = random.Random(20261019)
    samples = sorted(mean(rng.choices(delta,k=len(delta))) for _ in range(5000))
    return {"difference":mean(delta),"bootstrap_95ci":[samples[125],samples[4874]],
            "fixes":delta.count(1),"regressions":delta.count(-1),"n":len(rows),
            "bootstrap_seed":20261019,"bootstrap_resamples":5000}


def sql_usage(sql, schema):
    """Scope-aware base table/column resolution, including CTEs and aliases."""
    from sqlglot import exp, parse_one
    from sqlglot.optimizer.qualify import qualify
    from sqlglot.optimizer.scope import traverse_scope
    from sqlglot.optimizer.normalize_identifiers import normalize_identifiers
    table_schema = {table.lower():{column.lower():value.get("column_type","TEXT")
                                 for column,value in data["columns"].items()}
                    for table,data in schema["tables"].items()}
    ast = normalize_identifiers(parse_one(sql,read="sqlite"),dialect="sqlite")
    ast = qualify(ast,dialect="sqlite",schema=table_schema,
                  quote_identifiers=False,validate_qualify_columns=True)
    tables, columns = set(), set()
    for scope in traverse_scope(ast):
        for _,source in scope.sources.items():
            if isinstance(source,exp.Table) and source.name.lower() in table_schema:
                tables.add(source.name.lower())
        for column in scope.columns:
            current = scope
            source = None
            while current is not None:
                source = current.sources.get(column.table)
                if source is not None:
                    break
                current = current.parent
            if isinstance(source,exp.Table):
                table = source.name.lower()
                if table in table_schema and column.name.lower() in table_schema[table]:
                    columns.add((table,column.name.lower()))
    return tables, columns, ast


def schema_metrics(selected, gold_sql, schema):
    predicted_tables = {table.lower() for table in selected}
    predicted_columns = {(table.lower(),column.lower()) for table,columns in selected.items() for column in columns}
    try:
        gold_tables, gold_columns, _ = sql_usage(gold_sql,schema)
    except Exception as exc:
        return {"parser_error":f"{type(exc).__name__}: {exc}"}
    result = {}
    for name,pred,gold in (("table",predicted_tables,gold_tables),("column",predicted_columns,gold_columns)):
        result[name+"_precision"] = len(pred & gold)/len(pred) if pred else float(not gold)
        result[name+"_recall"] = len(pred & gold)/len(gold) if gold else 1.0
        result["missing_"+name+"s"] = sorted(gold-pred)
        result["extra_"+name+"s"] = sorted(pred-gold)
        result[name+"_count"] = len(pred)
    result["full_schema_recall"] = gold_tables <= predicted_tables and gold_columns <= predicted_columns
    return result


def structural_flags(gold_sql, selected_sql, schema):
    from sqlglot import exp
    try:
        gt,gc,gold = sql_usage(gold_sql,schema)
        pt,pc,pred = sql_usage(selected_sql,schema)
    except Exception as exc:
        return ["ast_resolution_failed"], str(exc)
    flags = []
    if gt != pt:
        flags.append("table_set_difference")
    if gc != pc:
        flags.append("column_set_difference")
    for tag,kind in (("where",exp.Where),("having",exp.Having),("join",exp.Join),
                     ("group_by",exp.Group),("order_by",exp.Order),("limit",exp.Limit),
                     ("distinct",exp.Distinct),("aggregation",exp.AggFunc),
                     ("subquery",exp.Subquery),("set_operation",exp.SetOperation)):
        if len(list(gold.find_all(kind))) != len(list(pred.find_all(kind))):
            flags.append(tag+"_count_difference")
    if len(gold.selects) != len(pred.selects):
        flags.append("projection_width_difference")
    if Counter(node.key for node in gold.find_all(exp.AggFunc)) != Counter(node.key for node in pred.find_all(exp.AggFunc)):
        flags.append("aggregation_function_difference")
    compare_types = (exp.EQ,exp.NEQ,exp.GT,exp.GTE,exp.LT,exp.LTE,exp.Like,exp.In,exp.Between,exp.Is,exp.And,exp.Or)
    if Counter(node.key for node in gold.walk() if isinstance(node,compare_types)) != Counter(
            node.key for node in pred.walk() if isinstance(node,compare_types)):
        flags.append("comparison_or_boolean_operator_difference")
    if [bool(node.args.get("desc")) for node in gold.find_all(exp.Ordered)] != [
            bool(node.args.get("desc")) for node in pred.find_all(exp.Ordered)]:
        flags.append("order_direction_difference")
    if Counter(l.sql() for l in gold.find_all(exp.Literal)) != Counter(l.sql() for l in pred.find_all(exp.Literal)):
        flags.append("literal_difference")
    if not flags:
        flags = ["same_coarse_structure_semantic_difference"]
    return flags,None


def analyze(out, historical=None):
    manifest = json.loads((out/"manifest.json").read_text(encoding="utf-8"))
    provenance_path = out/"RESUME_PROVENANCE.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.exists() else None
    metrics = json.loads((out/"metrics.json").read_text(encoding="utf-8"))
    candidates = {row["question_id"]:row for row in read_jsonl(out/"candidates.jsonl")}
    inputs = {row["input"]["question_id"]:row["input"]
              for row in read_jsonl(out/"dataset.snapshot.data/items.jsonl")}
    linked_records = {row["input"]["question_id"]:row["pipeline_artifacts"]["schema_linking"]
                      for row in read_jsonl(out/"schema_linking.snapshot.data/items.jsonl")}
    expected = set(manifest["question_ids"])
    actual = [row["question_id"] for row in metrics["per_question"]]
    if set(actual) != expected or len(actual) != len(expected) or set(candidates) != expected:
        raise ValueError("Refusing final report: evaluation/candidate IDs do not exactly match the frozen run")
    retained_ids = set()
    if provenance and provenance.get("mixed_models"):
        retained_ids = set(provenance["source_retained_schema_linking_ids"])
        new_ids = set(provenance["new_schema_linking_ids"])
        if retained_ids & new_ids or retained_ids | new_ids != expected:
            raise ValueError("Mixed-model schema linking provenance does not partition the frozen run")
    rows = metrics["per_question"]
    errors = []
    for row in rows:
        qid = row["question_id"]
        if provenance and provenance.get("mixed_models"):
            row["schema_linking_model"] = provenance["source_model"] if qid in retained_ids else provenance["new_model"]
            row["generation_model"] = provenance["new_model"]
        source, candidate = inputs[qid],candidates[qid]
        row["raw_schema"] = schema_metrics(candidate.get("schema_linking") or {},source["gold_sql"],source["database_schema"])
        linked_schema = linked_records.get(qid,{}).get("database_schema_after_schema_linking")
        effective = ({table:list(data["columns"]) for table,data in linked_schema["tables"].items()}
                     if linked_schema is not None else candidate.get("schema_linking") or {})
        row["schema"] = schema_metrics(effective,source["gold_sql"],source["database_schema"])
        row["schema"]["effective_snapshot_available"] = linked_schema is not None
        if not row["gold_valid"]:
            category = "invalid_gold"
        elif row["selected_ex_multiset"]:
            category = "correct"
        elif row["coverage_multiset"]:
            category = "selection_error"
        elif not row["candidate_count"]:
            category = "no_candidates"
        elif not row["executable_count"]:
            category = "all_candidates_invalid"
        elif row["schema"].get("full_schema_recall") is False:
            category = "coverage_miss_with_schema_omission"
        else:
            category = "coverage_miss_after_linking"
        row["error_category"] = category
        if category != "correct":
            flags,parse_error = structural_flags(source["gold_sql"],candidate.get("selected_sql") or "",source["database_schema"])
            errors.append({**row,"question":source["question"],"evidence":source.get("evidence",""),
                           "gold_sql":source["gold_sql"],"selected_sql":candidate.get("selected_sql"),
                           "structural_flags":flags,"structural_parser_error":parse_error,
                           "correct_candidate_examples":[r["sql"] for r in row["candidates"] if r["ex_multiset"]][:3]})
    valid = [r for r in rows if r["gold_valid"]]
    summary = metrics["summary"]
    summary["counts"] = {key:sum(r[key] for r in valid) for key in
                         ("coverage_set","coverage_multiset","selected_ex_set","selected_ex_multiset",
                          "sc_ex_set","sc_ex_multiset")}
    summary["selected_ex_multiset_95ci"] = wilson(summary["counts"]["selected_ex_multiset"],len(valid))
    summary["selector_vs_sc"] = paired_difference(valid)
    summary["selection_gap"] = summary["counts"]["coverage_multiset"]-summary["counts"]["selected_ex_multiset"]
    before = [r for r in valid if "coverage_before_revision_multiset" in r]
    summary["revision_diagnostics"] = {
        "evaluable":len(before),
        "coverage_before_revision":sum(r["coverage_before_revision_multiset"] for r in before),
        "lost_correct_pool":sum(r["coverage_before_revision_multiset"] and not r["coverage_multiset"] for r in before),
        "gained_correct_pool":sum(not r["coverage_before_revision_multiset"] and r["coverage_multiset"] for r in before)}
    summary["gold_empty_questions"] = sum(r.get("gold_status") == "empty_result" for r in valid)
    summary["gold_all_null_questions"] = sum(r.get("gold_status") == "all_null_result" for r in valid)
    oracle = summary["counts"]["coverage_multiset"]
    summary["accuracy_given_covered"] = summary["counts"]["selected_ex_multiset"]/oracle if oracle else None
    summary["average_unique_candidates"] = mean(r["candidate_count"] for r in rows)
    generation_path = out/"sql_generation.snapshot.data/items.jsonl"
    config_path = out/"config.toml"
    if generation_path.exists() and config_path.exists():
        budgets = tomllib.loads(config_path.read_text(encoding="utf-8"))["sql_generation"]
        direct_n = budgets["dc_sampling_budget"]
        skeleton_n = budgets["skeleton_sampling_budget"]
        generated = read_jsonl(generation_path)
        generation_ids = [r["input"]["question_id"] for r in generated]
        if len(generation_ids) != len(expected) or set(generation_ids) != expected:
            raise ValueError("Generation snapshot does not match the frozen question set")
        candidate_groups = [r["pipeline_artifacts"]["sql_generation"]["sql_candidates"] for r in generated]
        if any(len(group) < direct_n + skeleton_n for group in candidate_groups):
            raise ValueError("Generation snapshot is shorter than frozen sampling budgets")
        direct_groups = [group[:direct_n] for group in candidate_groups]
        skeleton_groups = [group[direct_n:direct_n+skeleton_n] for group in candidate_groups]
        summary["candidate_diversity"] = {
            "direct_samples_per_question":direct_n,
            "skeleton_samples_per_question":skeleton_n,
            "raw_generated_sql":sum(map(len,candidate_groups)),
            "unique_generated_sql":sum(len(set(group)) for group in candidate_groups),
            "unique_after_revision_sql":sum(r["candidate_count"] for r in rows),
            "direct_one_unique_questions":sum(len(set(group)) == 1 for group in direct_groups),
            "skeleton_one_unique_questions":sum(len(set(group)) == 1 for group in skeleton_groups),
            "same_sql_set_across_strategies":sum(set(direct) == set(skeleton)
                                                   for direct,skeleton in zip(direct_groups,skeleton_groups)),
        }
    summary["error_categories"] = dict(Counter(r["error_category"] for r in rows))
    schema_valid = [r["schema"] for r in rows if not r["schema"].get("parser_error")]
    summary["schema_parser_evaluable"] = len(schema_valid)
    summary["schema_macro"] = {key:mean(r[key] for r in schema_valid) if schema_valid else None for key in
                               ("table_precision","table_recall","column_precision","column_recall",
                                "full_schema_recall","table_count","column_count")}
    raw_schema_valid = [r["raw_schema"] for r in rows if not r["raw_schema"].get("parser_error")]
    summary["raw_schema_macro"] = {key:mean(r[key] for r in raw_schema_valid) if raw_schema_valid else None
                                   for key in summary["schema_macro"]}
    by_db = []
    for db in sorted({r["db_id"] for r in valid}):
        subset = [r for r in valid if r["db_id"] == db]
        by_db.append({"db_id":db,"n":len(subset),**{key:sum(r[key] for r in subset) for key in
                      ("coverage_multiset","selected_ex_multiset","sc_ex_multiset")}})
    usage = read_jsonl(out/"api_usage.jsonl")
    api_by_stage = {}
    for stage in sorted({row["stage"] for row in usage}):
        subset = [r for r in usage if r["stage"] == stage]
        api_by_stage[stage] = {"successful_requests":len(subset),
                              "length_limited_responses":sum("length" in (r.get("finish_reasons") or []) for r in subset),
                              **{key:sum(r.get(key) or 0 for r in subset) for key in
                                 ("prompt_tokens","completion_tokens","total_tokens")}}
    school_pricing = (manifest.get("model") == "Qwen3.6-35B-A3B" and
                      manifest.get("base_url", "").rstrip("/") == "https://aigc-api.hkust-gz.edu.cn/api/v2")
    if school_pricing:
        for stage,row in api_by_stage.items():
            row["estimated_list_price_yuan"] = round(
                (row["prompt_tokens"]*0.90 + row["completion_tokens"]*5.40)/1_000_000, 2)
    stages = read_jsonl(out/"stages.jsonl")
    timing = {"recorded_stage_wall_seconds":sum(r["elapsed_seconds"] or 0 for r in stages),
              "attempts":stages,"api_by_stage":api_by_stage,
              "cost_note":("Estimated from published school Qwen3.6 prompt/completion list prices; not an account bill. "
                           "Only successful chat responses with returned usage are counted; embedding and failed calls excluded."
                           if school_pricing else
                           "Chat API success records include retries returning responses; embedding usage and provider-failed calls excluded. No monetary estimate.")}
    if school_pricing:
        timing["estimated_list_price_yuan"] = round(sum(
            (row["prompt_tokens"]*0.90 + row["completion_tokens"]*5.40)/1_000_000
            for row in api_by_stage.values()), 2)
    if stages:
        timing["pipeline_wall_span_seconds"] = (
            datetime.fromisoformat(stages[-1]["completed_at"])-datetime.fromisoformat(stages[0]["completed_at"])
        ).total_seconds()+stages[0]["elapsed_seconds"]
    question_times = [sum(stage.get("seconds") or 0 for stage in r.get("stage_metrics",{}).values()) for r in candidates.values()]
    timing["per_question_stage_time_sum_seconds"] = {
        "mean":mean(question_times),"median":median(question_times),
        "p95":sorted(question_times)[math.ceil(0.95*len(question_times))-1],
        "note":"Sum of measured item work across stages, including internal worker waits; excludes pre-item outer queues and indexing; not end-user wall latency."}
    if provenance is not None:
        timing["resume_provenance"] = provenance
    comparisons = []
    if historical and historical.exists():
        with historical.open(encoding="utf-8-sig",newline="") as handle:
            old = {int(r["question_id"]):r for r in csv.DictReader(handle)}
        def boolean(value):
            return {"True":True,"False":False,"1":True,"0":False}.get(value)
        for name,field in (("Historical MCTS + SC (different model/budget)","llm_selected_ex"),
                           ("Historical Jev constrained","jev_selected_ex"),
                           ("Historical MCTS + Jev rerank (different model/budget)","shared_jev_rerank_selected_ex")):
            paired = [r for r in valid if r["question_id"] in old and boolean(old[r["question_id"]].get(field)) is not None]
            comparisons.append({"method":name,"common_n":len(paired),
                                "historical_ex":sum(boolean(old[r["question_id"]][field]) for r in paired),
                                "deepeye_ex":sum(r["selected_ex_multiset"] for r in paired),
                                "controlled":False})
    report = {"manifest":manifest,"summary":summary,"by_database":by_db,"timing_and_usage":timing,
              "historical_descriptive_only":comparisons,"per_question":rows,
              "limitations":["Frozen BIRD train subset, not full BIRD dev/test; not a held-out generalization claim.",
                             "Schema P/R is relative to one gold SQL, not every semantically valid alternative; COUNT(*) and alternate joins can change the reference column set.",
                             "Set/multiset row comparison ignores ordering and is not the official BIRD evaluator.",
                             "EX on one database can accept accidental/spurious matches, especially empty/all-NULL outputs.",
                             "Structural flags and schema omissions are diagnostics, not causal proofs of semantic error.",
                             "DeepEye does not emit calibrated correctness probabilities; SC vote share is not calibrated confidence.",
                             "Full profile with target-train few-shot disabled; not an exact original-model paper reproduction."]}
    if provenance and provenance.get("mixed_models"):
        report["limitations"].append("Mixed-model continuation: schema linking combines source and new model outputs. This is not a pure new-model result or a controlled model comparison.")
    write_json(out/"analysis.json",report)
    (out/"error_analysis.jsonl").write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in errors),encoding="utf-8")
    write_markdown(out,report,errors)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return report


def write_markdown(out,report,errors):
    summary = report["summary"]
    counts = summary["counts"]
    n = summary["evaluable"]
    manifest = report["manifest"]
    provenance = report["timing_and_usage"].get("resume_provenance")
    def fraction(value):
        return f"{value}/{n} ({value/n:.2%})" if n else "N/A"
    lines = ["# DeepEye-SQL：冻结 300 题扩展实验", "",
             f"模型：`{manifest['model']}`；BIRD `{manifest['dataset_split']}`；完成 {summary['questions']} 题，金标可执行 {n} 题。", "",
             "## 实验口径", "",
             "沿用原实验冻结题号；不微调模型；训练集动态 few-shot 关闭，避免检索目标题金标。向量检索使用独立 embedding 模型。",
             "主指标使用不区分行顺序、保留重复行的 multiset EX；另报与上游一致的 set EX。它不是官方 BIRD 测评，也不等于 SQL 语义等价。", "",
             "## 结果", "", "| 方法 | Candidate coverage | 最终 EX（multiset） |", "|---|---:|---:|",
             f"| DeepEye 候选池 + self-consistency | {fraction(counts['coverage_multiset'])} | {fraction(counts['sc_ex_multiset'])} |",
             f"| DeepEye 完整选择流程 | {fraction(counts['coverage_multiset'])} | {fraction(counts['selected_ex_multiset'])} |", "",
             f"Set 口径：coverage {fraction(counts['coverage_set'])}，最终 EX {fraction(counts['selected_ex_set'])}。",
             f"候选 SQL 可执行率 {summary['executable_rate']:.2%}；每题平均去重候选数 {summary['average_unique_candidates']:.2f}。空结果计为可执行。" if summary['executable_rate'] is not None else "未生成候选，SQL 可执行率无定义。",
             f"候选覆盖到最终选择之间仍损失 {summary['selection_gap']} 题。",
             "SC 对所有可执行结果（包括空结果）按重复候选投票；同票时取最早出现者。与上游偏好非空结果、按耗时打破同票的逻辑不同，明确作为独立、确定性的基线。", ""]
    if provenance and provenance.get("mixed_models"):
        lines[4:4] = [f"混合模型续跑：Schema Linking 的 {len(provenance['source_retained_schema_linking_ids'])} 题来自 `{provenance['source_model']}`，"
                      f"{len(provenance['new_schema_linking_ids'])} 题来自 `{provenance['new_model']}`；SQL 生成、修订与选择使用 `{provenance['new_model']}`。",
                      "这组数据不能解释为纯新模型实验或两模型的受控对照。", ""]
    paired = summary["selector_vs_sc"]
    if paired:
        low,high = paired["bootstrap_95ci"]
        lines += [f"相对同池 SC：修复 {paired['fixes']} 题，退化 {paired['regressions']} 题，净变化 {paired['difference']:+.2%}；配对 bootstrap 95% CI [{low:+.2%}, {high:+.2%}]。",
                  "区间只反映这批题上的重采样不确定性，不包含候选生成的随机性；CI 跨零时不能声称显著提升。", ""]
    diversity = summary.get("candidate_diversity")
    if diversity:
        lines += ["## 候选多样性", "",
                  f"生成阶段共返回 {diversity['raw_generated_sql']} 条 SQL，逐题去重合计 {diversity['unique_generated_sql']} 条；修订后逐题去重合计 {diversity['unique_after_revision_sql']} 条。",
                  f"每题 direct 采样 {diversity['direct_samples_per_question']} 次，有 {diversity['direct_one_unique_questions']}/{summary['questions']} 题的 direct 输出完全相同；"
                  f"skeleton 采样 {diversity['skeleton_samples_per_question']} 次，有 {diversity['skeleton_one_unique_questions']}/{summary['questions']} 题完全相同。"
                  f"两种策略的 SQL 集合相同的题有 {diversity['same_sql_set_across_strategies']} 道。",
                  "这些数字证明当前重复采样几乎没有带来候选多样性；不能仅凭输出推断是模型参数、代理或提示词哪一环导致。", ""]
    lines += ["## Schema Linking", "",f"使用 SQLGlot scope-aware 金标解析，成功评估 {summary['schema_parser_evaluable']} 题；解析失败不计入均值。", "",
              "主表为主外键补全后的 schema（来自 linking 阶段快照）；同时报告原始 linker 合并结果。有效 schema 仍可能在构建提示词时进一步压缩。", "",
              "| 指标（逐题宏平均） | 补全后 schema | 原始 linker 合并结果 |", "|---|---:|---:|"]
    for key,value in summary["schema_macro"].items():
        raw = summary["raw_schema_macro"][key]
        lines.append(f"| {key} | {value:.4f} | {raw:.4f} |" if value is not None and raw is not None else f"| {key} | N/A | N/A |")
    lines += ["", "主外键补全及冗余列也计入预测集合。与历史实验的不同解析器结果不可直接解释为 linker 的净提升。", "",
              "## 按数据库", "", "| 数据库 | 题数 | Coverage | DeepEye EX | SC EX |", "|---|---:|---:|---:|---:|"]
    for row in report["by_database"]:
        lines.append(f"| {row['db_id']} | {row['n']} | {row['coverage_multiset']} | {row['selected_ex_multiset']} | {row['sc_ex_multiset']} |")
    lines += ["", "## 错误分析", "", "以下分类互斥；schema omission 是伴随现象，不代表已证明因果。逐题 SQL、缺失表列、正确候选示例见 `error_analysis.jsonl`。", "",
              "| 类别 | 题数 |", "|---|---:|"]
    for key,count in summary["error_categories"].items():
        lines.append(f"| {key} | {count} |")
    lines += ["", f"{summary['questions']-counts['coverage_multiset']} 题候选池没有执行匹配的 SQL；"
              f"其中 {summary['error_categories'].get('coverage_miss_after_linking',0)} 题在当前金标解析口径下已有完整 schema，却仍未生成正确候选。"
              f"另外 {summary['selection_gap']} 题有正确候选但最终选择错误；其中 "
              f"{sum(r['sc_ex_multiset'] for r in report['per_question'] if r['error_category']=='selection_error')} 题被同池 SC 正确选中。", ""]
    rev = summary["revision_diagnostics"]
    lines += ["", f"Revision 前候选覆盖 {rev['coverage_before_revision']}/{rev['evaluable']}；修订后丢失正确候选池 {rev['lost_correct_pool']} 题，新增正确候选池 {rev['gained_correct_pool']} 题。",
              f"Gold 空结果 {summary['gold_empty_questions']} 题、全 NULL 结果 {summary['gold_all_null_questions']} 题。这些题的执行匹配容易出现偶合，不能直接视为语义正确；上游部分修订器和选择器偏好非空结果，相关失败需结合逐题 SQL 检查。"]
    flags = Counter(flag for error in errors for flag in error["structural_flags"])
    lines += ["", "错误 SQL 与金标的结构差异（可重叠，仅用于定位，不是人工确认的语义根因）：", ""]
    lines += [f"- `{key}`：{count} 题。" for key,count in flags.most_common()]
    for error in errors[:8]:
        lines += ["",f"### Q{error['question_id']} · {error['db_id']} · {error['error_category']}","",
                  error["question"],"","```sql", "-- Gold",error["gold_sql"],"-- Selected",error["selected_sql"] or "-- missing","```",
                  f"结构提示：{', '.join(error['structural_flags'])}；gold 行数：{error['gold_row_count']}。"]
    school_pricing = "estimated_list_price_yuan" in report["timing_and_usage"]
    lines += ["", "## 延迟与用量", "",f"已记录阶段累计 wall time：{report['timing_and_usage']['recorded_stage_wall_seconds']/60:.2f} 分钟（包含重跑尝试，不含离线评测）。", "",
              "| 阶段 | 成功 Chat 请求数 | 输入 tokens | 输出 tokens |" + (" 标价估算（元） |" if school_pricing else ""),
              "|---|---:|---:|---:|" + ("---:|" if school_pricing else "")]
    for stage,row in report["timing_and_usage"]["api_by_stage"].items():
        lines.append(f"| {stage} | {row['successful_requests']} | {row['prompt_tokens']} | {row['completion_tokens']} |"
                     + (f" {row['estimated_list_price_yuan']:.2f} |" if school_pricing else ""))
    qtime = report["timing_and_usage"]["per_question_stage_time_sum_seconds"]
    lines += ["", f"逐题各阶段工作时间之和：mean {qtime['mean']:.2f}s，median {qtime['median']:.2f}s，P95 {qtime['p95']:.2f}s；含内部 worker 等待，不含进入阶段前的外部排队和索引，不能当作用户端到端响应延迟。",
              (f"按[港科广公开价格](https://docs.hpc.hkust-gz.edu.cn/docs/aigc/api-model-list/)估算本次新增成功 Chat 请求合计 {report['timing_and_usage']['estimated_list_price_yuan']:.2f} 元；不是实付账单。"
               if school_pricing else "未核实单价时不推算费用。")
              + "不含旧 Qwen3.5 用量、embedding、无响应失败请求；阶段墙钟时间不含离线评测。", "",
              "## 历史结果（描述性对照，非受控消融）", "", "历史 MCTS 使用不同模型与更低采样预算；不能将差值全部归因于 DeepEye。", "",
              "| 历史方法 | 共同题数 | 历史正确数 | 本次 DeepEye 正确数 |", "|---|---:|---:|---:|"]
    for row in report["historical_descriptive_only"]:
        lines.append(f"| {row['method']} | {row['common_n']} | {row['historical_ex']} | {row['deepeye_ex']} |")
    lines += ["", "## 局限及后续验证", ""]
    lines += [f"- {note}" for note in report["limitations"]]
    lines += ["", "下一步受控对照须统一题集、Qwen 版本、候选数量/API token 预算及 EX 口径。若训练 selector，这 300 题不再用于训练或调参；另分 BIRD train 构造候选。", ""]
    (out/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name",default="bird-train-300-qwen35-p32")
    parser.add_argument("--historical",type=Path,default=ROOT/"results/llm_vs_jev_300/per_question.csv")
    args = parser.parse_args()
    analyze(run_dir(args.run_name),args.historical)


if __name__ == "__main__":
    main()
