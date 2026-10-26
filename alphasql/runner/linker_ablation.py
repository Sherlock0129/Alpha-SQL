"""Frozen 287-question standalone linking and mandatory-linker MCTS ablation.

All live calls are resumable. Gold SQL is used exclusively by evaluate().
No Jev SQL intent decisions, compiler or reranker are used by either arm.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import re
import time

from alphasql.algorithm.mcts import mcts_action as actions
from alphasql.algorithm.mcts.mcts import MCTSSolver
from alphasql.algorithm.mcts.mcts_node import get_valid_action_space_for_node
from alphasql.algorithm.mcts.reward import MajorityVoteRewardModel
from alphasql.algorithm.schema_linking.jev_linker import JevSchemaLinker
from alphasql.algorithm.selection.schema_signal import extract_sql_usage, result_signature
from alphasql.algorithm.selection.utils import get_subset_schema_dict
from alphasql.database.schema import DatabaseSchema
from alphasql.database.sql_execution import execute_sql_with_timeout, SQLExecutionResultType
from alphasql.database.utils import build_table_ddl_statement
from alphasql.jev.client import JevClient
from alphasql.jev.models import NoulQuestion, NoulResult
from alphasql.jev.runtime import JevSettings
from alphasql.llm_call.openai_llm import call_openai, reset_request_counter
from alphasql.llm_call.prompt_factory import get_prompt
from alphasql.llm_call.runtime import configure_environment
from alphasql.runner.llm_jev_phase1 import _candidate_sqls, _precision_recall, _self_consistency


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def safe_error(exc):
    message = f'{type(exc).__name__}: {exc}'
    for key in ('OPENAI_API_KEY', 'DASHSCOPE_API_KEY', 'TYPESAFE_API_KEY'):
        secret = os.getenv(key)
        if secret:
            message = message.replace(secret, '[REDACTED]')
    return message


def schema_for(task):
    return DatabaseSchema(db_id=task.db_id, tables=copy.deepcopy(task.table_schema_dict))


def shared_input(task):
    """Exact semantic payload for both linkers; no gold, SQL or path thoughts."""
    return {
        'question': task.question,
        'evidence': task.evidence or '',
        'schema': JevSchemaLinker._schema_state(schema_for(task)),
    }


def canonical_selection(selection, schema):
    if not isinstance(selection, dict) or not selection:
        raise ValueError('Linker returned an empty or non-object selection')
    table_names = {(table.table_name or key).lower(): key for key, table in schema.tables.items()}
    result = {}
    for table, columns in selection.items():
        if not isinstance(table, str) or table.lower() not in table_names:
            raise ValueError(f'Unknown table in linker result: {table!r}')
        if not isinstance(columns, list) or not all(isinstance(c, str) for c in columns):
            raise ValueError('Each selected table must map to a list of column names')
        table_key = table_names[table.lower()]
        column_names = {(c.original_column_name or key).lower(): key
                        for key, c in schema.tables[table_key].columns.items()}
        checked = []
        for column in columns:
            if column.lower() not in column_names:
                raise ValueError(f'Unknown column: {table}.{column}')
            name = column_names[column.lower()]
            if name not in checked:
                checked.append(name)
        result[table_key] = checked
    return result


def link_llm(state, schema, model, max_tokens):
    prompt = get_prompt('schema_selection', {
        'QUESTION': state['question'], 'HINT': state['evidence'],
        'SCHEMA_CONTEXT': json.dumps(state['schema'], ensure_ascii=False, indent=2),
    })
    response = call_openai(prompt, model=model, n=1, temperature=0, top_p=1,
                           max_tokens=max_tokens, n_strategy='multiple')[0]
    match = re.search(r'```json\s*(.*?)```', response, flags=re.S | re.I)
    if not match:
        raise ValueError('Original LLM schema prompt returned no fenced JSON')
    selection = canonical_selection(json.loads(match.group(1)), schema)
    return {'raw_selection': selection, 'response': response, 'probabilities': None}


def link_jev(state, schema, settings, client=None):
    questions, refs = {}, {}
    for ti, (tk, table) in enumerate(schema.tables.items()):
        table_name = table.table_name or tk
        key = f'table_{ti}'
        refs[key] = (tk, None)
        questions[key] = NoulQuestion(
            f"Is table '{table_name}' semantically necessary to answer the user question?")
        for ci, (ck, column) in enumerate(table.columns.items()):
            key = f'column_{ti}_{ci}'
            refs[key] = (tk, ck)
            questions[key] = NoulQuestion(
                f"Is column '{table_name}.{column.original_column_name or ck}' needed for projection, filtering, joining, "
                'grouping, ordering, or date logic in the answer?')
    response = (client or JevClient(settings)).ask(state, questions)
    probabilities, selection = {}, {}
    for key, (tk, ck) in refs.items():
        answer = response.require(key, NoulResult)
        probabilities[key] = {'table': tk, 'column': ck, **answer.to_dict()}
        if ck is None and answer.noul >= settings.schema_threshold:
            selection[tk] = []
    for key, (tk, ck) in refs.items():
        if ck is not None and tk in selection and probabilities[key]['noul'] >= settings.schema_threshold:
            selection[tk].append(ck)
    # No Jev-specific join-path expansion or forced argmax. Same empty-result policy.
    selection = canonical_selection(selection, schema)
    return {'raw_selection': selection, 'probabilities': probabilities,
            'usage': {'input_tokens': response.usage.input_tokens,
                      'output_tokens': response.usage.output_tokens}}


def completed_selection(raw, schema):
    selected = get_subset_schema_dict(schema.tables, raw)
    return {name: list(table.columns) for name, table in selected.items()}


def ddl_for(tables):
    return '\n'.join(build_table_ddl_statement(
        tables[name].to_dict(), add_value_description=True, add_column_description=True,
        add_value_examples=True, add_expanded_column_name=True) for name in tables)


class FixedLinkerMCTS(MCTSSolver):
    """Original search actions with one frozen upfront schema in every branch."""
    def __init__(self, *args, selection, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixed_tables = get_subset_schema_dict(self.task.table_schema_dict, selection)
        if not self.fixed_tables:
            raise ValueError('Cannot run generation without a successful linker selection')
        self.fixed_context = ddl_for(self.fixed_tables)

    def expand(self, node):
        assert not node.children
        node.selected_schema_dict = self.fixed_tables
        node.selected_schema_context = self.fixed_context
        valid_actions = [a for a in get_valid_action_space_for_node(node)
                         if not isinstance(a, actions.SchemaSelectionAction)]
        if node.depth >= self.max_depth - 1:
            if node.node_type in (actions.MCTSNodeType.SQL_GENERATION, actions.MCTSNodeType.SQL_REVISION):
                valid_actions = [actions.EndAction()]
            else:
                valid_actions = [actions.SQLGenerationAction()]
        for action in valid_actions:
            node.children.extend(action.create_children_nodes(node, self.llm_kwargs))
        random.shuffle(node.children)


def worker(payload):
    task, arm, stage, cfg = payload
    configure_environment()
    reset_request_counter()
    task = task.model_copy(update={'sql': None})
    out = Path(cfg['output_dir'])
    path = out / 'links' / arm / f'{task.question_id}.json'
    candidate_dir = out / 'candidates' / arm
    status_path = out / 'generation' / arm / f'{task.question_id}.json'
    started = time.monotonic()
    usage = out / 'usage' / f'{stage}_{arm}_{task.question_id}.jsonl'
    usage.parent.mkdir(parents=True, exist_ok=True)
    os.environ['ALPHASQL_USAGE_PATH'] = str(usage)
    try:
        if stage == 'link':
            if path.exists():
                cached = json.loads(path.read_text(encoding='utf-8'))
                if cached['input_sha256'] != fingerprint(shared_input(task)):
                    raise ValueError('Cached linker input differs from frozen task')
                return {'question_id': task.question_id, 'arm': arm, 'cached': True, 'ok': True}
            state, schema = shared_input(task), schema_for(task)
            record = (link_llm(state, schema, cfg['model'], cfg['link_max_tokens']) if arm == 'llm'
                      else link_jev(state, schema, JevSettings.from_environment()))
            record.update({'question_id': task.question_id, 'db_id': task.db_id,
                           'input_sha256': fingerprint(state), 'arm': arm,
                           'completed_selection': completed_selection(record['raw_selection'], schema),
                           'elapsed_seconds': time.monotonic() - started})
            save_json(path, record)
        else:
            if status_path.exists() and (candidate_dir / f'{task.question_id}.pkl').exists():
                return {'question_id': task.question_id, 'arm': arm, 'cached': True, 'ok': True}
            record = json.loads(path.read_text(encoding='utf-8'))
            if record['input_sha256'] != fingerprint(shared_input(task)):
                raise ValueError('Cached linker input differs from frozen task')
            random.seed(cfg['seed'] + int(task.question_id))
            actions.SQL_GENERATION_LLM_KWARGS_N = cfg['samples']
            actions.SQL_REVISION_LLM_KWARGS_N = cfg['samples']
            actions.SQL_VALIDATION_MAX_TRIES = cfg['validation_tries']
            candidate_dir.mkdir(parents=True, exist_ok=True)
            logs = out / 'logs' / arm
            logs.mkdir(parents=True, exist_ok=True)
            with (logs / f'{task.question_id}.log').open('w', encoding='utf-8') as log:
                with redirect_stdout(log), redirect_stderr(log):
                    solver = FixedLinkerMCTS(
                        db_root_dir=cfg['db_root'], task=task,
                        max_rollout_steps=cfg['rollouts'], max_depth=cfg['max_depth'],
                        exploration_constant=1.414, save_root_dir=str(candidate_dir),
                        llm_kwargs={'model': cfg['model'], 'n': 1, 'top_p': 0.8,
                                    'temperature': 0.8, 'max_tokens': cfg['generation_max_tokens'],
                                    'n_strategy': 'multiple'},
                        reward_model=MajorityVoteRewardModel(None), selection=record['raw_selection'])
                    solver.solve()
            save_json(status_path, {'question_id': task.question_id, 'arm': arm,
                                    'input_sha256': record['input_sha256'],
                                    'elapsed_seconds': time.monotonic() - started,
                                    'selector': 'execution_self_consistency',
                                    'jev_reranker_used': False})
        return {'question_id': task.question_id, 'arm': arm, 'ok': True,
                'seconds': round(time.monotonic() - started, 2)}
    except Exception as exc:
        failure = {'question_id': task.question_id, 'arm': arm, 'stage': stage,
                   'ok': False, 'error': safe_error(exc)}
        save_json(out / 'failures' / stage / arm / f'{task.question_id}.json', failure)
        return failure


def selection_sets(selection):
    return ({t.lower() for t in selection},
            {(t.lower(), c.lower()) for t, cols in selection.items() for c in cols})


def schema_metrics(predicted, gold):
    pt, pc = selection_sets(predicted)
    gt, gc = gold
    tp, tr = _precision_recall(pt, gt)
    cp, cr = _precision_recall(pc, gc)
    return {'table_precision': tp, 'table_recall': tr,
            'table_f1': 2*tp*tr/(tp+tr) if tp+tr else 0,
            'column_precision': cp, 'column_recall': cr,
            'column_f1': 2*cp*cr/(cp+cr) if cp+cr else 0,
            'full_schema_recall': int(gt <= pt and gc <= pc),
            'exact_schema_match': int(gt == pt and gc == pc),
            'table_count': len(pt), 'column_count': len(pc),
            'extra_tables': len(pt-gt), 'extra_columns': len(pc-gc),
            'missing_tables': len(gt-pt), 'missing_columns': len(gc-pc)}


def execution_metrics(sqls, db, gold_sig, timeout):
    results = [execute_sql_with_timeout(str(db), sql, timeout) for sql in sqls]
    signatures = [result_signature(r.result) if r.result_type is SQLExecutionResultType.SUCCESS
                  else None for r in results]
    winner = _self_consistency(signatures, [len(r.result or []) for r in results])
    return {'candidate_count': len(sqls), 'executable_count': sum(s is not None for s in signatures),
            'coverage': int(gold_sig in signatures),
            'ex': int(winner is not None and signatures[winner] == gold_sig),
            'selected_sql': sqls[winner] if winner is not None else None,
            'execution_errors': [r.error_message for r in results if r.result_type is not SQLExecutionResultType.SUCCESS]}


def paired_stats(pairs):
    if not pairs:
        return {'n': 0}
    fixes = sum(not a and b for a,b in pairs)
    regressions = sum(a and not b for a,b in pairs)
    diffs = [b-a for a,b in pairs]
    rng = random.Random(42)
    bs = sorted(sum(rng.choice(diffs) for _ in diffs)/len(diffs) for _ in range(10000))
    from scipy.stats import binomtest
    return {'n': len(pairs), 'llm_correct': sum(a for a,b in pairs),
            'jev_correct': sum(b for a,b in pairs), 'fixes': fixes, 'regressions': regressions,
            'delta': sum(diffs)/len(diffs), 'ci95': [bs[250], bs[9750]],
            'mcnemar_exact_p': float(binomtest(fixes, fixes+regressions).pvalue) if fixes+regressions else 1.0}


def efficiency_summary(out, stage, arm):
    records=[]
    for path in (out/'usage').glob(f'{stage}_{arm}_*.jsonl'):
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                records.append(json.loads(line))
    if stage=='link' and arm=='jev':
        links=[json.loads(p.read_text(encoding='utf-8')) for p in (out/'links'/'jev').glob('*.json')]
        return {'successful_calls':len(links),
                'input_tokens':sum(r.get('usage',{}).get('input_tokens',0) for r in links),
                'output_tokens':sum(r.get('usage',{}).get('output_tokens',0) for r in links),
                'aggregate_seconds':sum(r['elapsed_seconds'] for r in links)}
    return {'successful_calls':len(records),
            'input_tokens':sum(r.get('prompt_tokens') or 0 for r in records),
            'output_tokens':sum(r.get('completion_tokens') or 0 for r in records),
            'aggregate_seconds':sum(r.get('seconds') or 0 for r in records)}


def evaluate(tasks, cfg):
    out = Path(cfg['output_dir'])
    link_rows, execution_rows, errors = [], [], []
    link_records={}
    key_misses={arm:Counter() for arm in ('llm','jev')}
    for task in tasks:
        schema = schema_for(task)
        usage = extract_sql_usage(task.sql, schema)
        gold = ({t.lower() for t in usage.tables}, {(t.lower(),c.lower()) for t,c in usage.columns})
        db = Path(cfg['db_root']) / task.db_id / f'{task.db_id}.sqlite'
        gr = execute_sql_with_timeout(str(db), task.sql, cfg['execution_timeout'])
        if gr.result_type is not SQLExecutionResultType.SUCCESS:
            raise ValueError(f'Frozen gold no longer executable: {task.question_id}')
        gs = result_signature(gr.result)
        for arm in ('llm', 'jev'):
            p = out/'links'/arm/f'{task.question_id}.json'
            if p.exists():
                record = json.loads(p.read_text(encoding='utf-8'))
                link_records[(task.question_id,arm)] = record
                if record['input_sha256'] != fingerprint(shared_input(task)):
                    raise ValueError(f'Input hash mismatch: {arm}/{task.question_id}')
                for kind in ('raw', 'completed'):
                    row = {'question_id': task.question_id, 'db_id': task.db_id, 'arm': arm, 'kind': kind,
                           **schema_metrics(record[f'{kind}_selection'], gold)}
                    link_rows.append(row)
                raw_columns=selection_sets(record['raw_selection'])[1]
                completed_columns=selection_sets(record['completed_selection'])[1]
                columns={(t.lower(),c.lower()):v for t,table in schema.tables.items()
                         for c,v in table.columns.items()}
                for ref in gold[1]-raw_columns:
                    column=columns.get(ref)
                    is_key=bool(column and (column.primary_key or column.foreign_keys or column.referenced_by))
                    key_misses[arm]['raw_missing_key' if is_key else 'raw_missing_nonkey']+=1
                key_misses[arm]['completed_missing_columns']+=len(gold[1]-completed_columns)
                key_misses[arm]['gold_columns_recovered_by_completion']+=len((completed_columns-raw_columns)&gold[1])
            candidate = out/'candidates'/arm/f'{task.question_id}.pkl'
            status = out/'generation'/arm/f'{task.question_id}.json'
            if not (candidate.exists() and status.exists()):
                continue
            sqls, _ = _candidate_sqls(candidate)
            row = {'question_id': task.question_id, 'db_id': task.db_id, 'arm': arm,
                   **execution_metrics(sqls, db, gs, cfg['execution_timeout'])}
            execution_rows.append(row)
            if not row['ex']:
                link = next((r for r in link_rows if r['question_id']==task.question_id
                             and r['arm']==arm and r['kind']=='completed'), {})
                category = ('selection_error' if row['coverage'] else 'schema_recall_miss'
                            if not link.get('full_schema_recall') else 'generation_coverage_miss')
                errors.append({**row, 'category': category, 'question': task.question,
                               'evidence': task.evidence, 'gold_sql': task.sql})
    metrics = {'frozen_questions': len(tasks), 'selector': 'execution_self_consistency',
               'jev_reranker_used': False, 'linking': {}, 'generation': {}}
    for arm in ('llm','jev'):
        for kind in ('raw','completed'):
            rows = [r for r in link_rows if r['arm']==arm and r['kind']==kind]
            metrics['linking'][f'{arm}_{kind}'] = {'completed': len(rows), **{
                key: sum(r[key] for r in rows)/len(rows) for key in rows[0]
                if key not in ('question_id','db_id','arm','kind')} } if rows else {'completed': 0}
        rows = [r for r in execution_rows if r['arm']==arm]
        attempted_ids={t.question_id for t in tasks if (out/'generation'/arm/f'{t.question_id}.json').exists()
                       or (out/'failures'/'generate'/arm/f'{t.question_id}.json').exists()}
        metrics['generation'][arm] = {'completed': len(rows),
            'attempted':len(attempted_ids),'failed':len(attempted_ids)-len(rows),
            'not_attempted':len(tasks)-len(attempted_ids),
            'correct': sum(r['ex'] for r in rows), 'covered': sum(r['coverage'] for r in rows),
            'ex_on_completed': sum(r['ex'] for r in rows)/len(rows) if rows else None,
            'ex_lower_bound_frozen': sum(r['ex'] for r in rows)/len(tasks),
            'coverage_on_completed': sum(r['coverage'] for r in rows)/len(rows) if rows else None,
            'executable_rate': sum(r['executable_count'] for r in rows)/max(1,sum(r['candidate_count'] for r in rows))}
    byid = {(r['question_id'],r['arm']):r for r in execution_rows}
    pairs = [(byid[(t.question_id,'llm')]['ex'],byid[(t.question_id,'jev')]['ex']) for t in tasks
             if (t.question_id,'llm') in byid and (t.question_id,'jev') in byid]
    metrics['paired_ex'] = paired_stats(pairs)
    attempted_pairs=[]
    for task in tasks:
        attempted={arm:((out/'generation'/arm/f'{task.question_id}.json').exists() or
                        (out/'failures'/'generate'/arm/f'{task.question_id}.json').exists())
                   for arm in ('llm','jev')}
        if all(attempted.values()):
            attempted_pairs.append(tuple(byid.get((task.question_id,arm),{}).get('ex',0)
                                         for arm in ('llm','jev')))
    metrics['paired_pipeline_ex_including_failures'] = paired_stats(attempted_pairs)
    metrics['key_miss_analysis']={arm:dict(counts) for arm,counts in key_misses.items()}
    links_byid={(r['question_id'],r['arm'],r['kind']):r for r in link_rows}
    metrics['paired_full_schema_recall']={kind:paired_stats([
        (links_byid[(t.question_id,'llm',kind)]['full_schema_recall'],
         links_byid[(t.question_id,'jev',kind)]['full_schema_recall']) for t in tasks
        if (t.question_id,'llm',kind) in links_byid and (t.question_id,'jev',kind) in links_byid])
        for kind in ('raw','completed')}
    paired_details=[]
    for task in tasks:
        a=byid.get((task.question_id,'llm')); b=byid.get((task.question_id,'jev'))
        if a is None or b is None:
            continue
        la=next(r for r in link_rows if r['question_id']==task.question_id and r['arm']=='llm' and r['kind']=='completed')
        lb=next(r for r in link_rows if r['question_id']==task.question_id and r['arm']=='jev' and r['kind']=='completed')
        link_a=link_records[(task.question_id,'llm')]
        link_b=link_records[(task.question_id,'jev')]
        same_schema=selection_sets(link_a['completed_selection'])==selection_sets(link_b['completed_selection'])
        context_a=ddl_for(get_subset_schema_dict(task.table_schema_dict,link_a['raw_selection']))
        context_b=ddl_for(get_subset_schema_dict(task.table_schema_dict,link_b['raw_selection']))
        paired_details.append({
            'question_id':task.question_id,'db_id':task.db_id,'question':task.question,'evidence':task.evidence,
            'outcome':('both_correct' if a['ex'] and b['ex'] else 'jev_fix' if b['ex'] else
                       'jev_regression' if a['ex'] else 'both_wrong'),
            'llm_ex':a['ex'],'jev_ex':b['ex'],'llm_coverage':a['coverage'],'jev_coverage':b['coverage'],
            'llm_full_schema':la['full_schema_recall'],'jev_full_schema':lb['full_schema_recall'],
            'llm_column_recall':la['column_recall'],'jev_column_recall':lb['column_recall'],
            'identical_schema_elements':same_schema,'identical_generation_context':context_a==context_b,
            'llm_selected_sql':a['selected_sql'],'jev_selected_sql':b['selected_sql'],'gold_sql':task.sql})
    if paired_details:
        with (out/'paired_cases.csv').open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(paired_details[0]));writer.writeheader();writer.writerows(paired_details)
    metrics['schema_to_ex']={}
    for label,predicate in {
        'jev_schema_complete_llm_incomplete':lambda r:r['jev_full_schema'] and not r['llm_full_schema'],
        'llm_schema_complete_jev_incomplete':lambda r:r['llm_full_schema'] and not r['jev_full_schema'],
        'both_schema_complete':lambda r:r['llm_full_schema'] and r['jev_full_schema'],
        'both_schema_incomplete':lambda r:not r['llm_full_schema'] and not r['jev_full_schema'],
    }.items():
        subset=[r for r in paired_details if predicate(r)]
        metrics['schema_to_ex'][label]={'n':len(subset),'llm_correct':sum(r['llm_ex'] for r in subset),
                                      'jev_correct':sum(r['jev_ex'] for r in subset)}
    metrics['paired_context_strata']={label:paired_stats([(r['llm_ex'],r['jev_ex'])
        for r in paired_details if predicate(r)]) for label,predicate in {
        'identical_generation_context':lambda r:r['identical_generation_context'],
        'same_schema_different_order':lambda r:r['identical_schema_elements'] and not r['identical_generation_context'],
        'different_schema_elements':lambda r:not r['identical_schema_elements'],
    }.items()}
    metrics['paired_by_database'] = {}
    for database in sorted({t.db_id for t in tasks}):
        local=[(byid[(t.question_id,'llm')]['ex'],byid[(t.question_id,'jev')]['ex'])
               for t in tasks if t.db_id==database and
               (t.question_id,'llm') in byid and (t.question_id,'jev') in byid]
        metrics['paired_by_database'][database] = {
            'n':len(local), 'llm_correct':sum(a for a,b in local),
            'jev_correct':sum(b for a,b in local),
            'fixes':sum(not a and b for a,b in local),
            'regressions':sum(a and not b for a,b in local)}
    metrics['efficiency']={f'{stage}_{arm}':efficiency_summary(out,stage,arm)
                           for stage in ('link','generate') for arm in ('llm','jev')}
    metrics['missing']={stage:{arm:[t.question_id for t in tasks if not (
        out/('links' if stage=='link' else 'generation')/arm/f'{t.question_id}.json').exists()]
        for arm in ('llm','jev')} for stage in ('link','generate')}
    metrics['error_counts'] = dict(Counter(f"{r['arm']}:{r['category']}" for r in errors))
    for name,rows in [('schema_per_question',link_rows),('execution_per_question',execution_rows),('error_analysis',errors)]:
        if rows:
            with (out/f'{name}.csv').open('w', newline='', encoding='utf-8-sig') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    save_json(out/'metrics.json',metrics)
    lines=['# 独立 Schema Linker 与最终 EX 对照实验','',
           f"固定 {len(tasks)} 题；两组相同 Question、Evidence 和 Full Schema；最终使用 execution self-consistency。",'',
           '两个 linker 均使用原 task 中的 schema（描述与样例最多 5 个），输入哈希逐题验证；LLM 使用原 schema_selection 模板，schema 内容以同一 JSON 元数据传入。',
           'Jev 仅询问 table/column Noul；不调用 SQL intent、Join Choice、受约束 compiler 或 Jev selector。',
           '两侧使用同一 PK/FK 补全；MCTS 中强制使用 upfront schema 并关闭后续 schema-selection，其他动作保留。',
           '当前是 mandatory-linker MCTS 的配对实验，不能与旧的可跳过 linker 的 MCTS 分数直接作因果比较。','',
           '| Schema | 完成 | Table P | Table R | Column P | Column R | 完整召回 | 平均列数 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for key,r in metrics['linking'].items():
        if r['completed']:
            lines.append(f"| {key} | {r['completed']} | {r['table_precision']:.2%} | {r['table_recall']:.2%} | {r['column_precision']:.2%} | {r['column_recall']:.2%} | {r['full_schema_recall']:.2%} | {r['column_count']:.2f} |")
    lines += ['', '| 方法 | 完成 | 正确 | Coverage | 已完成 EX |','|---|---:|---:|---:|---:|']
    for arm,r in metrics['generation'].items():
        ex = f"{r['ex_on_completed']:.2%}" if r['completed'] else '尚未运行'
        lines.append(f"| {arm} linker + Qwen/MCTS + SC | {r['completed']} | {r['correct']} | {r['covered']} | {ex} |")
    lines += ['', '全流程配对 EX（已尝试的题；生成失败计错）：', '```json',
              json.dumps(metrics['paired_pipeline_ex_including_failures'],ensure_ascii=False,indent=2),'```','',
              '两侧都完成生成的配对 EX：', '```json',json.dumps(metrics['paired_ex'],ensure_ascii=False,indent=2),'```','',
              '错误分类（仅定位问题，不等价于人工因果归因）：','```json',
              json.dumps(metrics['error_counts'],ensure_ascii=False,indent=2),'```','',
              '各阶段实际用量：','```json',json.dumps(metrics['efficiency'],ensure_ascii=False,indent=2),'```','',
              'EX 复用此前评价脚本的无序多重集结果签名（保留重复行、浮点 8 位），不是 BIRD 官方完整 evaluator；空结果执行成功仍计为可执行。',
              'API/生成失败与尚未完成的题不伪装成成功运行；全量分母的 lower bound 单独保存在 metrics.json。',
              '单种子实验的显著性只反映问题级配对差异；生成随机性需多种子复验。','']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    return metrics


def prepare(args):
    configure_environment()
    out = Path(args.output_dir)
    with Path(args.tasks_file).open('rb') as f:
        all_tasks = pickle.load(f)
    with Path(args.reference_csv).open(encoding='utf-8-sig') as f:
        ids = sorted({int(r['question_id']) for r in csv.DictReader(f)})
    if len(ids) != 287:
        raise ValueError(f'Expected exactly the original 287 common questions, found {len(ids)}')
    tasks = sorted([t for t in all_tasks if t.question_id in ids], key=lambda t:t.question_id)
    if len(tasks)!=len(ids):
        raise ValueError('Missing or duplicate frozen tasks')
    cfg = {k:v for k,v in vars(args).items() if k not in ('stage','workers','limit','arms','offset')}
    cfg.update({'question_ids': ids,'input_hashes':{str(t.question_id):fingerprint(shared_input(t)) for t in tasks},
                'protocol': 'standalone-schema-only-shared-pk-fk-mandatory-mcts-v1',
                'selector': 'execution_self_consistency','jev_reranker_used':False,
                'schema_threshold':JevSettings.from_environment().schema_threshold,
                'jev_model':JevSettings.from_environment().model})
    manifest = out/'manifest.json'
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding='utf-8'))
        if cfg != previous:
            raise ValueError('Experiment configuration changed: use a new output directory')
    else:
        save_json(manifest,cfg)
        for t in tasks:
            save_json(out/'inputs'/f'{t.question_id}.json',shared_input(t))
    return tasks,cfg


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['prepare','link','generate','evaluate','all'],required=True)
    parser.add_argument('--tasks-file',default='data/preprocessed/llm_jev_300/train/tasks.pkl')
    parser.add_argument('--reference-csv',default='results/llm_vs_jev_300/per_question.csv')
    parser.add_argument('--db-root',default='data/bird/train/train_databases')
    parser.add_argument('--output-dir',default='results/linker_ablation_287_qwen35_35b_a3b')
    parser.add_argument('--model',default='qwen3.5-35B-a3b')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--arms',nargs='+',choices=['llm','jev'],default=['llm','jev'])
    parser.add_argument('--limit',type=int)
    parser.add_argument('--offset',type=int,default=0,help='Start at this frozen task index for disjoint batch execution')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--samples',type=int,default=2)
    parser.add_argument('--rollouts',type=int,default=2)
    parser.add_argument('--max-depth',type=int,default=5)
    parser.add_argument('--validation-tries',type=int,default=4)
    parser.add_argument('--link-max-tokens',type=int,default=4096)
    parser.add_argument('--generation-max-tokens',type=int,default=4096)
    parser.add_argument('--execution-timeout',type=int,default=15)
    args=parser.parse_args()
    tasks,cfg=prepare(args)
    if args.stage=='prepare':
        print({'frozen_questions':len(tasks),'output_dir':args.output_dir}); return
    if args.stage=='evaluate':
        print(json.dumps(evaluate(tasks,cfg),ensure_ascii=False,indent=2)); return
    stages=['link','generate'] if args.stage=='all' else [args.stage]
    for stage in stages:
        selected_tasks=tasks[args.offset:]
        if args.limit is not None:
            selected_tasks=selected_tasks[:args.limit]
        jobs=[(t,arm,stage,cfg) for t in selected_tasks for arm in args.arms]
        fatal=False
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(worker,p):p for p in jobs}
            for i,future in enumerate(as_completed(futures),1):
                result=future.result()
                print(json.dumps({'stage':stage,'progress':f'{i}/{len(jobs)}',**result},ensure_ascii=False),flush=True)
                error=result.get('error','').lower()
                if any(s in error for s in ['arrearage','overdue-payment','invalid_api_key','insufficient_quota','unauthorized','error code: 401','error code: 403']):
                    fatal=True
                    for f in futures: f.cancel()
                    break
        if fatal:
            print('Stopped on provider account/authentication error; successful checkpoints retained.',flush=True)
            break
    print(json.dumps(evaluate(tasks,cfg),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
