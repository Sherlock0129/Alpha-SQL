"""Preflight, API probe, and a bounded five-question Alpha-SQL pipeline."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alphasql.llm_call.runtime import configure_environment


def settings():
    configure_environment()
    import yaml
    return yaml.safe_load((ROOT / 'config/qwen3_flash_smoke5.yaml').read_text())


def preflight(require_key=True):
    cfg = settings()
    rows = json.loads((ROOT / cfg['subset_file_path']).read_text())
    if len(rows) != 5 or len({r['question_id'] for r in rows}) != 5:
        raise ValueError('Expected exactly five distinct question IDs')
    from alphasql.database.sql_execution import execute_sql_with_timeout
    for row in rows:
        db = ROOT / cfg['db_root_dir'] / row['db_id'] / (row['db_id'] + '.sqlite')
        answer = execute_sql_with_timeout(str(db), row['SQL'], timeout=30)
        if answer.result_type.value != 'success':
            raise ValueError(f"Gold SQL failed for question {row['question_id']}")
    key_set = bool(os.getenv('DASHSCOPE_API_KEY'))
    print(json.dumps({'model': cfg['mcts_model_kwargs']['model'],
        'base_url': os.getenv('DASHSCOPE_BASE_URL'),
        'embedding_model': os.getenv('EMBEDDING_MODEL'),
        'key_configured': key_set, 'questions': len(rows),
        'gold_sql_checks': 'passed'}, ensure_ascii=False))
    if require_key and not key_set:
        raise ValueError('DASHSCOPE_API_KEY is missing. Set it in .env.local or the shell.')
    return cfg, rows


def probe():
    cfg, _ = preflight()
    from alphasql.llm_call.openai_llm import call_openai
    from alphasql.llm_call.embedding_utils import get_embedding_model
    outputs = call_openai('Return only <sql>SELECT 1;</sql>',
        model=cfg['mcts_model_kwargs']['model'], n=2, max_tokens=128)
    if not all('<sql>' in output and '</sql>' in output for output in outputs):
        raise ValueError('SQL output format check failed')
    vectors = get_embedding_model().embed_documents(['hello', 'world'])
    if len(vectors) != 2 or not vectors[0] or len(vectors[0]) != len(vectors[1]):
        raise ValueError('Embedding response check failed')
    print('API probe passed: two SQL samples and two embedding vectors.')


def worker(index, result_dir):
    cfg = settings()
    from alphasql.algorithm.mcts.mcts import MCTSSolver
    from alphasql.algorithm.mcts.reward import MajorityVoteRewardModel
    with (ROOT / cfg['tasks_file_path']).open('rb') as handle:
        task = pickle.load(handle)[index]
    random.seed(cfg['random_seed'] + task.question_id)
    # Never supply reference SQL to search or reward computation.
    task = task.model_copy(update={'sql': None})
    MCTSSolver(task=task, db_root_dir=str(ROOT / cfg['db_root_dir']),
        max_rollout_steps=cfg['max_rollout_steps'], max_depth=cfg['max_depth'],
        exploration_constant=cfg['exploration_constant'], save_root_dir=str(result_dir),
        llm_kwargs=cfg['mcts_model_kwargs'],
        reward_model=MajorityVoteRewardModel(cfg['reward_model_kwargs'])).solve()


def run():
    cfg, rows = preflight()
    run_dir = ROOT / cfg['save_root_dir'] / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    run_dir.mkdir(parents=True)
    os.environ['ALPHASQL_USAGE_PATH'] = str(run_dir / 'usage.jsonl')
    (run_dir / 'config.json').write_text(json.dumps(cfg, indent=2))
    command = [sys.executable, '-m', 'alphasql.runner.preprocessor',
        '--data_file_path', cfg['subset_file_path'], '--database_root_dir', cfg['db_root_dir'],
        '--save_root_dir', 'data/preprocessed/smoke5', '--lsh_threshold', '0.5',
        '--lsh_signature_size', '128', '--lsh_n_gram', '3', '--lsh_top_k', '20',
        '--edit_similarity_threshold', '0.3', '--embedding_similarity_threshold', '0.6',
        '--n_parallel_processes', '1', '--max_dataset_samples', '5']
    with (run_dir / 'preprocess.log').open('w') as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=900)
    from alphasql.runner.sql_selection import select_final_sql_query
    from alphasql.database.sql_execution import execute_sql_with_timeout
    report, predictions = [], {}
    for index, row in enumerate(rows):
        qid = row['question_id']
        started = time.monotonic()
        record = {'question_id': qid, 'db_id': row['db_id'], 'correct': False}
        try:
            with (run_dir / f'{qid}.log').open('w') as log:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), 'worker',
                    '--index', str(index), '--result-dir', str(run_dir)], cwd=ROOT,
                    stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
            prediction = select_final_sql_query(str(run_dir / f'{qid}.pkl'), str(ROOT / cfg['db_root_dir']))['sql']
            predictions[str(qid)] = prediction
            db = str(ROOT / cfg['db_root_dir'] / row['db_id'] / f"{row['db_id']}.sqlite")
            actual = execute_sql_with_timeout(db, prediction, timeout=30)
            expected = execute_sql_with_timeout(db, row['SQL'], timeout=30)
            record.update(sql=prediction, status=actual.result_type.value,
                correct=actual.result_type.value == 'success' and expected.result_type.value == 'success'
                and set(actual.result) == set(expected.result))
        except Exception as exc:
            # Only exception type is reported; credentials are never copied into reports.
            record.update(status='failed', error_type=type(exc).__name__)
        record['seconds'] = round(time.monotonic() - started, 2)
        report.append(record)
        summary = {'evaluation': 'BIRD-style execution result set equality; not full official evaluator',
            'correct': sum(r['correct'] for r in report), 'total': 5,
            'completed': len(report), 'questions': report}
        (run_dir / 'evaluation.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        (run_dir / 'pred_sqls.json').write_text(json.dumps(predictions, indent=2, ensure_ascii=False))
        print(f"Question {qid}: {record['status']}, correct={record['correct']}", flush=True)
    print(f'Results: {run_dir}')


if __name__ == '__main__':
    os.chdir(ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['check', 'probe', 'run', 'worker'])
    parser.add_argument('--index', type=int)
    parser.add_argument('--result-dir', type=Path)
    args = parser.parse_args()
    try:
        if args.action == 'check':
            preflight(require_key=False)
        elif args.action == 'probe':
            probe()
        elif args.action == 'run':
            run()
        else:
            worker(args.index, args.result_dir)
    except Exception as exc:
        print(f'{type(exc).__name__}: operation failed; check configuration and local logs.', file=sys.stderr)
        if isinstance(exc, ValueError):
            print(str(exc), file=sys.stderr)
        sys.exit(1)
