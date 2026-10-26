import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from alphasql.algorithm.mcts.mcts_action import SchemaSelectionAction, SQLGenerationAction, MCTSNodeType
from alphasql.algorithm.mcts.mcts_node import MCTSNode
from alphasql.database.schema import ColumnSchema, TableSchema, DatabaseSchema
from alphasql.jev.client import JevClient
from alphasql.jev.runtime import JevSettings
from alphasql.runner.task import Task
from alphasql.runner.linker_ablation import (
    shared_input, fingerprint, canonical_selection, completed_selection, link_llm,
    link_jev, FixedLinkerMCTS, schema_metrics, execution_metrics, paired_stats,
    evaluate, save_json,
)
from alphasql.algorithm.selection.schema_signal import result_signature


def example():
    schema = DatabaseSchema(tables={
        'customers': TableSchema('customers', {
            'id': ColumnSchema(original_column_name='id', column_type='INTEGER', primary_key=True),
            'name': ColumnSchema(original_column_name='name', column_type='TEXT'),
        }),
        'orders': TableSchema('orders', {
            'id': ColumnSchema(original_column_name='id', column_type='INTEGER', primary_key=True),
            'customer_id': ColumnSchema(original_column_name='customer_id', column_type='INTEGER',
                                        foreign_keys=[('customers','id')]),
            'amount': ColumnSchema(original_column_name='amount', column_type='REAL'),
        }),
    })
    return schema


class LinkerAblationTest(unittest.TestCase):
    def test_shared_input_excludes_gold_and_matches_llm_schema_payload(self):
        schema=example()
        task=Task(question_id=1, db_id='toy', question='Names?', evidence='hint',
                  sql='SECRET GOLD', table_schema_dict=schema.tables)
        state=shared_input(task)
        self.assertNotIn('SECRET GOLD', str(state))
        self.assertEqual(fingerprint(state), fingerprint(shared_input(task)))
        with patch('alphasql.runner.linker_ablation.call_openai',return_value=['```json\n{"customers":["name"]}\n```']) as call:
            result=link_llm(state,schema,'model',4096)
        self.assertEqual(result['raw_selection'],{'customers':['name']})
        self.assertIn('Names?',call.call_args.args[0])
        self.assertIn('hint',call.call_args.args[0])
        self.assertNotIn('SECRET GOLD',call.call_args.args[0])

    def test_jev_is_schema_only_and_preserves_probabilities(self):
        schema=example(); settings=JevSettings(schema_threshold=0.5)
        state={'question':'Names?', 'evidence':'', 'schema':[]}
        def transport(actual, questions):
            self.assertEqual(actual,state)
            self.assertFalse(any(k.startswith('intent') or k=='join_path' for k in questions))
            return {'answers':{k:{'noul':0.9 if k in ('table_0','column_0_1') else 0.1}
                               for k in questions}}
        result=link_jev(state,schema,settings,JevClient(settings,transport))
        self.assertEqual(result['raw_selection'],{'customers':['name']})
        self.assertEqual(len(result['probabilities']),7)
        self.assertEqual(completed_selection(result['raw_selection'],schema),{'customers':['id','name']})

    def test_common_completion_adds_keys_but_no_unselected_bridge_tables(self):
        schema=example()
        self.assertEqual(completed_selection({'orders':['amount']},schema),{'orders':['id','amount']})
        both=completed_selection({'orders':['amount'],'customers':['name']},schema)
        self.assertIn('customer_id',both['orders'])
        self.assertEqual(set(both),{'orders','customers'})

    def test_unknown_references_and_empty_selections_fail_without_full_schema_fallback(self):
        for selection in ({},{'missing':['x']},{'orders':['missing']},{'orders':'amount'}):
            with self.assertRaises(ValueError): canonical_selection(selection,example())

    def test_mcts_enforces_identical_fixed_context_without_second_linker(self):
        schema=example()
        solver=FixedLinkerMCTS.__new__(FixedLinkerMCTS)
        solver.fixed_tables={'customers':schema.tables['customers']}
        solver.fixed_context='FIXED CONTEXT'; solver.max_depth=5; solver.llm_kwargs={}
        node=MCTSNode(MCTSNodeType.ROOT); node.path_nodes=[node]
        def generated(n,kwargs):
            self.assertEqual(n.selected_schema_context,'FIXED CONTEXT')
            self.assertEqual(set(n.selected_schema_dict),{'customers'})
            return []
        with patch('alphasql.runner.linker_ablation.get_valid_action_space_for_node',
                   return_value=[SchemaSelectionAction(),SQLGenerationAction()]), \
             patch.object(SchemaSelectionAction,'create_children_nodes',side_effect=AssertionError('second linker called')), \
             patch.object(SQLGenerationAction,'create_children_nodes',side_effect=generated):
            solver.expand(node)

    def test_completion_increases_recall_and_can_lower_precision(self):
        raw=schema_metrics({'customers':['name']},({'customers'},{('customers','id'),('customers','name')}))
        final=schema_metrics({'customers':['id','name']},({'customers'},{('customers','id'),('customers','name')}))
        self.assertEqual(raw['column_recall'],0.5)
        self.assertEqual(final['full_schema_recall'],1)

    def test_empty_result_is_executable_and_duplicates_preserved(self):
        import sqlite3
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'toy.sqlite'
            with closing(sqlite3.connect(path)) as conn:
                conn.execute('CREATE TABLE t (v INTEGER)')
                conn.commit()
            result=execution_metrics(['SELECT v FROM t'],path,result_signature([]),2)
        self.assertEqual(result['executable_count'],1)
        self.assertEqual(result['ex'],1)
        self.assertNotEqual(result_signature([(1,),(1,)]),result_signature([(1,)]))

    def test_paired_statistics_track_fixes_and_regressions(self):
        stats=paired_stats([(0,1),(1,0),(1,1),(0,1)])
        self.assertEqual(stats['fixes'],2)
        self.assertEqual(stats['regressions'],1)
        self.assertEqual(stats['delta'],0.25)

    def test_failed_generation_stays_in_pipeline_accuracy_denominator(self):
        import sqlite3
        schema=example()
        task=Task(question_id=1,db_id='toy',question='Names?',evidence='',
                  sql='SELECT name FROM customers',table_schema_dict=schema.tables)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'toy').mkdir()
            with closing(sqlite3.connect(root/'toy'/'toy.sqlite')) as conn:
                conn.execute('CREATE TABLE customers (id INTEGER, name TEXT)')
                conn.execute("INSERT INTO customers VALUES (1, 'Ada')"); conn.commit()
            for arm in ('llm','jev'):
                save_json(root/'links'/arm/'1.json',{
                    'input_sha256':fingerprint(shared_input(task)),
                    'raw_selection':{'customers':['name']},
                    'completed_selection':{'customers':['id','name']},
                    'elapsed_seconds':1,
                })
            save_json(root/'generation'/'llm'/'1.json',{})
            save_json(root/'failures'/'generate'/'jev'/'1.json',{'error':'output truncated'})
            (root/'candidates'/'llm').mkdir(parents=True)
            (root/'candidates'/'llm'/'1.pkl').touch()
            with patch('alphasql.runner.linker_ablation._candidate_sqls',
                       return_value=(['SELECT name FROM customers'],[])):
                result=evaluate([task],{'output_dir':tmp,'db_root':tmp,'execution_timeout':2})
        self.assertEqual(result['paired_ex']['n'],0)
        self.assertEqual(result['paired_pipeline_ex_including_failures']['n'],1)
        self.assertEqual(result['paired_pipeline_ex_including_failures']['delta'],-1)
        self.assertEqual(result['generation']['jev']['failed'],1)


if __name__=='__main__': unittest.main()
