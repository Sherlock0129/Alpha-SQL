import os
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import patch, MagicMock


class LiteTests(unittest.TestCase):
    def test_environment_mapping(self):
        from alphasql.llm_call.runtime import configure_environment
        with patch.dict(os.environ, {'DASHSCOPE_API_KEY': 'test-placeholder',
            'DASHSCOPE_BASE_URL': 'https://example.invalid/v1'}, clear=True):
            configure_environment()
            self.assertEqual(os.environ['OPENAI_API_KEY'], 'test-placeholder')
            self.assertEqual(os.environ['EMBEDDING_API_KEY'], 'test-placeholder')
            self.assertEqual(os.environ['OPENAI_BASE_URL'], 'https://example.invalid/v1')

    def test_sampling_and_budget(self):
        import alphasql.llm_call.openai_llm as module
        response = NS(usage=NS(prompt_tokens=10, completion_tokens=5),
            choices=[NS(message=NS(content='<sql>SELECT 1</sql>'), finish_reason='stop')])
        client = MagicMock()
        client.__enter__.return_value = client
        client.chat.completions.create.return_value = response
        with patch.object(module, 'OpenAI', return_value=client), patch.object(module, '_requests', 0), \
             patch.dict(os.environ, {'LLM_MAX_REQUESTS': '2'}):
            self.assertEqual(len(module.call_openai('test', model='qwen3-coder-flash', n=2)), 2)
            self.assertTrue(all(c.kwargs['n'] == 1 and not c.kwargs['stream']
                for c in client.chat.completions.create.call_args_list))
            with self.assertRaisesRegex(RuntimeError, 'budget'):
                module.call_openai('test', model='qwen3-coder-flash')

    def test_embedding_batch_order(self):
        from alphasql.llm_call.embedding_utils import EmbeddingModel
        model = EmbeddingModel.__new__(EmbeddingModel)
        model.model = 'text-embedding-v4'
        model.client = MagicMock()
        def response(**kwargs):
            return NS(data=[NS(index=i, embedding=[float(text)])
                for i, text in reversed(list(enumerate(kwargs['input'])))])
        model.client.embeddings.create.side_effect = response
        with patch.dict(os.environ, {'EMBEDDING_BATCH_SIZE': '10'}):
            self.assertEqual(model.embed_documents([str(i) for i in range(23)]), [[float(i)] for i in range(23)])
            self.assertEqual(model.client.embeddings.create.call_count, 3)

    def test_malformed_sql_stops(self):
        import alphasql.algorithm.mcts.mcts_action as module
        with patch.object(module, 'call_openai', return_value=['not SQL']) as call, \
             patch.object(module, 'SQL_VALIDATION_MAX_TRIES', 2):
            for cls in [module.SQLGenerationAction, module.SQLRevisionAction]:
                call.reset_mock()
                with self.assertRaises(RuntimeError):
                    cls().generate_most_consistent_sql_query('test', {'n': 1}, 'unused')
                self.assertEqual(call.call_count, 2)

    def test_database_read_only(self):
        import sqlite3
        from alphasql.database.sql_execution import execute_sql_with_timeout
        with tempfile.NamedTemporaryFile(suffix='.sqlite') as file:
            with sqlite3.connect(file.name) as conn:
                conn.execute('CREATE TABLE example (id INTEGER)')
            self.assertEqual(execute_sql_with_timeout(file.name, 'DROP TABLE example').result_type.value, 'error')


if __name__ == '__main__':
    unittest.main()
