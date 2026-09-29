import unittest

from alphasql.algorithm.mcts.mcts_action import MCTSNodeType, extract_sql_from_response
from alphasql.algorithm.mcts.mcts_node import MCTSNode
from alphasql.runner.direct_candidate_runner import deduplicate_paths, terminalize


class DirectCandidateRunnerTest(unittest.TestCase):
    def test_terminalize_and_deduplicate(self):
        root = MCTSNode(MCTSNodeType.ROOT)
        root.path_nodes = [root]
        nodes = []
        for sql in ("SELECT 1", " select   1 ", "SELECT 2"):
            node = MCTSNode(
                MCTSNodeType.SQL_GENERATION, parent_node=root, depth=1, sql_query=sql
            )
            node.path_nodes = [root, node]
            nodes.append(node)
        paths = deduplicate_paths(terminalize(node) for node in nodes)
        self.assertEqual(len(paths), 2)
        self.assertTrue(all(path[-1].node_type == MCTSNodeType.END for path in paths))

    def test_sql_extractor_accepts_common_model_formats(self):
        self.assertEqual(extract_sql_from_response("<sql>SELECT 1</sql>"), "SELECT 1")
        self.assertEqual(extract_sql_from_response("```sql\nSELECT 2\n```"), "SELECT 2")
        self.assertEqual(extract_sql_from_response("Answer:\nSELECT 3"), "SELECT 3")


if __name__ == "__main__":
    unittest.main()
