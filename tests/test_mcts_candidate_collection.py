import unittest

from alphasql.algorithm.mcts.mcts import MCTSSolver
from alphasql.algorithm.mcts.mcts_action import MCTSNodeType
from alphasql.algorithm.mcts.mcts_node import MCTSNode


class MCTSCandidateCollectionTest(unittest.TestCase):
    def test_collects_unvisited_generated_sql_and_deduplicates(self):
        root = MCTSNode(MCTSNodeType.ROOT)
        root.path_nodes = [root]
        first = MCTSNode(
            MCTSNodeType.SQL_GENERATION, parent_node=root, depth=1,
            sql_query="SELECT 1",
        )
        first.path_nodes = [root, first]
        duplicate = MCTSNode(
            MCTSNodeType.SQL_REVISION, parent_node=first, depth=2,
            revised_sql_query=" select   1 ",
        )
        duplicate.path_nodes = [root, first, duplicate]
        second = MCTSNode(
            MCTSNodeType.SQL_GENERATION, parent_node=root, depth=1,
            sql_query="SELECT 2",
        )
        second.path_nodes = [root, second]
        root.children = [first, second]
        first.children = [duplicate]

        solver = MCTSSolver.__new__(MCTSSolver)
        paths = solver.find_all_valid_reasoning_paths(root)
        sqls = {path[-1].final_sql_query.strip().lower() for path in paths}
        self.assertEqual(sqls, {"select 1", "select 2"})
        self.assertTrue(all(path[-1].node_type == MCTSNodeType.END for path in paths))


if __name__ == "__main__":
    unittest.main()
