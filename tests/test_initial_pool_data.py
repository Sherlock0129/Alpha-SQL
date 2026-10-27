import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from alphasql.runner.initial_pool_data import label_pool


class InitialPoolDataTest(unittest.TestCase):
    def test_execution_labels_are_offline_and_do_not_expose_gold_sql(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "tmp") as raw:
            db = Path(raw) / "demo.sqlite"
            with closing(sqlite3.connect(db)) as connection:
                with connection:
                    connection.execute("CREATE TABLE items (name TEXT)")
                    connection.executemany("INSERT INTO items VALUES (?)", [("Ada",), ("Grace",)])
            pool = {
                "question_id": 1, "db_id": "demo", "generator": {"version": "frozen-v1"},
                "candidates": [
                    {"sql": "SELECT name FROM items WHERE name = 'Ada'", "route": "full"},
                    {"sql": "SELECT name FROM items WHERE name = 'Grace'", "route": "linked"},
                ],
            }
            gold = "SELECT name FROM items WHERE name = 'Ada'"
            labelled = label_pool(pool, gold, str(db), near_miss_max_edits=10)
        self.assertEqual(labelled["covered_label"], 1)
        self.assertEqual([item["execution_label"] for item in labelled["candidates"]],
                         [1, 0])
        self.assertTrue(labelled["candidates"][1]["near_miss"])
        self.assertEqual(labelled["jev_status"], "deferred")
        self.assertNotIn(gold, json.dumps(labelled))


if __name__ == "__main__":
    unittest.main()
