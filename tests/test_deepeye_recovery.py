import json
from pathlib import Path
import tempfile
import unittest

from deepeye.recover import quarantine_link_tail
from deepeye.runner import ROOT


class RecoveryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT/"tmp",prefix="recovery-test-")
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        (self.out/"manifest.json").write_text(json.dumps({"question_ids":[1,2,3]}),encoding="utf-8")
        self.checkpoint = self.out/"schema_linking.artifacts/schema_linking.jsonl"
        self.checkpoint.parent.mkdir()
        good = {"direct_linked_tables_and_columns":{"people":["id"]},
                "reversed_linked_tables_and_columns":{"people":["id"]}}
        bad = {"direct_linked_tables_and_columns":{"people":["id"]},
               "reversed_linked_tables_and_columns":{}}
        rows = [{"item_id":1,"stage_artifact":good},{"item_id":2,"stage_artifact":bad}]
        self.original = "".join(json.dumps(row)+"\n" for row in rows)
        self.checkpoint.write_text(self.original,encoding="utf-8")

    def test_backup_is_exact_and_retry_ids_include_missing_question(self):
        result = quarantine_link_tail(self.out,1,2,"offline test")
        self.assertEqual(result["retry_ids"],[2,3])
        self.assertEqual(Path(result["backup"]).read_text(encoding="utf-8"),self.original)
        self.assertEqual(len(self.checkpoint.read_text(encoding="utf-8").splitlines()),1)
        self.assertEqual(result["network_calls"],0)

    def test_changed_checkpoint_is_never_modified(self):
        with self.assertRaisesRegex(ValueError,"Checkpoint changed"):
            quarantine_link_tail(self.out,1,3,"offline test")
        self.assertEqual(self.checkpoint.read_text(encoding="utf-8"),self.original)

    def test_incomplete_prefix_is_never_retained(self):
        with self.assertRaisesRegex(ValueError,"incomplete linking"):
            quarantine_link_tail(self.out,2,2,"offline test")
        self.assertEqual(self.checkpoint.read_text(encoding="utf-8"),self.original)

    def test_existing_backup_is_not_overwritten(self):
        backup = self.out/"schema_linking.checkpoint.before-billing-recovery.jsonl"
        backup.write_text("original evidence",encoding="utf-8")
        with self.assertRaises(FileExistsError):
            quarantine_link_tail(self.out,1,2,"offline test")
        self.assertEqual(backup.read_text(encoding="utf-8"),"original evidence")


if __name__ == "__main__":
    unittest.main()
