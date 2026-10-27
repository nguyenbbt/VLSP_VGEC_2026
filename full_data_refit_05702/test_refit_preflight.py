import json
from pathlib import Path
import tempfile
import unittest

from audit_data import audit
from modal_refit import resolve_paths
from refit import COUNTS, EPOCHS, inputs


def record(identifier, text, label=0, refs=None):
    return {"ID": identifier, "Input": text,
            "Output": {"Error_Type_Id": label, "Fixed Sentence": refs if refs is not None else []}}


class FullDataPreflightTests(unittest.TestCase):
    def test_modal_entrypoint_paths_in_container(self):
        here, root = resolve_paths("/root/modal_refit.py", False)
        self.assertEqual(here, Path("/workspace/experiments/full_data_refit_05702"))
        self.assertEqual(root, Path("/workspace"))

    def make_root(self, root, final, dev, prepared):
        (root / "data/prepared_final").mkdir(parents=True)
        (root / "the_final_data (1).json").write_text(json.dumps(final), encoding="utf-8")
        (root / "dev.json").write_text(json.dumps(dev), encoding="utf-8")
        (root / "data/prepared_final/train.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in prepared), encoding="utf-8")

    def test_dev_is_already_in_final_despite_different_id(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_root(root,
                           [record(0, "Sai câu", 1, ["Câu đúng"]), record(1, "Câu khác")],
                           [record("dev-7", "Sai câu", 1, ["Câu đúng"])],
                           [dict(uid="train:1", source_raw="Câu khác", label=0, references_raw=[])])
            report = audit(root)
            self.assertEqual(report["full_refit_unique_sources"], 2)
            self.assertEqual(report["dev_rows_with_different_typed_id_in_final"], 1)

    def test_reference_or_label_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_root(root, [record(0, "Sai câu", 1, ["Câu đúng"])],
                           [record(0, "Sai câu", 2, ["Câu đúng"])], [])
            with self.assertRaisesRegex(ValueError, "gold mismatches"):
                audit(root)

    def test_duplicate_normalized_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_root(root, [record(0, "Câu  mẫu"), record(1, "Câu mẫu")],
                           [record(2, "Câu mẫu")], [])
            with self.assertRaisesRegex(ValueError, "Duplicate normalized"):
                audit(root)

    def test_real_data_and_historical_fixed_epochs(self):
        self.assertEqual(EPOCHS, {"bami": 14, "s0": 10, "bart": 6})
        for role in EPOCHS:
            rows, manifest = inputs(role)
            self.assertEqual(len(rows), COUNTS[role])
            self.assertEqual(manifest["dev_sources_in_optimizer"], 90)


if __name__ == "__main__":
    unittest.main()
