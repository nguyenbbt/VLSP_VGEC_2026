"""CPU-only tests for the fixed inference contract; no model weights are loaded."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import assemble as pipeline
from infer_components import check_checkpoint, select_bart


def probability_row(identifier, source, values):
    return {"ID": identifier, "Input_sha256": pipeline.sha_text(source),
            "probabilities": values, "argmax": max(range(5), key=lambda k: values[k])}


def correction_row(identifier, source, text, fallback=False):
    return {"ID": identifier, "Input": source,
            "correction_checkpoint": pipeline.SPEC["bart_corrector"]["snapshot"],
            "selected_text_before_postprocessing": text,
            "selection": {"text": text, "fallback": fallback,
                          "provenance": {"family": "copy" if fallback else "bart"}}}


class PipelineTests(unittest.TestCase):
    def test_typed_id_join_average_and_label_zero_export(self):
        raw = [{"ID": 1, "Input": "First."}, {"ID": "1", "Input": "Second."}]
        bami = [probability_row("1", "Second.", [0.8, 0.2, 0, 0, 0]),
                probability_row(1, "First.", [0.8, 0.2, 0, 0, 0])]
        s0 = [probability_row(1, "First.", [0.1, 0.9, 0, 0, 0]),
              probability_row("1", "Second.", [0.8, 0.2, 0, 0, 0])]
        bart = [correction_row("1", "Second.", "Second." , fallback=True),
                correction_row(1, "First.", "Corrected first.")]
        internal, result = pipeline.assemble(raw, bami, s0, bart)
        self.assertEqual([row["predicted_label"] for row in internal], [1, 0])
        self.assertEqual(internal[0]["ensemble_probabilities"], [0.45, 0.55, 0, 0, 0])
        self.assertEqual(result[0]["Output"]["Fixed Sentence"], ["Corrected first."])
        self.assertEqual(result[1]["Output"]["Fixed Sentence"], [])
        self.assertEqual([pipeline.typed_id(row["ID"]) for row in result],
                         [("int", 1), ("str", "1")])

    def test_nonzero_fallback_keeps_label_and_raw_source(self):
        raw = [{"ID": 4, "Input": " Raw  text. "}]
        probs = [probability_row(4, raw[0]["Input"], [0.1, 0.9, 0, 0, 0])]
        corrector = [correction_row(4, raw[0]["Input"], raw[0]["Input"], fallback=True)]
        _, result = pipeline.assemble(raw, probs, probs, corrector)
        self.assertEqual(result[0]["Output"], {"Error_Type_Id": 1,
                                                 "Fixed Sentence": [" Raw  text. "]})

    def test_reject_missing_duplicate_or_bool_id_and_wrong_checkpoint(self):
        raw = [{"ID": 1, "Input": "Text."}]
        p = [probability_row(1, "Text.", [1, 0, 0, 0, 0])]
        c = [correction_row(1, "Text.", "Text.", fallback=True)]
        for rows in ([], p + p, [probability_row("1", "Text.", [1, 0, 0, 0, 0])]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                pipeline.assemble(raw, rows, p, c)
        with self.assertRaises(ValueError):
            pipeline.assemble([{"ID": True, "Input": "Text."}], p, p, c)
        c[0]["correction_checkpoint"] = "best_correction"
        with self.assertRaisesRegex(ValueError, "Wrong correction-only"):
            pipeline.assemble(raw, p, p, c)

    def test_bart_first_valid_trace_order_and_copy(self):
        trace = [{"mode": "beam", "beam": 0, "text": "Beam zero.",
                  "ended": True, "forced_boundary": False},
                 {"mode": "beam", "beam": 1, "text": "Beam one.",
                  "ended": True, "forced_boundary": False}]
        self.assertEqual(select_bart("Source.", trace)["text"], "Beam zero.")
        trace[0]["forced_boundary"] = True
        self.assertEqual(select_bart("Source.", trace)["text"], "Beam one.")
        trace[1]["ended"] = False
        result = select_bart(" Raw  source. ", trace)
        self.assertEqual(result["text"], " Raw  source. ")
        self.assertTrue(result["fallback"])

    def test_s0_requires_trained_attention_pooling(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / pipeline.SPEC["s0_classifier"]["run"] / "best_classification"
            base.mkdir(parents=True)
            (base / "resolved_config.json").write_text(json.dumps({"kind": "bartpho", "joint": True,
                "classification_pooling": "attention"}), encoding="utf-8")
            (base / "run_manifest.json").write_text(json.dumps({"identity_hash": pipeline.SPEC["s0_classifier"]["identity"]}), encoding="utf-8")
            with self.assertRaisesRegex(FileNotFoundError, "attention_pool.pt"):
                for name in ("model.safetensors", "classifier.pt"):
                    (base / name).write_bytes(b"test")
                # Fails at the first absent required file; supply tokenizer stubs.
                for name in ("sentencepiece.bpe.model", "dict.txt"):
                    (base / name).write_bytes(b"test")
                check_checkpoint("s0", base)

    def test_checkpoint_hash_rejects_swapped_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / pipeline.SPEC["bami_classifier"]["snapshot"]
            tokenizer = base / "tokenizer"
            tokenizer.mkdir(parents=True)
            (base / "identity.json").write_text(json.dumps({
                "identity": pipeline.SPEC["bami_classifier"]["identity"]}), encoding="utf-8")
            model = base / "model.pt"
            model.write_bytes(b"original test weights")
            token = tokenizer / "tokenizer.json"
            token.write_bytes(b"original test tokenizer")
            expected = {"model.pt": hashlib.sha256(model.read_bytes()).hexdigest(),
                        "tokenizer/tokenizer.json": hashlib.sha256(token.read_bytes()).hexdigest()}
            with patch.dict(pipeline.SPEC["bami_classifier"], {"file_sha256": expected}):
                self.assertEqual(check_checkpoint("bami", base),
                                 pipeline.SPEC["bami_classifier"]["identity"])
                model.write_bytes(b"swapped test weights")
                with self.assertRaisesRegex(ValueError, "model.pt"):
                    check_checkpoint("bami", base)

    def test_bart_cache_metadata_rejects_changed_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            input_path = base / "input.json"
            input_path.write_text('[{"ID": 1, "Input": "Source."}]', encoding="utf-8")
            cache = base / "bart_predictions.jsonl"
            cache.write_text('{"ID": 1}\n', encoding="utf-8")
            expected = pipeline.SPEC["bart_corrector"]
            meta = {"role": "bart", "rows": 1,
                    "identity": expected["identity"],
                    "checkpoint": "bart-final/" + expected["snapshot"],
                    "input_file_sha256": pipeline.sha_file(input_path),
                    "output_sha256": pipeline.sha_file(cache),
                    "model_eval": True, "inference_mode": True,
                    "tokenizer_sha256": {Path(name).name: digest for name, digest in expected["file_sha256"].items()
                                          if name.startswith("tokenizer/")}}
            cache.with_suffix(".metadata.json").write_text(json.dumps(meta), encoding="utf-8")
            self.assertEqual(pipeline.validate_bart_metadata(cache, input_path, 1),
                             cache.with_suffix(".metadata.json"))
            cache.write_text('{"ID": 2}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum"):
                pipeline.validate_bart_metadata(cache, input_path, 1)


if __name__ == "__main__":
    unittest.main()
