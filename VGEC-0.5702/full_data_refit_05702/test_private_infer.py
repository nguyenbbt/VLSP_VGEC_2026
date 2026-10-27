import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch
import zipfile

import private_infer as inference
from assemble import assemble as frozen_assemble, SPEC


def probability(identifier, source, values):
    return dict(ID=identifier, Input_sha256=inference.sha_text(source), probabilities=values,
                argmax=max(range(5), key=lambda k: values[k]))


def correction(identifier, source, texts, snapshot="step-refit"):
    trace = [dict(mode="beam", beam=i, text=text, ended=True, forced_boundary=False)
             for i, text in enumerate(texts)]
    chosen = inference.select_bart(source, trace)
    return dict(ID=identifier, Input=source, Input_sha256=inference.sha_text(source),
                generation_trace=trace, selection=chosen, selected_text_before_postprocessing=chosen["text"],
                correction_checkpoint=snapshot)


class PrivateInferTests(unittest.TestCase):
    def fixture(self):
        raw = [dict(ID=1, Input="First."), dict(ID="1", Input=" Raw  second. ")]
        bami = [probability("1", raw[1]["Input"], [.8, .2, 0, 0, 0]),
                probability(1, raw[0]["Input"], [.8, .2, 0, 0, 0])]
        s0 = [probability(1, raw[0]["Input"], [.1, .9, 0, 0, 0]),
              probability("1", raw[1]["Input"], [.8, .2, 0, 0, 0])]
        bart = [correction("1", raw[1]["Input"], ["Changed second.", "Other."]),
                correction(1, raw[0]["Input"], ["Corrected first.", "Other."])]
        return raw, dict(bami=bami, s0=s0, bart=bart)

    def test_typed_id_average_and_zero_export_match_frozen_pipeline(self):
        raw, records = self.fixture()
        internal, exported = inference.assemble_predictions(raw, records, "step-refit")
        self.assertEqual(internal[0]["ensemble_probabilities"], [.45, .55, 0, 0, 0])
        self.assertEqual([r["predicted_label"] for r in internal], [1, 0])
        self.assertEqual(exported[0]["Output"]["Fixed Sentence"], ["Corrected first."])
        self.assertEqual(exported[1]["Output"]["Fixed Sentence"], [])
        old_bart = copy.deepcopy(records["bart"])
        for r in old_bart:
            r["correction_checkpoint"] = SPEC["bart_corrector"]["snapshot"]
        _, frozen = frozen_assemble(raw, records["bami"], records["s0"], old_bart)
        self.assertEqual(exported, frozen)

    def test_first_valid_then_raw_copy_preserves_nonzero_label(self):
        raw, records = self.fixture()
        row = records["bart"][1]
        row["generation_trace"][0]["forced_boundary"] = True
        row["selection"] = inference.select_bart(row["Input"], row["generation_trace"])
        row["selected_text_before_postprocessing"] = row["selection"]["text"]
        _, result = inference.assemble_predictions(raw, records, "step-refit")
        self.assertEqual(result[0]["Output"]["Fixed Sentence"], ["Other."])
        row["generation_trace"][1]["ended"] = False
        row["selection"] = inference.select_bart(row["Input"], row["generation_trace"])
        row["selected_text_before_postprocessing"] = row["selection"]["text"]
        _, result = inference.assemble_predictions(raw, records, "step-refit")
        self.assertEqual(result[0]["Output"], {"Error_Type_Id": 1, "Fixed Sentence": ["First."]})

    def test_reject_join_hash_probability_checkpoint_or_trace_mismatch(self):
        raw, records = self.fixture()
        cases = []
        for kind in ("missing", "duplicate", "input", "mapping", "checkpoint", "trace"):
            changed = copy.deepcopy(records)
            if kind == "missing": changed["s0"].pop()
            if kind == "duplicate": changed["bami"].append(changed["bami"][0])
            if kind == "input": changed["bami"][0]["Input_sha256"] = "wrong"
            if kind == "mapping": changed["s0"][0]["probabilities"] = [.5, .5]
            if kind == "checkpoint": changed["bart"][0]["correction_checkpoint"] = "old-checkpoint"
            if kind == "trace": changed["bart"][1]["selected_text_before_postprocessing"] = "Other."
            cases.append((kind, changed))
        for kind, changed in cases:
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                inference.assemble_predictions(raw, changed, "step-refit")
        with self.assertRaises(ValueError):
            inference.source_only([dict(ID=True, Input="Text.")])
        self.assertEqual(inference.source_only([dict(ID=1, Input="Text.", Output={"gold": 1})]),
                         [dict(ID=1, Input="Text.")])

    def full_checkpoint(self, root, role):
        identity, steps = inference.training_identity(role)
        run_id = inference.RUNS[role]
        run = root / run_id
        manifest = dict(identity=identity, sources=inference.COUNTS[role], optimizer_steps=steps,
                        complete=True, smoke_only=False)
        if role == "s0":
            checkpoint = run / "best_classification"
            checkpoint.mkdir(parents=True)
            for name in ("model.safetensors", "classifier.pt", "attention_pool.pt", "tokenizer_config.json",
                         "sentencepiece.bpe.model", "dict.txt"):
                (checkpoint / name).write_bytes(b"test inference bytes")
            inference.write_json(run / "identity.json", identity)
            inference.write_json(checkpoint / "resolved_config.json", identity["config"])
            inference.write_json(checkpoint / "run_manifest.json", identity)
            manifest["epoch"] = inference.EPOCHS[role]
        else:
            store = run / "checkpoints" / f"{role}-full"
            checkpoint = store / "step-refit"
            (checkpoint / "tokenizer").mkdir(parents=True)
            for name in ("model.pt", "tokenizer/tokenizer_config.json", "tokenizer/tokenizer.json",
                         "tokenizer/sentencepiece.bpe.model", "tokenizer/dict.txt"):
                (checkpoint / name).write_bytes(b"test inference bytes")
            inference.write_json(checkpoint / "identity.json", dict(identity=identity))
            inference.write_json(checkpoint / "progress.json", dict(done=True, epoch=inference.EPOCHS[role], step=steps))
            inference.write_json(checkpoint / "complete.json", dict(identity=identity, files={
                p.relative_to(checkpoint).as_posix(): inference.sha_file(p)
                for p in checkpoint.rglob("*") if p.is_file()}))
            inference.write_json(store / "pointers.json", dict(identity=identity, best=checkpoint.name))
            manifest.update(role=role, stop_epoch=inference.EPOCHS[role], checkpoint=checkpoint.name)
        inference.write_json(run / "run_manifest.json", manifest)
        return run, checkpoint

    def test_only_completed_full_checkpoints_and_trained_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for role in inference.ROLES:
                run, checkpoint = self.full_checkpoint(root, role)
                proof = inference.checkpoint_info(root, role, run.name)
                self.assertEqual(proof["epoch"], inference.EPOCHS[role])
                manifest = inference.read_json(run / "run_manifest.json")
                manifest["smoke_only"] = True
                inference.write_json(run / "run_manifest.json", manifest)
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    inference.checkpoint_info(root, role, run.name)
                manifest["smoke_only"] = False
                inference.write_json(run / "run_manifest.json", manifest)
                if role == "s0":
                    (checkpoint / "attention_pool.pt").unlink()
                    with self.assertRaisesRegex(FileNotFoundError, "attention_pool.pt"):
                        inference.checkpoint_info(root, role, run.name)
                else:
                    (checkpoint / "model.pt").write_bytes(b"swapped weights")
                    with self.assertRaisesRegex(ValueError, "checksum"):
                        inference.checkpoint_info(root, role, run.name)
            with self.assertRaisesRegex(ValueError, "smoke"):
                inference.checkpoint_info(root, "bami", "smoke-bami-05702")

    def test_export_zip_hash_and_cache_identity(self):
        raw, records = self.fixture()
        identity = dict(checkpoint_proofs={"bart": {"checkpoint": "/run/step-refit"}})
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            result = inference.export_private(directory, raw, records, identity)
            with zipfile.ZipFile(result["zip"]) as archive:
                self.assertEqual(archive.namelist(), ["results.json"])
                self.assertEqual(archive.read("results.json"), Path(result["results"]).read_bytes())
            self.assertEqual(inference.sha_file(Path(result["results"])), result["results_sha256"])
            self.assertEqual(inference.cached_rows(directory, "bami", identity, raw),
                             inference.align(raw, records["bami"]))
            self.assertEqual(inference.export_private(directory, raw, records, identity)["zip_sha256"], result["zip_sha256"])
            with self.assertRaisesRegex(ValueError, "different checkpoints"):
                inference.export_private(directory, raw, records, dict(checkpoint_proofs={}))
            data = inference.read_json(directory / "bami.json")
            data["rows"][0]["probabilities"] = [.2, .8, 0, 0, 0]
            inference.write_json(directory / "bami.json", data)
            with self.assertRaisesRegex(ValueError, "corrupt"):
                inference.cached_rows(directory, "bami", identity, raw)

    def test_modal_entrypoint_import_from_root_in_container(self):
        fake_modal = types.ModuleType("modal")
        fake_modal.is_local = lambda: False
        fake_modal.Volume = MagicMock()
        fake_app = MagicMock()
        fake_app.function.side_effect = lambda **kwargs: lambda f: f
        fake_app.local_entrypoint.side_effect = lambda **kwargs: lambda f: f
        fake_modal.App = lambda name: fake_app
        fake_image = MagicMock()
        for method in ("pip_install", "env", "add_local_dir", "add_local_file"):
            getattr(fake_image, method).return_value = fake_image
        fake_modal.Image = MagicMock()
        fake_modal.Image.debian_slim.return_value = fake_image
        scope = dict(__name__="simulated_modal_private", __file__="/root/modal_private.py")
        with patch.dict(sys.modules, modal=fake_modal), patch.object(sys, "path", list(sys.path)):
            exec(compile((inference.HERE / "modal_private.py").read_text(encoding="utf-8"),
                         "/root/modal_private.py", "exec"), scope)
        self.assertEqual(scope["ROOT"], Path("/workspace"))
        self.assertEqual(scope["HERE"], Path("/workspace/experiments/full_data_refit_05702"))
        mounted_files = {call.args[0].name for call in fake_image.add_local_file.call_args_list}
        self.assertEqual(mounted_files, {"private_infer.py", "modal_private.py", "refit.py",
                                        "prepared_manifest.json", "checkpoints.json"})


if __name__ == "__main__":
    unittest.main()
