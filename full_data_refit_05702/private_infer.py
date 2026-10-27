"""Inference and export for the completed full-data refit, never historical weights."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUNDLE = ROOT / "VGEC-0.5702"
for directory in (BUNDLE / "components/sub1/src", BUNDLE / "components/s0/src", BUNDLE / "scripts"):
    sys.path.insert(0, str(directory))

# Reuse the frozen input join, probability checks and exact generator selector.
# The historical checkpoint checks and historical model_rows are never called.
from assemble import align, probabilities, sha_file, sha_text, typed_id, validate_sources
from infer_components import select_bart, valid_probs
from refit import COUNTS, DATA, EPOCHS, code_identity

ROLES = ("bami", "s0", "bart")
RUNS = {role: f"{role}-full-05702" for role in ROLES}
LABELS = [0, 1, 2, 3, 4]
POLICY = dict(ensemble="0.5*p_bami+0.5*p_s0", label_order=LABELS, temperature=1,
              argmax="first maximum in label order", bami_softmax="FP32 logits to FP64",
              s0_softmax="FP32", correction="BART first valid beam in generator order; else Copy raw Input",
              export="label 0: []; labels 1-4: [selected_bart_text]")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def digest(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(json_bytes(value))
    temporary.replace(path)


def source_only(raw):
    validate_sources(raw)
    return [{"ID": row["ID"], "Input": row["Input"]} for row in raw]


def safe_id(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("Unsafe inference/run ID")
    return value


def training_identity(role):
    """Reconstruct the exact identity written by the unchanged training runner."""
    manifest = read_json(HERE / "prepared_manifest.json")
    if role == "s0":
        from vgec.io import load_config
        cfg = load_config(BUNDLE / "components/s0/configs/models/bartpho_attention.yaml")
        identity = dict(experiment="full-data-refit-05702-v1", role="s0",
                        data_sha256=manifest["prepared_sha256"][DATA[role]], config=cfg,
                        code_sha256=code_identity(role), stop_epoch=EPOCHS[role],
                        horizon_epochs=cfg["max_epochs"], selection="fixed prior selected epoch")
        batch = cfg["micro_batch"] * cfg["accumulation"]
    else:
        from vgec_sub1.config import load, role_config
        from vgec_sub1.io import digest as training_digest
        cfg = load(BUNDLE / "components/sub1/configs/sub1.yaml")
        rc = role_config(cfg, role)
        identity = training_digest(dict(experiment="full-data-refit-05702-v1", role=role,
                    data_sha256=manifest["prepared_sha256"][DATA[role]], config_hash=cfg["_config_hash"],
                    stop_epoch=EPOCHS[role], horizon_epochs=rc["epochs"],
                    micro=32 if role == "bami" else 16, code_sha256=code_identity(role)))
        batch = rc["global_batch"]
    return identity, math.ceil(COUNTS[role] / batch) * EPOCHS[role]


def checkpoint_info(artifact_root, role, run_id):
    """CPU validation of completed refit identity and inference files; no weights loaded."""
    if role not in ROLES or safe_id(run_id).startswith("smoke-"):
        raise ValueError("Full refit role/run required; smoke is forbidden")
    run = Path(artifact_root) / run_id
    manifest = read_json(run / "run_manifest.json")
    expected_identity, steps = training_identity(role)
    if (manifest.get("complete") is not True or manifest.get("smoke_only") is not False
            or manifest.get("identity") != expected_identity or manifest.get("sources") != COUNTS[role]
            or manifest.get("optimizer_steps") != steps):
        raise ValueError(f"{role}: full train incomplete or training identity/count/steps mismatch")
    if role == "s0":
        if manifest.get("epoch") != EPOCHS[role] or read_json(run / "identity.json") != expected_identity:
            raise ValueError("S0 fixed epoch/identity mismatch")
        checkpoint = run / "best_classification"
        cfg = read_json(checkpoint / "resolved_config.json")
        if (cfg != expected_identity["config"] or cfg.get("joint") is not True
                or cfg.get("classification_pooling") != "attention"
                or read_json(checkpoint / "run_manifest.json") != expected_identity):
            raise ValueError("Wrong S0 architecture/checkpoint identity")
        required = {"model.safetensors", "classifier.pt", "attention_pool.pt", "resolved_config.json",
                    "run_manifest.json", "tokenizer_config.json", "sentencepiece.bpe.model", "dict.txt"}
        files = {p.relative_to(checkpoint).as_posix(): sha_file(p)
                 for p in checkpoint.rglob("*") if p.is_file()}
    else:
        if manifest.get("role") != role or manifest.get("stop_epoch") != EPOCHS[role]:
            raise ValueError("Sub1 role/fixed epoch mismatch")
        store = run / "checkpoints" / f"{role}-full"
        pointer = read_json(store / "pointers.json")
        name = pointer.get("best", "")
        checkpoint = store / name
        if (not name.startswith("step-") or checkpoint.resolve().parent != store.resolve()
                or pointer.get("identity") != expected_identity or manifest.get("checkpoint") != name):
            raise ValueError("Sub1 best pointer/manifest mismatch")
        complete = read_json(checkpoint / "complete.json")
        if (complete.get("identity") != expected_identity
                or read_json(checkpoint / "identity.json").get("identity") != expected_identity):
            raise ValueError("Sub1 snapshot identity mismatch")
        state = read_json(checkpoint / "progress.json")
        if state.get("done") is not True or state.get("epoch") != EPOCHS[role] or state.get("step") != steps:
            raise ValueError("Sub1 checkpoint is not the completed fixed epoch")
        required = {"model.pt", "identity.json", "progress.json", "tokenizer/tokenizer_config.json"}
        required |= ({"tokenizer/tokenizer.json"} if role == "bami" else
                     {"tokenizer/sentencepiece.bpe.model", "tokenizer/dict.txt"})
        files = {}
        for name, expected in complete["files"].items():
            if name == "training.pt":  # Optimizer state is never loaded for inference.
                continue
            path = (checkpoint / name).resolve()
            if not path.is_relative_to(checkpoint.resolve()) or sha_file(path) != expected:
                raise ValueError("Sub1 snapshot checksum failure: " + name)
            files[name] = expected
    if not required <= set(files):
        raise FileNotFoundError(f"{role}: missing trained inference files: {sorted(required-set(files))}")
    return dict(role=role, run_id=run_id, checkpoint=str(checkpoint), identity=expected_identity,
                optimizer_steps=steps, epoch=EPOCHS[role], files_sha256=files,
                run_manifest_sha256=sha_file(run / "run_manifest.json"))


def job_identity(sources, input_sha256, proofs):
    paths = [HERE / "private_infer.py", HERE / "modal_private.py", BUNDLE / "scripts/assemble.py",
             BUNDLE / "scripts/infer_components.py", BUNDLE / "components/sub1/src/vgec_sub1/candidates.py",
             BUNDLE / "components/sub1/src/vgec_sub1/models.py",
             BUNDLE / "components/s0/src/vgec/models/seq2seq.py",
             BUNDLE / "components/s0/src/vgec/inference/pipeline.py"]
    return dict(experiment="full-data-refit-05702-private-v1", source_sha256=digest(source_only(sources)),
                input_file_sha256=input_sha256, checkpoint_proofs=proofs, policy=POLICY,
                code_sha256={p.relative_to(ROOT).as_posix(): sha_file(p) for p in paths})


def cached_rows(directory, role, identity, sources):
    path = Path(directory) / f"{role}.json"
    if not path.exists():
        return None
    cached = read_json(path)
    if (cached.get("identity") != digest(identity) or cached.get("rows_sha256") != digest(cached.get("rows"))
            or len(cached["rows"]) != len(sources)):
        raise ValueError("Stale/corrupt inference cache: " + role)
    rows = align(sources, cached["rows"])
    for source, row in zip(sources, rows, strict=True):
        if row.get("Input_sha256") != sha_text(source["Input"]):
            raise ValueError("Cached Input hash mismatch")
    return rows


def infer_rows(role, sources, checkpoint):
    """Same inference numerics/decoding as the frozen 0.5702 component script."""
    import torch
    output = []
    if role in ("bami", "bart"):
        from vgec_sub1.config import load
        from vgec_sub1.models import Runtime
        cfg = load(BUNDLE / "components/sub1/configs/sub1.yaml")
        runtime = Runtime(cfg, role, training=False, checkpoint=Path(checkpoint), device=torch.device("cuda"))
        runtime.net.eval()
        if any(module.training for module in runtime.net.modules()):
            raise RuntimeError("Sub1 model/dropout remains in training mode")
        with torch.inference_mode():
            for start in range(0, len(sources), 32):
                part = sources[start:start+32]
                texts = [r["Input"] for r in part]
                if role == "bami":
                    encoded = [runtime.encode(dict(source=text, label=0, uid=str(start+i)))
                               for i, text in enumerate(texts)]
                    ids, mask = runtime.pad([r["input_ids"] for r in encoded])
                    specials, _ = runtime.pad([r["special_tokens_mask"] for r in encoded], value=1)
                    with runtime.autocast():
                        logits = runtime.net(ids, mask, specials)
                    for row, probs in zip(part, logits.float().double().softmax(-1).cpu().tolist(), strict=True):
                        valid_probs(probs)
                        output.append(dict(ID=row["ID"], Input_sha256=sha_text(row["Input"]),
                                           probabilities=probs, argmax=max(LABELS, key=lambda k: probs[k])))
                else:
                    runtime.generate(texts, two=True)
                    for row, trace in zip(part, runtime.last_generation_trace, strict=True):
                        chosen = select_bart(row["Input"], trace)
                        output.append(dict(ID=row["ID"], Input=row["Input"], Input_sha256=sha_text(row["Input"]),
                                           correction_checkpoint=Path(checkpoint).name,
                                           selected_text_before_postprocessing=chosen["text"],
                                           selection=chosen, generation_trace=trace))
                print(json.dumps(dict(role=role, source_done=len(output), total=len(sources))), flush=True)
    else:
        from vgec.models.seq2seq import load_checkpoint
        from vgec.inference.pipeline import encode
        device = torch.device("cuda")
        model, tokenizer, cfg = load_checkpoint(checkpoint, device)
        if cfg.get("classification_pooling") != "attention" or model.attention_pool is None:
            raise ValueError("S0 trained attention pooling missing")
        model.eval()
        if any(module.training for module in model.modules()):
            raise RuntimeError("S0 model/dropout remains in training mode")
        with torch.inference_mode():
            for start in range(0, len(sources), 8):
                part = sources[start:start+8]
                tensors = encode([r["Input"] for r in part], tokenizer, cfg, device)
                for row, probs in zip(part, model.predict_proba(tensors).cpu().tolist(), strict=True):
                    valid_probs(probs)
                    output.append(dict(ID=row["ID"], Input_sha256=sha_text(row["Input"]),
                                       probabilities=probs, argmax=max(LABELS, key=lambda k: probs[k])))
                print(json.dumps(dict(role=role, source_done=len(output), total=len(sources))), flush=True)
    return output


def assemble_predictions(sources, records, corrector_snapshot):
    sources = source_only(sources)
    bami, s0, bart = (align(sources, records[role]) for role in ROLES)
    internal, submission = [], []
    for source, b, s, c in zip(sources, bami, s0, bart, strict=True):
        text = source["Input"]
        bp, sp = probabilities(b, text), probabilities(s, text)
        if (c.get("Input") != text or c.get("Input_sha256") != sha_text(text)
                or c.get("correction_checkpoint") != corrector_snapshot):
            raise ValueError("Correction source/hash/checkpoint mismatch")
        chosen = select_bart(text, c["generation_trace"])
        if chosen != c.get("selection") or chosen["text"] != c.get("selected_text_before_postprocessing"):
            raise ValueError("Correction does not match first-valid generator selection")
        ep = [(bp[k] + sp[k]) / 2 for k in LABELS]
        label = max(LABELS, key=lambda k: ep[k])
        internal.append(dict(ID=source["ID"], Input=text, Input_sha256=sha_text(text),
                             bami_probabilities=bp, s0_probabilities=sp, ensemble_probabilities=ep,
                             predicted_label=label, selected_bart_text=chosen["text"], selection=chosen,
                             text_after_label0_policy=text if label == 0 else chosen["text"]))
        submission.append(dict(ID=source["ID"], Output={"Error_Type_Id": label,
                               "Fixed Sentence": [] if label == 0 else [chosen["text"]]}))
    return internal, submission


def export_private(directory, sources, records, identity):
    """CPU export; same identity may be replayed, a different run cannot overwrite it."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / "identity.json"
    if lock.exists() and read_json(lock) != identity:
        raise ValueError("Output directory belongs to different checkpoints/input/code")
    write_json(lock, identity)
    snapshot = Path(identity["checkpoint_proofs"]["bart"]["checkpoint"]).name
    internal, submission = assemble_predictions(sources, records, snapshot)
    for role in ROLES:
        write_json(directory / f"{role}.json", dict(identity=digest(identity), rows=records[role],
                                                   rows_sha256=digest(records[role])))
    for name, rows in (("predictions.jsonl", internal),):
        (directory / name).write_bytes(b"".join(json_bytes(row) for row in rows))
    result = directory / "results.json"
    result.write_bytes((json.dumps(submission, ensure_ascii=False, indent=2, allow_nan=False)+"\n").encode("utf-8"))
    archive_path = directory / "ensemble50_full_refit_bart_top1.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        info = zipfile.ZipInfo("results.json", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, result.read_bytes())
    with zipfile.ZipFile(archive_path) as archive:
        if archive.namelist() != ["results.json"] or archive.read("results.json") != result.read_bytes():
            raise ValueError("ZIP root/content mismatch")
    if [typed_id(r["ID"]) for r in submission] != [typed_id(r["ID"]) for r in sources]:
        raise ValueError("Submission typed ID/order/coverage mismatch")
    manifest = dict(identity=identity, rows=len(sources),
                    ensemble_labels_sha256=digest([[r["ID"], r["predicted_label"]] for r in internal]),
                    output_sha256={p.name: sha_file(p) for p in directory.iterdir()
                                   if p.is_file() and p.name != "manifest.json"},
                    leaderboard_submitted=False, dev_is_optimizer_exposed=True)
    write_json(directory / "manifest.json", manifest)
    return dict(rows=len(sources), results=str(result), zip=str(archive_path),
                results_sha256=sha_file(result), zip_sha256=sha_file(archive_path))
