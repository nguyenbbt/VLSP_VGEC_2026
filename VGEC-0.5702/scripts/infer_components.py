"""Run one frozen component at a time from user-supplied checkpoints.

This script is provided for reproducibility. Importing it does not load a model;
running it does. It neither trains nor submits anything.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "components/sub1/src"))
sys.path.insert(0, str(ROOT / "components/s0/src"))
from assemble import SPEC, LABELS, load_json, sha_file, sha_text, typed_id, validate_sources


def check_checkpoint(role: str, checkpoint: Path):
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Missing checkpoint directory: {checkpoint}")
    if role in ("bami", "bart"):
        expected = SPEC["bami_classifier" if role == "bami" else "bart_corrector"]
        if checkpoint.name != expected["snapshot"]:
            raise ValueError(f"Wrong {role} snapshot directory")
        identity_path = checkpoint / "identity.json"
        if not identity_path.is_file() or load_json(identity_path).get("identity") != expected["identity"]:
            raise ValueError(f"Wrong {role} checkpoint identity")
        if not (checkpoint / "model.pt").is_file() or not (checkpoint / "tokenizer").is_dir():
            raise FileNotFoundError(f"Incomplete {role} checkpoint")
        for name, expected_hash in expected["file_sha256"].items():
            path = checkpoint / name
            if not path.is_file() or sha_file(path) != expected_hash:
                raise ValueError(f"Missing or mismatched {role} checkpoint file: {name}")
        return expected["identity"]
    expected = SPEC["s0_classifier"]
    cfg, manifest = checkpoint / "resolved_config.json", checkpoint / "run_manifest.json"
    if checkpoint.name != expected["checkpoint"] or checkpoint.parent.name != expected["run"]:
        raise ValueError("Wrong S0 run or checkpoint; best_classification is required")
    if not cfg.is_file() or not manifest.is_file():
        raise FileNotFoundError("Missing S0 checkpoint config/manifest")
    config = load_json(cfg)
    if config.get("kind") != "bartpho" or config.get("joint") is not True or config.get("classification_pooling") != "attention":
        raise ValueError("S0 checkpoint must have joint attention classifier")
    if load_json(manifest).get("identity_hash") != expected["identity"]:
        raise ValueError("Wrong S0 checkpoint identity")
    for name in ("model.safetensors", "classifier.pt", "attention_pool.pt", "sentencepiece.bpe.model", "dict.txt"):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError("S0 checkpoint missing " + name)
    for name, expected_hash in expected["file_sha256"].items():
        path = checkpoint / name
        if not path.is_file() or sha_file(path) != expected_hash:
            raise ValueError("Missing or mismatched S0 checkpoint file: " + name)
    return expected["identity"]


def select_bart(source: str, trace: list[dict]):
    """Preserve generator order, EOS and the original pool's format guard."""
    from vgec_sub1.candidates import pool

    if not isinstance(trace, list) or len(trace) != 2 or [t.get("beam") for t in trace] != [0, 1]:
        raise ValueError("Expected raw BART beam-0/beam-1 trace")
    items = []
    for attempt in trace:
        if attempt.get("mode") != "beam" or type(attempt.get("ended")) is not bool or type(attempt.get("forced_boundary")) is not bool:
            raise ValueError("Missing BART decoding/EOS evidence")
        text = attempt.get("text")
        if attempt["ended"] and not attempt["forced_boundary"] and isinstance(text, str) and text.strip():
            items.append({"text": text, "mode": "beam", "index": attempt["beam"]})
    candidates = pool(source, {"bart": items})
    valid = {c["text"] for c in candidates if any(p["family"] == "bart" for p in c["provenance"])}
    for index, attempt in enumerate(trace):
        if attempt.get("ended") and not attempt.get("forced_boundary") and attempt.get("text") in valid:
            return {"text": attempt["text"], "fallback": False, "trace_index": index,
                    "provenance": {"family": "bart", "mode": "beam", "index": attempt["beam"]}}
    return {"text": source, "fallback": True, "trace_index": None,
            "provenance": {"family": "copy", "reason": "no_valid_bart_output"}}


def valid_probs(probs):
    if len(probs) != 5 or any(type(p) not in (float, int) or not math.isfinite(p) or p < 0 for p in probs):
        raise ValueError("Invalid classifier probabilities")
    if abs(sum(probs) - 1) > 1e-4:
        raise ValueError("Classifier probabilities do not sum to 1")


def model_rows(role: str, raw: list[dict], checkpoint: Path, device: str):
    import torch

    if role in ("bami", "bart"):
        from vgec_sub1.config import load
        from vgec_sub1.models import Runtime

        cfg = load(ROOT / "components/sub1/configs/sub1.yaml")
        if cfg["_config_hash"] != SPEC["bami_classifier"]["config_hash"]:
            raise ValueError("Sub1 config differs from the frozen checkpoint recipe")
        model = Runtime(cfg, role, training=False, checkpoint=checkpoint, device=torch.device(device))
        model.net.eval()
        if any(module.training for module in model.net.modules()):
            raise ValueError("Dropout/model remains in training mode")
        output = []
        batch = 32  # Frozen A100 inference benchmark for both Sub1 components.
        with torch.inference_mode():
            for start in range(0, len(raw), batch):
                part = raw[start:start + batch]
                sources = [row["Input"] for row in part]
                if role == "bami":
                    encoded = [model.encode({"source": text, "label": 0, "uid": str(start + index)})
                               for index, text in enumerate(sources)]
                    ids, mask = model.pad([row["input_ids"] for row in encoded])
                    specials, _ = model.pad([row["special_tokens_mask"] for row in encoded], value=1)
                    with model.autocast():
                        logits = model.net(ids, mask, specials)
                    for row, probs in zip(part, logits.float().double().softmax(-1).cpu().tolist(), strict=True):
                        valid_probs(probs)
                        output.append({"ID": row["ID"], "Input_sha256": sha_text(row["Input"]),
                                       "probabilities": probs, "argmax": max(LABELS, key=lambda k: probs[k])})
                else:
                    model.generate(sources, two=True)
                    for row, trace in zip(part, model.last_generation_trace, strict=True):
                        chosen = select_bart(row["Input"], trace)
                        output.append({"ID": row["ID"], "Input": row["Input"],
                                       "Input_sha256": sha_text(row["Input"]),
                                       "correction_checkpoint": SPEC["bart_corrector"]["snapshot"],
                                       "selected_text_before_postprocessing": chosen["text"],
                                       "selection": chosen, "generation_trace": trace})
        meta = {"role": role, "checkpoint": role + "-final/" + checkpoint.name,
                "identity": SPEC["bami_classifier" if role == "bami" else "bart_corrector"]["identity"],
                "config_hash": cfg["_config_hash"], "model_eval": True, "inference_mode": True,
                "inference_batch": batch, "device": str(device), "label_order": LABELS,
                "softmax_temperature": 1}
        meta["tokenizer_sha256"] = {Path(name).name: expected_hash for name, expected_hash in
             SPEC["bami_classifier" if role == "bami" else "bart_corrector"]["file_sha256"].items()
             if name.startswith("tokenizer/")}
        return output, meta

    from vgec.models.seq2seq import load_checkpoint
    from vgec.inference.pipeline import encode

    torch_device = torch.device(device)
    model, tokenizer, cfg = load_checkpoint(checkpoint, torch_device)
    if cfg.get("classification_pooling") != "attention" or model.attention_pool is None:
        raise ValueError("S0 trained attention pooling was not loaded")
    model.eval()
    if any(module.training for module in model.modules()):
        raise ValueError("S0 dropout/model remains in training mode")
    output = []
    with torch.inference_mode():
        for start in range(0, len(raw), 8):
            part = raw[start:start + 8]
            tensors = encode([row["Input"] for row in part], tokenizer, cfg, torch_device)
            for row, probs in zip(part, model.predict_proba(tensors).cpu().tolist(), strict=True):
                valid_probs(probs)
                output.append({"ID": row["ID"], "Input_sha256": sha_text(row["Input"]),
                               "probabilities": probs, "argmax": max(LABELS, key=lambda k: probs[k])})
    meta = {"role": "s0", "checkpoint": SPEC["s0_classifier"]["run"] + "/best_classification",
            "identity": SPEC["s0_classifier"]["identity"],
            "config_sha256": sha_file(checkpoint / "resolved_config.json"),
            "head_sha256": sha_file(checkpoint / "classifier.pt"),
            "attention_pool_sha256": sha_file(checkpoint / "attention_pool.pt"),
            "tokenizer_sha256": {name: sha_file(checkpoint / name)
                                  for name in ("sentencepiece.bpe.model", "dict.txt", "tokenizer_config.json")},
            "model_eval": True, "inference_mode": True, "inference_batch": 8,
            "device": str(device), "label_order": LABELS, "softmax_temperature": 1}
    return output, meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--role", required=True, choices=("bami", "s0", "bart"))
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    raw = load_json(args.input)
    validate_sources(raw)
    identity = check_checkpoint(args.role, args.checkpoint)
    generated, metadata = model_rows(args.role, raw, args.checkpoint, args.device)
    if [typed_id(row["ID"]) for row in generated] != [typed_id(row["ID"]) for row in raw]:
        raise AssertionError("Model output coverage/order mismatch")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in generated), encoding="utf-8")
    metadata.update({"input_file_sha256": sha_file(args.input), "rows": len(raw),
                     "checkpoint_identity": identity, "output_sha256": sha_file(args.output)})
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"role": args.role, "rows": len(raw), "output": str(args.output),
                      "output_sha256": metadata["output_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
