"""Assemble the fixed 0.5702 pipeline from source-only model outputs.

No model, gold label, reference, reranker or private-test target is loaded here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / "checkpoints.json").read_text(encoding="utf-8"))
LABELS = SPEC["label_order"]


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def typed_id(value):
    if type(value) not in (int, str):
        raise ValueError("ID must be an int or string, never bool")
    return type(value).__name__, value


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def validate_sources(raw: list[dict]) -> None:
    if not isinstance(raw, list) or not raw:
        raise ValueError("Input must be a nonempty JSON list")
    seen = set()
    for row in raw:
        if not isinstance(row, dict) or "ID" not in row or not isinstance(row.get("Input"), str) or not row["Input"].strip():
            raise ValueError("Each input needs typed ID and nonempty Input")
        key = typed_id(row["ID"])
        if key in seen:
            raise ValueError("Duplicate typed input ID")
        seen.add(key)


def align(raw: list[dict], records: list[dict]) -> list[dict]:
    indexed = {}
    for row in records:
        key = typed_id(row["ID"])
        if key in indexed:
            raise ValueError("Duplicate prediction ID")
        indexed[key] = row
    keys = [typed_id(row["ID"]) for row in raw]
    if set(indexed) != set(keys):
        raise ValueError("Missing or extra prediction ID")
    return [indexed[key] for key in keys]


def validate_metadata(path: Path, role: str, input_path: Path, count: int):
    meta_path = path.with_suffix(".metadata.json")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing {role} probability metadata: {meta_path}")
    meta = load_json(meta_path)
    expected = SPEC["bami_classifier" if role == "bami" else "s0_classifier"]
    if meta.get("role") != role or meta.get("rows") != count or meta.get("label_order") != LABELS:
        raise ValueError(f"{role} metadata role/count/label order mismatch")
    if meta.get("input_file_sha256") != sha_file(input_path):
        raise ValueError(f"{role} inference input hash mismatch")
    if meta.get("identity") != expected["identity"]:
        raise ValueError(f"{role} checkpoint identity mismatch")
    checkpoint = str(meta.get("checkpoint", ""))
    if role == "bami":
        if not checkpoint.endswith("/" + expected["snapshot"]) and checkpoint != expected["snapshot"]:
            raise ValueError("Wrong Bami snapshot")
        if meta.get("config_hash") != expected["config_hash"] or meta.get("inference_batch") != 32:
            raise ValueError("Wrong Bami config or inference batch")
    elif not checkpoint.endswith(expected["run"] + "/" + expected["checkpoint"]):
        raise ValueError("Wrong S0 best_classification checkpoint")
    if meta.get("model_eval") is not True or meta.get("inference_mode") is not True or meta.get("softmax_temperature") != 1:
        raise ValueError(f"{role} was not run in the recorded inference mode")
    return meta, meta_path


def probabilities(row: dict, source: str) -> list[float]:
    values = row.get("probabilities")
    if row.get("Input_sha256") != sha_text(source) or not isinstance(values, list) or len(values) != 5:
        raise ValueError("Input hash or five-label probability mapping mismatch")
    if any(type(v) not in (float, int) or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Invalid classifier probability")
    if abs(sum(values) - 1) > 1e-4:
        raise ValueError("Classifier probabilities do not sum to 1")
    argmax = max(LABELS, key=lambda k: values[k])
    if type(row.get("argmax")) is not int or row["argmax"] != argmax:
        raise ValueError("Saved argmax differs from probabilities")
    return values


def correction(row: dict, source: str) -> tuple[str, bool]:
    if row.get("correction_checkpoint") != SPEC["bart_corrector"]["snapshot"]:
        raise ValueError("Wrong correction-only BARTpho snapshot")
    if row.get("Input", source) != source or row.get("Input_sha256", sha_text(source)) != sha_text(source):
        raise ValueError("BART correction source mismatch")
    text = row.get("selected_text_before_postprocessing", row.get("selected_text"))
    selection = row.get("selection")
    if not isinstance(text, str) or not text.strip() or not isinstance(selection, dict) or type(selection.get("fallback")) is not bool:
        raise ValueError("Missing internal BART selection; exported [] is insufficient")
    if selection.get("text") != text:
        raise ValueError("Selected text differs from BART trace decision")
    family = selection.get("provenance", {}).get("family")
    if selection["fallback"]:
        if text != source or family != "copy":
            raise ValueError("BART fallback must Copy raw Input")
    elif family != "bart":
        raise ValueError("Corrector must be the new BARTpho generator")
    return text, selection["fallback"]


def assemble(raw: list[dict], bami: list[dict], s0: list[dict], bart: list[dict]):
    validate_sources(raw)
    bami, s0, bart = (align(raw, records) for records in (bami, s0, bart))
    internal, submission = [], []
    for source, b, s, c in zip(raw, bami, s0, bart, strict=True):
        bp, sp = probabilities(b, source["Input"]), probabilities(s, source["Input"])
        text, fallback = correction(c, source["Input"])
        ep = [(bp[k] + sp[k]) / 2 for k in LABELS]
        label = max(LABELS, key=lambda k: ep[k])
        internal.append({"ID": source["ID"], "Input": source["Input"],
                         "Input_sha256": sha_text(source["Input"]),
                         "bami_probabilities": bp, "s0_probabilities": sp,
                         "ensemble_probabilities": ep, "predicted_label": label,
                         "selected_bart_text": text, "selection_fallback": fallback,
                         "text_after_label0_policy": source["Input"] if label == 0 else text})
        submission.append({"ID": source["ID"], "Output": {"Error_Type_Id": label,
                           "Fixed Sentence": [] if label == 0 else [text]}})
    if [typed_id(row["ID"]) for row in submission] != [typed_id(row["ID"]) for row in raw]:
        raise AssertionError("Output ID order changed")
    return internal, submission


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--bami-probs", required=True, type=Path)
    p.add_argument("--s0-probs", required=True, type=Path)
    p.add_argument("--bart-predictions", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    args = p.parse_args()
    raw = load_json(args.input)
    bami = load_jsonl(args.bami_probs)
    s0 = load_jsonl(args.s0_probs)
    bart = load_jsonl(args.bart_predictions)
    bmeta, bmeta_path = validate_metadata(args.bami_probs, "bami", args.input, len(raw))
    smeta, smeta_path = validate_metadata(args.s0_probs, "s0", args.input, len(raw))
    internal, submission = assemble(raw, bami, s0, bart)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    internal_path = args.output_dir / "internal.jsonl"
    results_path = args.output_dir / "results.json"
    zip_path = args.output_dir / "results.zip"
    internal_path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in internal), encoding="utf-8")
    results_path.write_text(json.dumps(submission, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(results_path, arcname="results.json")
    with zipfile.ZipFile(zip_path) as archive:
        if archive.namelist() != ["results.json"] or archive.read("results.json") != results_path.read_bytes():
            raise AssertionError("ZIP is not the exact root results.json")
    manifest = {"configuration": SPEC, "rows": len(raw),
                "input_sha256": sha_file(args.input),
                "source_hashes": {name: sha_file(path) for name, path in {
                    "bami_probabilities": args.bami_probs, "s0_probabilities": args.s0_probs,
                    "bart_predictions": args.bart_predictions,
                    "bami_metadata": bmeta_path, "s0_metadata": smeta_path}.items()},
                "checkpoint_identities": {"bami": bmeta["identity"], "s0": smeta["identity"]},
                "results_sha256": sha_file(results_path), "zip_sha256": sha_file(zip_path),
                "no_gold_or_reranker_used": True, "leaderboard_submitted": False}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(raw), "results": str(results_path), "zip": str(zip_path),
                      "results_sha256": manifest["results_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
