"""Read-only audit for a possible full-data refit of the 0.5702 system.

The supplied final set already contains dev. This script never concatenates
the two files or modifies an existing training run.
"""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import unicodedata


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "data_audit.json"


def file_sha(path):
    return sha256(path.read_bytes()).hexdigest()


def key(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def typed_id(value):
    if type(value) not in (int, str):
        raise ValueError(f"Invalid ID type: {type(value)}")
    return f"{type(value).__name__}:{value}"


def read_json(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Expected a list: {path}")
    return value


def index_raw(rows, name):
    by_key, ids = {}, set()
    for row in rows:
        tid = typed_id(row["ID"])
        if tid in ids:
            raise ValueError(f"Duplicate {name} typed ID: {tid}")
        ids.add(tid)
        source = row["Input"]
        label = row["Output"]["Error_Type_Id"]
        refs = row["Output"]["Fixed Sentence"]
        if not isinstance(source, str) or not source.strip() or type(label) is not int or label not in range(5):
            raise ValueError(f"Invalid {name} source or label: {tid}")
        if not isinstance(refs, list) or any(not isinstance(r, str) or not r.strip() for r in refs):
            raise ValueError(f"Invalid {name} references: {tid}")
        if label and not refs:
            raise ValueError(f"Missing correction reference: {tid}")
        source_key = key(source)
        if source_key in by_key:
            raise ValueError(f"Duplicate normalized {name} Input: {tid}")
        by_key[source_key] = row
    return by_key


def audit(root=ROOT):
    final_path = root / "the_final_data (1).json"
    dev_path = root / "dev.json"
    prepared_path = root / "data/prepared_final/train.jsonl"
    final = index_raw(read_json(final_path), "final")
    dev = index_raw(read_json(dev_path), "dev")
    prepared = {}
    for line in prepared_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        source_key = key(row["source_raw"])
        if source_key in prepared:
            raise ValueError("Duplicate prepared train Input")
        prepared[source_key] = row
    missing_dev = set(dev) - set(final)
    if missing_dev:
        raise ValueError(f"Dev Inputs missing from supplied final: {len(missing_dev)}")
    mismatches = []
    id_changes = []
    for source_key, dev_row in dev.items():
        final_row = final[source_key]
        if dev_row["Output"] != final_row["Output"] or dev_row["Input"] != final_row["Input"]:
            mismatches.append(source_key)
        if typed_id(dev_row["ID"]) != typed_id(final_row["ID"]):
            id_changes.append(source_key)
    if mismatches:
        raise ValueError(f"Dev/final Input or gold mismatches: {len(mismatches)}")
    expected_prepared = set(final) - set(dev)
    if set(prepared) != expected_prepared:
        raise ValueError(f"Prepared train coverage mismatch: missing={len(expected_prepared-set(prepared))}, extra={len(set(prepared)-expected_prepared)}")
    prepared_mismatch = [source_key for source_key, row in prepared.items()
                         if row["label"] != final[source_key]["Output"]["Error_Type_Id"]
                         or row["source_raw"] != final[source_key]["Input"]
                         or row["references_raw"] != final[source_key]["Output"]["Fixed Sentence"]]
    if prepared_mismatch:
        raise ValueError(f"Prepared train label/Input/reference mismatches: {len(prepared_mismatch)}")
    return {
        "status": "passed",
        "purpose": "full_data_refit_preflight_only",
        "input_sha256": {str(p.relative_to(root)): file_sha(p) for p in (final_path, dev_path, prepared_path)},
        "raw_final_rows": len(final),
        "current_prepared_train_rows": len(prepared),
        "dev_rows": len(dev),
        "dev_sources_already_in_raw_final": len(dev),
        "dev_rows_with_different_typed_id_in_final": len(id_changes),
        "full_refit_unique_sources": len(final),
        "label_support_full": dict(sorted(Counter(r["Output"]["Error_Type_Id"] for r in final.values()).items())),
        "label_support_current_train": dict(sorted(Counter(r["label"] for r in prepared.values()).items())),
        "label_support_dev": dict(sorted(Counter(r["Output"]["Error_Type_Id"] for r in dev.values()).items())),
        "data_action": "Use raw final once; do not append dev to it",
        "evaluation_limit": "Dev becomes optimizer-exposed in a full-data refit; the old dev score can no longer measure generalization",
    }


def main():
    result = audit()
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
