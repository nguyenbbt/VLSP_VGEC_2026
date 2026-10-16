from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from ..io import read_records, write_json, write_jsonl, file_hash
from ..schemas import adapt, source_key
from ..evaluation.diagnostics import evaluate


def audit(config_path):
    from ..io import load_config
    cfg = load_config(config_path)
    train = [adapt(r, "train") for r in read_records(cfg["train"])]
    dev = [adapt(r, "dev") for r in read_records(cfg["dev"])]
    for name, rows in (("train", train), ("dev", dev)):
        if len({r["uid"] for r in rows}) != len(rows):
            raise ValueError(f"Duplicate IDs in {name}")
    dev_keys = {source_key(r["source_raw"]) for r in dev}
    dev_refs = {source_key(ref) for r in dev for ref in r["references_raw"]}
    excluded, kept, seen = [], [], {}
    for row in train:
        key = source_key(row["source_raw"])
        reason = None
        if key in dev_keys:
            reason = "source_in_dev"
        elif key in dev_refs or any(source_key(ref) in dev_keys | dev_refs for ref in row["references_raw"]):
            reason = "reference_family_overlap_dev"
        elif key in seen:
            old = seen[key]
            if old["label"] != row["label"]:
                raise ValueError(f"Conflicting duplicate labels: {old['uid']}, {row['uid']}")
            old["references_raw"] = list(dict.fromkeys(old["references_raw"] + row["references_raw"]))
            old["correction_references"] = list(dict.fromkeys(old["correction_references"] + row["correction_references"]))
            reason = "duplicate_source_merged"
        if reason:
            excluded.append(dict(uid=row["uid"], reason=reason))
        else:
            kept.append(row)
            seen[key] = row
    # Near overlap is advisory: lexical similarity alone does not establish ancestry.
    near = []
    for row in kept:
        key = source_key(row["source_raw"])
        for other in dev:
            other_key = source_key(other["source_raw"])
            if min(len(key), len(other_key)) / max(len(key), len(other_key)) < .9:
                continue
            score = SequenceMatcher(None, key, other_key).ratio()
            if score >= .92:
                near.append(dict(train_uid=row["uid"], dev_uid=other["uid"], similarity=score))
    out = Path(cfg["prepared_dir"])
    out.mkdir(parents=True, exist_ok=True)
    report_dir = Path(cfg.get("report_dir", "reports"))
    write_jsonl(out / "train.jsonl", kept)
    write_jsonl(out / "dev.jsonl", dev)
    copy = [dict(uid=r["uid"], id=r["id"], source_raw=r["source_raw"], predicted_label=0,
                 selected_text=r["source_raw"], candidates=[]) for r in dev]
    baseline = evaluate(copy, dev)
    report = dict(original_train=len(train), expected_train=cfg["expected_train"], dev_records=len(dev),
                  prepared_train=len(kept), exclusions_by_reason=dict(Counter(r["reason"] for r in excluded)),
                  excluded_records=excluded, near_overlaps=near,
                  dev_label_counts=dict(Counter(r["label"] for r in dev)),
                  train_label_counts=dict(Counter(r["label"] for r in kept)),
                  dev_incomplete_reference_records=sum("incomplete_reference" in r["flags"] for r in dev),
                  empty_no_error_reference_records=dict(train=sum(not r["references_raw"] for r in train),dev=sum(not r["references_raw"] for r in dev)),
                  train_without_complete_correction_target=sum(not r["correction_references"] for r in kept),
                  raw_hashes={"train": file_hash(cfg["train"]), "dev": file_hash(cfg["dev"])},
                  prepared_hashes={"train": file_hash(out / "train.jsonl"), "dev": file_hash(out / "dev.jsonl")},
                  lineage_status="unverified_user_authorized_experimental",
                  lineage_limit="Exact filtering cannot prove absence of synthetic dev descendants or dev prompt exposure",
                  constrained_eligible=False, copy_baseline=baseline)
    write_json(report_dir / "data_audit.json", report)
    write_json(report_dir / "copy_baseline.json", baseline)
    write_jsonl(report_dir / "copy_predictions.jsonl", copy)
    return report
