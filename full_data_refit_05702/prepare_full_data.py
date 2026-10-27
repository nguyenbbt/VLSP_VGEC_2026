"""Prepare the already-combined final data for the three 0.5702 components."""

import json
from pathlib import Path
import sys

from audit_data import ROOT, audit, file_sha, key, read_json

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "VGEC-0.5702/components/s0/src"))
sys.path.insert(0, str(ROOT / "VGEC-0.5702/components/sub1/src"))
from vgec.schemas import adapt  # noqa: E402
from vgec_sub1.data import _dedup, _view  # noqa: E402


def write_once(path, rows):
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != payload:
        raise ValueError(f"Existing prepared file differs: {path}")
    if not path.exists():
        path.write_text(payload, encoding="utf-8")


def main():
    preflight = audit(ROOT)
    raw = read_json(ROOT / "the_final_data (1).json")
    s0 = [adapt(row, "train") for row in raw]
    sub1 = _dedup(_view(raw, "final"))
    assert len(s0) == len(sub1) == preflight["full_refit_unique_sources"] == 7153
    assert {key(row["source_raw"]) for row in s0} == {key(row["source_raw"]) for row in sub1}
    bart = [row for row in sub1 if row["references"]]
    for name, rows in (("s0_full.jsonl", s0), ("sub1_full.jsonl", sub1), ("sub1_bart_full.jsonl", bart)):
        write_once(HERE / "data" / name, rows)
    output = dict(
        status="prepared",
        source_sha256=preflight["input_sha256"],
        prepared_sha256={name: file_sha(HERE / "data" / name) for name in ("s0_full.jsonl", "sub1_full.jsonl", "sub1_bart_full.jsonl")},
        counts=dict(s0=len(s0), bami=len(sub1), bart=len(bart)),
        dev_sources_in_optimizer=90,
        selection="fixed_historical_epoch; no validation on optimizer-exposed dev",
    )
    manifest = HERE / "prepared_manifest.json"
    content = json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    if manifest.exists() and manifest.read_text(encoding="utf-8") != content:
        raise ValueError("Existing prepared manifest differs")
    if not manifest.exists():
        manifest.write_text(content, encoding="utf-8")
    print(content)


if __name__ == "__main__":
    main()
