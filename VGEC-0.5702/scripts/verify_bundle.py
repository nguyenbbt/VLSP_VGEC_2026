"""Verify byte-identical S0/Sub1 source snapshots included in this code bundle."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    expected = manifest["files"]
    actual = {str(path.relative_to(ROOT)).replace("\\", "/") for path in (ROOT / "components").rglob("*")
              if path.is_file() and "__pycache__" not in path.parts
              and not any(part.endswith(".egg-info") for part in path.parts)
              and path.suffix != ".pyc"}
    if actual != set(expected):
        raise ValueError(f"Bundle file set differs: missing={set(expected)-actual}, extra={actual-set(expected)}")
    for relative, digest in expected.items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != digest:
            raise ValueError("Source snapshot changed: " + relative)
    print(json.dumps({"status": "verified", "source_files": len(expected)}))


if __name__ == "__main__":
    main()
