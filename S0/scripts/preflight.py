"""Validate local S0 data and splits without starting Modal."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vgec.training.routing import preflight

if __name__ == '__main__':
    try:
        print(json.dumps(preflight(ROOT), ensure_ascii=False, indent=2))
    except (FileNotFoundError, ValueError) as exc:
        print(f'Preflight failed: {exc}', file=sys.stderr)
        raise SystemExit(1)
