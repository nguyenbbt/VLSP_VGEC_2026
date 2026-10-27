"""Read legacy immutable checkpoint pointers without the V2 adapter trainer."""
import json
from pathlib import Path
from ..inference.staged import verify_directory


def checkpoint_path(out, pointer, identity=None):
    out = Path(out)
    ref = json.loads((out / (pointer + '.json')).read_text(encoding='utf-8'))
    key = ref['key']
    if not isinstance(key, str) or len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
        raise ValueError('Invalid checkpoint pointer')
    path = out / 'checkpoints' / key
    marker = verify_directory(path)
    if identity is not None and marker['identity']['run_identity'] != identity:
        raise ValueError('Checkpoint belongs to another run')
    return path
