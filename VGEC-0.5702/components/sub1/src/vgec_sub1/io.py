"""Strict JSON, streaming hashes, and local atomic publishing."""
import hashlib
import json
import os
import uuid
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON: '+value)))


def records(path):
    text = Path(path).read_text(encoding='utf-8-sig')
    def decode(value):
        return json.loads(value,parse_constant=lambda constant: (_ for _ in ()).throw(ValueError('Nonfinite JSON: '+constant)))
    values = decode(text) if text.lstrip().startswith('[') else [decode(line) for line in text.splitlines() if line.strip()]
    if not isinstance(values, list) or any(not isinstance(row, dict) for row in values):
        raise ValueError('Expected JSON array/JSONL objects')
    return values


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temp.open('w', encoding='utf-8', newline='\n') as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def write_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def write_jsonl(path, values):
    atomic_text(path, ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n' for row in values))


def code_hash(root):
    root = Path(root)
    paths = list((root/'src').rglob('*.py')) + list(root.glob('modal_*.py'))
    paths += list(root.glob('requirements-*.lock'))
    return digest({p.relative_to(root).as_posix(): file_hash(p) for p in sorted(paths)})
