"""Immutable checksummed stage/checkpoint publishing. No torch dependency."""
import json
import os
import uuid
from pathlib import Path
from ..io import file_hash, object_hash, read_records, write_json, write_jsonl


def directory_hash(path):
    path = Path(path)
    files = sorted(p for p in path.rglob('*') if p.is_file() and p.name != 'COMPLETED.json')
    if not files: raise ValueError(f'Empty artifact: {path}')
    return object_hash({str(p.relative_to(path)).replace('\\', '/'): file_hash(p) for p in files})


def publish_directory(root, identity, writer):
    root = Path(root)
    key = object_hash(identity)
    destination = root / key
    if destination.exists():
        verify_directory(destination, identity)
        return destination
    root.mkdir(parents=True, exist_ok=True)
    temp = root / ('.pending-' + uuid.uuid4().hex)
    temp.mkdir()
    writer(temp)
    write_json(temp / 'COMPLETED.json', dict(identity=identity, checksum=directory_hash(temp)))
    try:
        os.rename(temp, destination)
    except FileExistsError:
        verify_directory(destination, identity)
    return destination


def verify_directory(path, identity=None):
    path = Path(path)
    marker = json.loads((path / 'COMPLETED.json').read_text(encoding='utf-8'))
    if identity is not None and marker['identity'] != identity: raise ValueError('Artifact identity mismatch')
    if marker['checksum'] != directory_hash(path): raise ValueError('Incomplete/corrupt artifact checksum')
    return marker


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    write_json(temp, value)
    os.replace(temp, path)


def write_stage(root, identity, rows, metrics=None):
    def writer(temp):
        write_jsonl(temp/'rows.jsonl', rows)
        if metrics is not None: write_json(temp/'resource_metrics.json', metrics)
    path = publish_directory(root, identity, writer)
    return str(path)


def read_stage(path, identity=None):
    verify_directory(path, identity)
    return read_records(Path(path) / 'rows.jsonl')


def source_only(records, namespace):
    from ..schemas import adapt
    result = [adapt(dict(ID=r['ID'], Input=r['Input']), namespace, supervised=False) for r in records]
    for row in result: row['uid'] = f"{namespace}:{type(row['id']).__name__}:{row['id']}"
    ids = [(type(r['id']).__name__, str(r['id'])) for r in result]
    if len(set(ids)) != len(ids): raise ValueError('Duplicate input IDs')
    # Never serialize the optional training fields into the inference workers.
    return [{k: r[k] for k in ('uid', 'id', 'source_raw')} for r in result]


def merge_stages(rows, stage_rows):
    expected = {r['uid']: r for r in rows}
    found = {r['uid']: r for r in stage_rows}
    if len(found) != len(stage_rows) or set(found) != set(expected): raise ValueError('Stage UID coverage mismatch')
    for uid, row in found.items():
        if row['source_raw'] != expected[uid]['source_raw'] or type(row['id']) is not type(expected[uid]['id']) or row['id'] != expected[uid]['id']:
            raise ValueError('Stage source/ID mismatch')
    return [found[r['uid']] for r in rows]
