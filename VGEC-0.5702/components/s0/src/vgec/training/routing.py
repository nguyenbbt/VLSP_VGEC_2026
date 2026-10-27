"""S0-only jobs and data preflight; no GPU or network required."""
import re
from pathlib import Path
from ..io import file_hash, load_config, read_records


MODELS = {'bartpho_attention', 'vit5'}
S0_STEPS = (
    ('bartpho_attention', 's0-bartpho-attention-smoke-42', 20),
    ('vit5', 's0-vit5-base-smoke-42', 20),
    ('bartpho_attention', 's0-bartpho-attention-full-42', 0),
    ('vit5', 's0-vit5-base-full-42', 0),
)


def validate_run_id(run_id):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', run_id):
        raise ValueError('run-id must be a simple directory name')
    return run_id


def resolve_job(model, root):
    root = Path(root)
    if model not in MODELS:
        raise ValueError('S0 supports bartpho_attention and vit5 only')
    config = root / 'configs/models' / (model + '.yaml')
    cfg = load_config(config)
    if model == 'bartpho_attention':
        if (cfg['model_id'] != 'vinai/bartpho-syllable' or not cfg['joint']
                or cfg.get('classification_pooling') != 'attention'):
            raise ValueError('S0 requires joint BARTpho with attention classification pooling')
    elif cfg['model_id'] != 'VietAI/vit5-base' or cfg['joint']:
        raise ValueError('S0 requires correction-only ViT5-base')
    data_cfg = load_config(root / 'configs/data.yaml')
    return config, root / data_cfg['prepared_dir'], cfg


def verify_data(root):
    root = Path(root)
    cfg = load_config(root / 'configs/data.yaml')
    prepared = root / cfg['prepared_dir']
    report_path = root / cfg['report_dir'] / 'data_audit.json'
    checks = {
        ('raw_hashes', 'train'): root / cfg['train'],
        ('raw_hashes', 'dev'): root / cfg['dev'],
        ('prepared_hashes', 'train'): prepared / 'train.jsonl',
        ('prepared_hashes', 'dev'): prepared / 'dev.jsonl',
    }
    missing = [str(p) for p in [report_path, *checks.values()] if not p.is_file()]
    if missing:
        raise FileNotFoundError('Place your own data and run python -m vgec audit --config configs/data.yaml. Missing: ' + ', '.join(missing))
    report = load_config(report_path)
    for (section, key), path in checks.items():
        if file_hash(path) != report[section][key]:
            raise ValueError(f'Data changed since audit: {path}')
    train, dev = read_records(prepared / 'train.jsonl'), read_records(prepared / 'dev.jsonl')
    if not train or not dev:
        raise ValueError('Train and dev must both be nonempty')
    if len(train) != report['prepared_train'] or len(dev) != report['dev_records']:
        raise ValueError('Audit row counts do not match prepared data')
    if {r['group_id'] for r in train} & {r['group_id'] for r in dev}:
        raise ValueError('Train/dev group overlap')
    return report


def preflight(root):
    from ..data.split import inner_split
    report = verify_data(root)
    splits = {}
    for name in sorted(MODELS):
        _, prepared, cfg = resolve_job(name, root)
        rows = read_records(prepared / 'train.jsonl')
        if not cfg['joint']:
            rows = [r for r in rows if r['correction_references']]
        train, val = inner_split(rows, cfg['seed'])
        splits[name] = dict(train=len(train), validation=len(val))
    return dict(status='passed', gpu_started=False, raw_train=report['original_train'],
                prepared_train=report['prepared_train'], dev=report['dev_records'], splits=splits)


def run_sequence(invoke):
    results = {}
    for model, run_id, steps in S0_STEPS:
        print(f'Starting {run_id}', flush=True)
        result = invoke(model, run_id, steps, True)
        if result.get('status') != 'finished' or bool(result.get('smoke_only')) != bool(steps):
            raise RuntimeError(f'Unexpected training result for {run_id}')
        if steps and result.get('optimizer_steps') != steps:
            raise RuntimeError(f'Incomplete smoke for {run_id}')
        results[run_id] = result
    return results
