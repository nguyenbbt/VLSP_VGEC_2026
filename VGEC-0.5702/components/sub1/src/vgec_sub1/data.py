"""Preserve raw data; join views by source, keep dev and known ancestry out."""
import random
import re
import unicodedata
from collections import Counter, defaultdict
from .io import digest, file_hash, records, read_json, write_json, write_jsonl
from .config import path, output, role_config


def source_key(text):
    return digest(' '.join(unicodedata.normalize('NFC', text).split()))


def validate_record(row, supervised=True):
    if not isinstance(row, dict) or 'ID' not in row:
        raise ValueError('Missing ID')
    if type(row['ID']) not in (int, str):
        raise ValueError('ID must be int/string, not bool')
    if not isinstance(row.get('Input'), str) or not row['Input'].strip():
        raise ValueError('Input must be a nonempty string')
    parents=row.get('parent_source_keys',[])
    if not isinstance(parents,list) or any(not isinstance(p,str) or not re.fullmatch('[a-f0-9]{64}',p) for p in parents):
        raise ValueError('parent_source_keys must be a list of canonical 64-character source hashes')
    if supervised:
        out = row.get('Output')
        if not isinstance(out, dict) or type(out.get('Error_Type_Id')) is not int or out['Error_Type_Id'] not in range(5):
            raise ValueError('Invalid error label')
        refs = out.get('Fixed Sentence')
        if not isinstance(refs, list) or any(not isinstance(r, str) or not r.strip() for r in refs):
            raise ValueError('Invalid reference list')
        if out['Error_Type_Id'] and not refs:
            raise ValueError('Nonzero label requires raw references')


def typed_id(value):
    if type(value) not in (int, str):
        raise ValueError('Invalid ID type')
    return type(value).__name__+':'+str(value)


def source_rows(values, namespace):
    result, seen = [], set()
    for index, r in enumerate(values):
        validate_record(r, False)
        key = typed_id(r['ID'])
        if key in seen:
            raise ValueError('Duplicate input ID: '+key)
        seen.add(key)
        result.append(dict(uid=namespace+':'+key, ID=r['ID'], id=r['ID'], row_index=index, source=r['Input'], source_raw=r['Input']))
    if not result:
        raise ValueError('Empty input')
    return result


def _view(values, namespace):
    seen, result = set(), []
    for index, row in enumerate(values):
        validate_record(row)
        tid = typed_id(row['ID'])
        if tid in seen:
            raise ValueError('Duplicate training ID: '+namespace+':'+tid)
        seen.add(tid)
        label = row['Output']['Error_Type_Id']
        raw_refs = row['Output']['Fixed Sentence']
        refs, flags = [], []
        if label == 0:
            refs = [row['Input']]
        else:
            for ref in raw_refs:
                if re.fullmatch(r'\s*[\(\[].*[\)\]]\s*', ref, re.S):
                    flags.append('comment_reference_needs_review')
                elif ('...' in ref or '…' in ref) and (not ('...' in row['Input'] or '…' in row['Input']) or len(ref) < .75*len(row['Input'])):
                    flags.append('abbreviated_reference_needs_review')
                else:
                    refs.append(ref)
        key = source_key(row['Input'])
        result.append(dict(uid=key, id=row['ID'], row_index=index, source=row['Input'], source_raw=row['Input'],
                           source_key=key, label=label, references_raw=raw_refs,
                           references=list(dict.fromkeys(refs)), group_id=key, weight=1.,
                           parent_source_keys=list(row.get('parent_source_keys', [])),
                           origin=namespace, flags=flags,
                           original_ids=[dict(namespace=namespace, id=row['ID'], row_index=index)]))
    return result


def _dedup(rows):
    by_key = {}
    for row in rows:
        key = row['source_key']
        if key not in by_key:
            by_key[key] = row
            continue
        old = by_key[key]
        if old['label'] != row['label']:
            raise ValueError('Conflicting labels within one source view: '+key)
        old['references_raw'] = list(dict.fromkeys(old['references_raw']+row['references_raw']))
        old['references'] = list(dict.fromkeys(old['references']+row['references']))
        old['flags'] = sorted(set(old['flags']+row['flags']))
        old['original_ids'].extend(row['original_ids'])
        old['parent_source_keys'] = sorted(set(old['parent_source_keys']+row['parent_source_keys']))
    return list(by_key.values())


def partition(rows, fractions, seed, gold_keys=()):
    """Greedy grouped label/gold-support stratification; no group is split."""
    if not rows or abs(sum(fractions)-1) > 1e-8 or any(f <= 0 for f in fractions):
        raise ValueError('Invalid nonempty partition request')
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['group_id']].append(row)
    if len(grouped) < len(fractions):
        raise ValueError('Insufficient source groups')
    groups = list(grouped.values())
    random.Random(seed).shuffle(groups)
    groups.sort(key=len, reverse=True)
    gold_keys = set(gold_keys)
    label_counts = Counter((r['label'], r['uid'] in gold_keys) for r in rows)
    target = [{k: max(1., n*f) for k, n in label_counts.items()} for f in fractions]
    counts = [Counter() for _ in fractions]
    buckets = [[] for _ in fractions]
    for group in groups:
        features = Counter((r['label'], r['uid'] in gold_keys) for r in group)
        slot = min(range(len(fractions)), key=lambda i: (
            sum(features[k]*counts[i][k]/target[i][k] for k in features),
            len(buckets[i])/max(1., len(rows)*fractions[i]), i))
        buckets[slot].extend(group)
        counts[slot].update(features)
    if any(not b for b in buckets):
        raise ValueError('Empty partition')
    return buckets


def disjoint(*parts):
    seen = set()
    for rows in parts:
        groups = {r['group_id'] for r in rows}
        if groups & seen:
            raise ValueError('Group leakage')
        seen |= groups


def prepare(cfg):
    base = path(cfg, cfg['data']['root'])
    names = dict(final=cfg['data']['supplied_final'], gold=cfg['data']['gold'], dev=cfg['data']['dev'])
    views = {name: _dedup(_view(records(base/filename), name)) for name, filename in names.items()}
    if any(not rows for rows in views.values()):
        raise ValueError('Final, processed gold, and dev must be nonempty')
    # A union of known source/parent connections includes dev before excluding families.
    all_keys = {r['uid'] for rows in views.values() for r in rows}
    parents = {k: k for k in all_keys}
    def find(k):
        parents.setdefault(k, k)
        if parents[k] != k:
            parents[k] = find(parents[k])
        return parents[k]
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parents[max(a,b)] = min(a,b)
    for rows in views.values():
        for r in rows:
            for parent in r['parent_source_keys']:
                union(r['uid'], parent)
    for rows in views.values():
        for r in rows:
            r['group_id'] = find(r['uid'])
    dev_groups = {r['group_id'] for r in views['dev']}
    dev_family_refs = {source_key(ref) for r in views['dev'] for ref in r['references_raw']}
    dev_family_refs.update(r['source_key'] for r in views['dev'])
    # Reference overlap excludes the whole connected source/parent family,
    # including siblings that do not individually contain the dev reference.
    dev_groups.update(r['group_id'] for name in ('final','gold') for r in views[name]
                      if r['source_key'] in dev_family_refs or any(source_key(ref) in dev_family_refs for ref in r['references_raw']))
    excluded = []
    for name in ('final', 'gold'):
        kept = []
        for r in views[name]:
            overlap = r['group_id'] in dev_groups or r['source_key'] in dev_family_refs or any(source_key(ref) in dev_family_refs for ref in r['references_raw'])
            if overlap:
                excluded.append(dict(view=name, uid=r['uid'], reason='dev_source_or_known_family'))
            else:
                kept.append(r)
        views[name] = kept
    master = {r['uid']: r for r in views['final']}
    for row in views['gold']:
        master.setdefault(row['uid'], row)
    universe = list(master.values())
    gold_keys = {r['uid'] for r in views['gold']}
    split_seed=cfg['data']['fold_seed']; final_fraction=cfg['data']['final_inner_fraction']
    folds = partition(universe, [1/3]*3, split_seed, gold_keys)
    final_fit, final_inner = partition(universe, [1-final_fraction,final_fraction], split_seed+100, gold_keys)
    jobs = {'final': dict(fit=[r['uid'] for r in final_fit], inner=[r['uid'] for r in final_inner])}
    for k in range(3):
        outer = [r for i, fold in enumerate(folds) if i != k for r in fold]
        fraction=cfg['data']['inner_validation_fraction']
        fit, inner = partition(outer, [1-fraction,fraction], split_seed+k+200, gold_keys)
        disjoint(fit, inner, folds[k], views['dev'])
        jobs[str(k)] = dict(fit=[r['uid'] for r in fit], inner=[r['uid'] for r in inner], heldout=[r['uid'] for r in folds[k]])
    disjoint(final_fit, final_inner, views['dev'])
    fraction=cfg['models']['reranker']['rank_fit_fraction']
    rank_fit, rank_val = partition(universe, [fraction,1-fraction], split_seed+300, gold_keys)
    manifest = dict(schema_version=1, config_hash=cfg['_config_hash'],
                    raw_hashes={name: file_hash(base/file) for name,file in names.items()},
                    folds=jobs, rank_split=dict(fit=[r['uid'] for r in rank_fit], inner=[r['uid'] for r in rank_val]),
                    known_group_exclusion_passed=True, all_ancestry_verified=False,
                    constrained_eligibility='unverified', excluded=excluded,
                    counts={name: len(rows) for name,rows in views.items()},
                    labels={name: dict(Counter(r['label'] for r in rows)) for name,rows in views.items()})
    manifest['data_identity'] = digest(dict(raw=manifest['raw_hashes'], jobs=jobs,
                                           rank=manifest['rank_split'], config=cfg['_config_hash']))
    dest = path(cfg, cfg['data']['prepared_dir'])
    for name, rows in views.items():
        write_jsonl(dest/(name+'.jsonl'), rows)
    write_jsonl(dest/'universe.jsonl', universe)
    manifest['prepared_hashes'] = {name: file_hash(dest/(name+'.jsonl')) for name in (*views, 'universe')}
    # Every role-specific validation split must actually contain usable targets.
    for fold in jobs:
        for role in ('bami','bart','vit5','qwen_corrector'):
            select_training(cfg, role, fold, manifest=manifest)
    write_json(dest/'manifest.json', manifest)
    write_json(output(cfg)/'reports/data_audit.json', manifest)
    write_jsonl(output(cfg)/'reports/reference_issues.jsonl', [r for rows in views.values() for r in rows if r['flags']])
    return manifest


def load_manifest(cfg):
    dest = path(cfg, cfg['data']['prepared_dir'])
    manifest = read_json(dest/'manifest.json')
    if manifest['config_hash'] != cfg['_config_hash']:
        raise ValueError('Config changed; audit/splits must be regenerated before launch')
    for name, filename in dict(final=cfg['data']['supplied_final'], gold=cfg['data']['gold'], dev=cfg['data']['dev']).items():
        if file_hash(path(cfg,cfg['data']['root'])/filename) != manifest['raw_hashes'][name]:
            raise ValueError('Raw data changed since audit: '+name)
    for name, expected in manifest['prepared_hashes'].items():
        if file_hash(dest/(name+'.jsonl')) != expected:
            raise ValueError('Prepared data changed since audit: '+name)
    return manifest


def select_training(cfg, role, fold='final', manifest=None):
    manifest = manifest or load_manifest(cfg)
    view = 'gold' if role == 'vit5' else 'final'
    rows = records(path(cfg,cfg['data']['prepared_dir'])/(view+'.jsonl'))
    job = manifest['folds'][str(fold)]
    parts = []
    for part in ('fit','inner'):
        ids = set(job[part])
        selected = [r for r in rows if r['uid'] in ids and (role=='bami' or r['references'])]
        if not selected:
            raise ValueError(f'No usable {role}/{fold}/{part} data')
        parts.append(selected)
    disjoint(*parts)
    return tuple(parts)


def execution_plan(cfg):
    manifest = load_manifest(cfg)
    runs = []
    for fold in ('final', '0','1','2'):
        for role in ('bami','bart','vit5','qwen_corrector'):
            if role == 'bami' and fold != 'final':
                continue
            fit, inner = select_training(cfg, role, fold, manifest)
            rc = role_config(cfg, role)
            runs.append(dict(run_id=f'{role}-{fold}', role=role, fold=fold, fit_sources=len(fit),
                             validation_sources=len(inner), max_epochs=rc['epochs'], global_batch=rc['global_batch'],
                             optimizer_steps_upper_bound=((len(fit)+rc['global_batch']-1)//rc['global_batch'])*rc['epochs'],
                             fit_hash=digest(fit), selection_hash=digest(inner)))
    runs.append(dict(run_id='reranker-final', role='reranker', fold='final', status='requires_audited_preferences'))
    result = dict(status='planned', full_finetunes=len(runs), runs=runs, data_identity=manifest['data_identity'],
                  inherited_sub2_runtime_verified=False, eligibility='unverified_exploratory')
    write_json(output(cfg)/'execution_plan.json', result)
    return result
