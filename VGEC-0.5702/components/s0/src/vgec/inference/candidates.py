"""Exact surface deduplication, stable identifiers, independent provenance."""
from copy import deepcopy
from ..io import object_hash
from .verifier import candidate


def make_candidate(row, text, producer, rank=0, truncated=False, provenance=None):
    value = candidate(row['source_raw'], text, producer, rank, truncated)
    value.update(uid=row['uid'], id=row['id'], source_raw=row['source_raw'], truncated=truncated,
                 candidate_id=object_hash([row['uid'], row['source_raw'], text]),
                 provenance=[dict(provenance or {}, producer=producer, rank=rank,
                                  valid=bool(value['valid'] and (not truncated or producer == 'copy')),
                                  flags=list(value['flags']), truncated=bool(truncated))])
    if producer == 'copy': value.update(valid=True, flags=[])
    return value


def deduplicate(pool, budget=None):
    if budget is not None and len(pool) > budget:
        raise ValueError(f'Candidate budget exceeded: {len(pool)} > {budget}')
    result = {}
    for item in pool:
        c = deepcopy(item)
        text = c['text_raw']
        producer = c['producer_id']
        provenance = c.get('provenance', [dict(producer=producer, rank=c.get('rank_in_producer', 0), valid=c['valid'])])
        if text not in result:
            c['agreeing_producers'] = sorted({p['producer'] for p in provenance if p.get('valid', c['valid'])})
            c['provenance'] = provenance
            result[text] = c
        else:
            old = result[text]
            merged = old['provenance'] + provenance
            producers = sorted(set(old['agreeing_producers']) | {p['producer'] for p in provenance if p.get('valid', c['valid'])})
            copy = old['is_copy'] or c['is_copy']
            if c['valid'] and not old['valid']: old = c
            old.update(provenance=merged, agreeing_producers=producers, is_copy=copy)
            result[text] = old
        result[text]['independent_agreement'] = len(set(result[text]['agreeing_producers']) - {'copy'})
    return list(result.values())
