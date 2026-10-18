"""Human audit import. Unknown judgments never become negative examples."""
from .config import output,path
from .data import load_manifest
from .io import records,read_json,write_json,write_jsonl,digest,file_hash


def load_oof(cfg):
    manifest=load_manifest(cfg); dest=output(cfg)/'oof'
    proof=read_json(dest/'proof.json')
    if proof['data_identity']!=manifest['data_identity'] or proof['hash']!=file_hash(dest/'pools.jsonl'):
        raise ValueError('OOF proof/data/checksum mismatch')
    from .training import identity
    from .data import select_training
    if sorted(str(item['fold']) for item in proof['folds'])!=['0','1','2']:
        raise ValueError('OOF proof must cover each of the three folds exactly once')
    for item in proof['folds']:
        fold=str(item['fold']); heldout=set(manifest['folds'][fold]['heldout'])
        if item['heldout_hash']!=digest(sorted(heldout)): raise ValueError('OOF heldout proof changed')
        for role in ('bart','vit5','qwen_corrector'):
            fit,inner=select_training(cfg,role,fold,manifest)
            if item['model_identities'][role]!=identity(cfg,role,fold,fit,inner): raise ValueError('OOF model recipe/split/code mismatch')
    rows=records(dest/'pools.jsonl')
    expected={uid for k in ('0','1','2') for uid in manifest['folds'][k]['heldout']}
    if len(rows)!=len(expected) or {r['uid'] for r in rows}!=expected: raise ValueError('OOF coverage mismatch')
    for row in rows:
        if row['uid'] not in set(manifest['folds'][str(row['fold'])]['heldout']): raise ValueError('Candidate was not generated out of fold')
    return rows


def audit_template(cfg):
    oof=load_oof(cfg)
    rows=[]
    for r in oof:
        rows.append(dict(uid=r['uid'],source=r['source'],pool_hash=digest(r['candidates']),
                         candidates=[dict(text=c['text'],grammar=None,meaning=None,unnecessary=None,comment='') for c in r['candidates']],
                         ordered_pairs=[],annotator='',audited=False))
    dest=output(cfg)/'preferences/audit_template.jsonl'; write_jsonl(dest,rows)
    manifest=load_manifest(cfg); by_uid={r['uid']:r for r in oof}; sample=[]
    for split,count in (('fit',32),('inner',8)):
        ids=set(manifest['rank_split'][split])
        for label in range(5):
            eligible=sorted((r for r in rows if r['uid'] in ids and by_uid[r['uid']]['label']==label),key=lambda r:digest([cfg['seed'],r['uid']]))
            sample+=eligible[:count]
    sample_path=dest.with_name('audit_sample_200.jsonl'); write_jsonl(sample_path,sample)
    return dict(status='awaiting_annotations',sources=len(rows),file=str(dest),sample_file=str(sample_path),sample_sources=len(sample))


def import_audit(cfg,filename):
    manifest=load_manifest(cfg)
    pools={r['uid']:r for r in load_oof(cfg)}
    annotations=records(filename); seen=set(); usable=[]; rejected=[]
    for row in annotations:
        uid=row['uid']
        if uid in seen or uid not in pools: raise ValueError('Duplicate/non-OOF audit source')
        seen.add(uid); original=pools[uid]
        if row.get('audited') is not True or not row.get('annotator','').strip(): continue
        if row['source']!=original['source'] or row['pool_hash']!=digest(original['candidates']): raise ValueError('Audit does not match current OOF pool')
        cs=row['candidates']
        if [c['text'] for c in cs]!=[c['text'] for c in original['candidates']]: raise ValueError('Candidate order/text changed in annotation')
        candidates=[]
        for base,c in zip(original['candidates'],cs):
            for key in ('grammar','meaning','unnecessary'):
                if c[key] is not None and type(c[key]) is not bool: raise ValueError('Audit fields must be true/false/null')
            values=[c['grammar'],c['meaning'],c['unnecessary']]
            quality=None
            if all(v is not None for v in values): quality=c['grammar'] and c['meaning'] and not c['unnecessary']
            candidates.append(dict(base,quality=quality,audit=c))
        pairs=[]
        for p in row.get('ordered_pairs',[]):
            a,b=p['better'],p['worse']
            if type(a) is not int or type(b) is not int or a==b or not 0<=a<len(cs) or not 0<=b<len(cs): raise ValueError('Invalid audited pair indexes')
            if p.get('certain') is not True: continue
            if candidates[a]['quality'] is not True or candidates[b]['quality'] is not False:
                raise ValueError('Clear preference requires audited acceptable versus unacceptable; unknown cannot be a negative')
            if (a,b) not in pairs: pairs.append((a,b))
        if not pairs:
            rejected.append(uid); continue
        usable.append(dict(original,candidates=candidates,pairs=[dict(better=a,worse=b) for a,b in pairs],annotator=row['annotator']))
    split=manifest['rank_split']; fit=[r for r in usable if r['uid'] in set(split['fit'])]; inner=[r for r in usable if r['uid'] in set(split['inner'])]
    n_pairs=sum(len(r['pairs']) for r in usable)
    minimum=cfg['models']['reranker']
    # Audit diversity must be visible across source labels and all generated families.
    labels=sorted({r['label'] for r in usable}); families=sorted({p['family'] for r in usable for c in r['candidates'] for p in c['provenance']})
    valid_inner=sum(bool(r['references']) for r in inner)
    ready=len(usable)>=minimum['min_audited_sources'] and n_pairs>=minimum['min_audited_ordered_pairs'] and len(fit)>=20 and valid_inner>=10 and len(labels)>=3 and all(f in families for f in ('bart','vit5','qwen','copy'))
    dest=output(cfg)/'preferences'; write_jsonl(dest/'fit.jsonl',fit); write_jsonl(dest/'inner.jsonl',inner)
    report=dict(status='ready' if ready else 'awaiting_annotations',sources=len(usable),ordered_pairs=n_pairs,fit_sources=len(fit),validation_sources=len(inner),
                labels=labels,families=families,validation_valid_reference_sources=valid_inner,no_clear_pairs=rejected,annotation_hash=file_hash(filename),
                oof_hash=file_hash(output(cfg)/'oof/pools.jsonl'),data_identity=manifest['data_identity'],
                files={p.name:file_hash(p) for p in (dest/'fit.jsonl',dest/'inner.jsonl')})
    write_json(dest/'audit_report.json',report); return report


def training_preferences(cfg):
    dest=output(cfg)/'preferences'; report=read_json(dest/'audit_report.json')
    if report['status']!='ready': raise RuntimeError('awaiting_annotations: audited preference gates not met')
    if report['data_identity']!=load_manifest(cfg)['data_identity'] or report['oof_hash']!=file_hash(output(cfg)/'oof/pools.jsonl'): raise ValueError('Preferences refer to stale OOF/data')
    for name,h in report['files'].items():
        if file_hash(dest/name)!=h: raise ValueError('Preference checksum mismatch')
    load_oof(cfg)
    return records(dest/'fit.jsonl'),records(dest/'inner.jsonl')


def smoke_fixtures():
    from .candidates import pool
    fixtures=[('Với cuốn sách này đã giúp tôi hiểu bài.','Cuốn sách này đã giúp tôi hiểu bài.'),
              ('Em đang một bài thơ.','Em đang viết một bài thơ.'),
              ('Qua cuộc thi đã giúp em tự tin.','Cuộc thi đã giúp em tự tin.'),
              ('Hai tay em cầm cặp, bước vào lớp.','Em cầm cặp bằng hai tay, bước vào lớp.')]
    rows=[]
    for i,(s,t) in enumerate(fixtures):
        candidates=pool(s,dict(bart=[t]));
        for c in candidates: c['quality']=c['text']==t
        a=next(j for j,c in enumerate(candidates) if c['text']==t); b=next(j for j,c in enumerate(candidates) if c['text']==s)
        rows.append(dict(uid='structural-fixture-'+str(i),source=s,label=1,references_raw=[t],references=[t],candidates=candidates,pairs=[dict(better=a,worse=b)]))
    return rows
