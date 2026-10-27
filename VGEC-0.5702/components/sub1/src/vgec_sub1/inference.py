"""Stage caches are source-only, immutable by identity, with chunk checksums."""
import gc
import random
import time
from pathlib import Path
from .io import digest,read_json,write_json,write_jsonl,records,file_hash,code_hash
from .config import output,path,role_config
from .data import load_manifest,source_rows
from .models import Runtime
from .training import best_checkpoint
from .checkpoints import seed_all
from .candidates import pool,select,export


def inputs(cfg,view='test',fold='final'):
    if view=='oof':
        m=load_manifest(cfg); heldout=set(m['folds'][str(fold)]['heldout'])
        rows=records(path(cfg,cfg['data']['prepared_dir'])/'universe.jsonl')
        rows=[r for r in rows if r['uid'] in heldout]
        return [dict(ID=r['uid'],source=r['source'],uid=r['uid'],row_index=i) for i,r in enumerate(rows)]
    if view=='dev':
        rows=records(path(cfg,cfg['data']['prepared_dir'])/'dev.jsonl')
        return [dict(ID=r['uid'],source=r['source'],uid=r['uid'],row_index=i) for i,r in enumerate(rows)]
    raw=records(path(cfg,cfg['data']['root'])/cfg['data']['test'])
    return source_rows(raw,'private')


def cache_root(cfg,view,fold):
    return output(cfg)/'inference'/f'{view}-{fold}'


def prepare_pools(cfg,view='test',fold='final',commit=lambda:None):
    """Pure CPU verification/alignment before a reranker GPU is allocated."""
    rows=inputs(cfg,view,fold)
    generations={family:completed_stage(cfg,family,view,fold) for family in ('bart','vit5','qwen_corrector')}
    identity=digest(dict(rows=rows,generations=generations,config=cfg['_config_hash'],code=code_hash(Path(cfg['_root']))))
    dest=cache_root(cfg,view,fold)/'candidate-pools'
    if (dest/'complete.json').exists():
        info=read_json(dest/'complete.json')
        if info.get('identity')!=identity or info['hash']!=file_hash(dest/'rows.jsonl'):
            raise ValueError('Stale/corrupt CPU candidate pool')
        return records(dest/'rows.jsonl')
    result=[]
    for i,r in enumerate(rows):
        cs=pool(r['source'],{f:generations[k][i].get('items',generations[k][i]['texts']) for f,k in (('bart','bart'),('vit5','vit5'),('qwen','qwen_corrector'))})
        result.append(dict(uid=r['uid'],candidates=cs))
        if (i+1)%128==0: print(dict(stage='cpu_candidate_alignment',view=view,fold=fold,source_done=i+1,total=len(rows)),flush=True)
    write_jsonl(dest/'rows.jsonl',result)
    write_json(dest/'complete.json',dict(identity=identity,hash=file_hash(dest/'rows.jsonl'),count=len(rows)))
    commit(); return result


def stage_context(cfg,role,view='test',fold='final',adapted=False):
    from .config import ROLES
    if role not in ROLES: raise ValueError('Unknown inference family')
    rows=inputs(cfg,view,fold)
    if role=='reranker' and not adapted: checkpoint=None; checkpoint_id='frozen-base'
    else: checkpoint,checkpoint_id=best_checkpoint(cfg,role,fold)
    identity=digest(dict(source_inputs=rows,role=role,recipe=role_config(cfg,role),checkpoint=checkpoint_id,
                         snapshot=str(checkpoint.name) if checkpoint else None,code=code_hash(Path(cfg['_root'])),
                         config_hash=cfg['_config_hash'],benchmark_hash=file_hash(output(cfg)/'benchmark.json')))
    generations={}
    if role=='reranker':
        generations={family:completed_stage(cfg,family,view,fold) for family in ('bart','vit5','qwen_corrector')}
        identity=digest(dict(identity=identity,generations=generations))
    return rows,checkpoint,identity,generations


def stage(cfg,role,view='test',fold='final',adapted=False,commit=lambda:None,soft_seconds=19800):
    rows,checkpoint,identity,generations=stage_context(cfg,role,view,fold,adapted)
    root=cache_root(cfg,view,fold)
    stage_name=role+('-adapted' if adapted and role=='reranker' else '')
    dest=root/stage_name; dest.mkdir(parents=True,exist_ok=True)
    complete=dest/'complete.json'
    if complete.exists():
        info=read_json(complete)
        if info['identity']!=identity or info['hash']!=file_hash(dest/'rows.jsonl'): raise ValueError('Stale/corrupt inference cache')
        return dict(status='completed',stage=stage_name,cached=True,count=len(rows))
    prepared_pools=None
    if role=='reranker':
        # The coordinator creates these before submitting this GPU invocation.
        # Fail rather than spending GPU time on a hidden CPU alignment fallback.
        dest_pools=root/'candidate-pools'
        proof=read_json(dest_pools/'complete.json')
        pool_identity=digest(dict(rows=rows,generations=generations,config=cfg['_config_hash'],code=code_hash(Path(cfg['_root']))))
        if proof['identity']!=pool_identity or proof['hash']!=file_hash(dest_pools/'rows.jsonl'):
            raise ValueError('Prepare current candidate pools on CPU before reranker GPU')
        prepared_pools=records(dest_pools/'rows.jsonl')
        if [r['uid'] for r in prepared_pools]!=[r['uid'] for r in rows]: raise ValueError('CPU candidate pool coverage mismatch')
    info_path=dest/'identity.json'
    if info_path.exists() and read_json(info_path)['identity']!=identity: raise ValueError('Partial stage belongs to another experiment; choose a new run_name')
    write_json(info_path,dict(identity=identity)); commit()
    seed_all(cfg['seed']); runtime=Runtime(cfg,role,checkpoint=checkpoint)
    runtime.net.eval(); start=time.monotonic()
    bench=read_json(output(cfg)/'benchmark.json')
    batch=bench['roles'][role].get('inference_batch',1)
    chunk_size=32; all_results=[]; valid=0; invalid=0
    for start_index in range(0,len(rows),chunk_size):
        group=rows[start_index:start_index+chunk_size]; chunk=dest/f'chunk-{start_index:06d}.jsonl'; marker=chunk.with_suffix('.complete.json')
        if marker.exists():
            if read_json(marker)!=dict(identity=identity,hash=file_hash(chunk)): raise ValueError('Chunk checksum/identity failure')
            generated=records(chunk)
        else:
            generated=[]
            for i in range(0,len(group),batch):
                subset=group[i:i+batch]
                seed_all(cfg['seed']+start_index+i)
                if role=='bami':
                    labels=runtime.classify([r['source'] for r in subset]); generated.extend(dict(uid=r['uid'],label=l) for r,l in zip(subset,labels))
                elif role=='reranker':
                    import torch
                    groups=[]; flat_pairs=[]
                    for offset,r in enumerate(subset):
                        idx=start_index+i+offset
                        cs=prepared_pools[idx]['candidates']
                        groups.append(cs); flat_pairs.extend((r['source'],c['text']) for c in cs)
                    flat_scores=[]
                    with torch.no_grad(),runtime.autocast():
                        for j in range(0,len(flat_pairs),batch): flat_scores+=runtime.margins(flat_pairs[j:j+batch]).tolist()
                    score_cursor=0
                    for r,cs in zip(subset,groups):
                        scores=flat_scores[score_cursor:score_cursor+len(cs)]; score_cursor+=len(cs)
                        import math
                        for c,s in zip(cs,scores): c['rank_margin']=s if math.isfinite(s) else None
                        generated.append(dict(uid=r['uid'],candidates=cs,finite_scores=sum(math.isfinite(s) for s in scores),selection=select(r['source'],cs,scores,cfg['selection']['tie_tolerance'])))
                else:
                    texts=runtime.generate([r['source'] for r in subset],two=True)
                    traces=runtime.last_generation_trace
                    # One schema retry, using the identical contract and a deterministic new sampling draw.
                    if role=='qwen_corrector' and cfg['models']['qwen_corrector']['schema_retry_count']:
                        missing=[j for j,x in enumerate(texts) if not x]
                        for j in missing:
                            retry=runtime.generate([subset[j]['source']],two=True)
                            texts[j]=retry[0]
                            traces[j]+=runtime.last_generation_trace[0]
                    for r,t,trace in zip(subset,texts,traces):
                        items=[dict(text=z['text'],mode=z['mode'],index=z.get('beam',1 if z['mode']=='sampling' else 0))
                               for z in trace if z.get('text') and z.get('ended') and z.get('schema_valid',True)]
                        generated.append(dict(uid=r['uid'],texts=t,items=items,trace=trace))
            write_jsonl(chunk,generated); write_json(marker,dict(identity=identity,hash=file_hash(chunk))); commit()
        if [r['uid'] for r in generated]!=[r['uid'] for r in group]: raise ValueError('Stage row order/coverage mismatch')
        all_results+=generated
        if role in ('bart','vit5','qwen_corrector'):
            valid+=sum(bool(r['texts']) for r in generated); invalid+=sum(not r['texts'] for r in generated)
        print(dict(stage=stage_name,view=view,fold=fold,source_done=len(all_results),total=len(rows),valid_sources=valid,invalid_sources=invalid),flush=True)
        if time.monotonic()-start>=soft_seconds and len(all_results)<len(rows): return dict(status='needs_resume',stage=stage_name,source_done=len(all_results))
    if role in ('bart','vit5','qwen_corrector') and not valid: raise RuntimeError('Entire family has zero valid outputs; degraded mode is forbidden')
    if role=='reranker' and not any(r['finite_scores'] for r in all_results): raise RuntimeError('Entire reranker stage has zero finite scores; degraded mode forbidden')
    write_jsonl(dest/'rows.jsonl',all_results)
    write_json(complete,dict(identity=identity,hash=file_hash(dest/'rows.jsonl'),count=len(rows),valid_sources=valid,invalid_sources=invalid,seconds=time.monotonic()-start))
    commit(); return dict(status='completed',stage=stage_name,count=len(rows),valid_sources=valid,invalid_sources=invalid)


def completed_stage(cfg,role,view,fold='final',adapted=False):
    dest=cache_root(cfg,view,fold)/(role+('-adapted' if role=='reranker' and adapted else ''))
    complete=read_json(dest/'complete.json')
    if complete['hash']!=file_hash(dest/'rows.jsonl'): raise ValueError('Completed stage checksum failure')
    rows=records(dest/'rows.jsonl'); expected,_,identity,_=stage_context(cfg,role,view,fold,adapted)
    if [r['uid'] for r in rows]!=[r['uid'] for r in expected]: raise ValueError('Completed stage source coverage failure')
    info=read_json(dest/'identity.json')
    if info['identity']!=complete['identity'] or identity!=complete['identity']:
        raise ValueError('Cached stage differs from current checkpoint/config/inputs/benchmark')
    return rows


def assemble_oof(cfg):
    m=load_manifest(cfg); universe=records(path(cfg,cfg['data']['prepared_dir'])/'universe.jsonl'); lookup={r['uid']:r for r in universe}
    result=[]; proofs=[]
    for fold in ('0','1','2'):
        rows=inputs(cfg,'oof',fold)
        generations={family:completed_stage(cfg,family,'oof',fold) for family in ('bart','vit5','qwen_corrector')}
        prepared=prepare_pools(cfg,'oof',fold)
        for i,r in enumerate(rows):
            original=lookup[r['uid']]
            cs=prepared[i]['candidates']
            result.append(dict(original,candidates=cs,fold=fold))
        heldout=set(m['folds'][fold]['heldout'])
        if heldout & (set(m['folds'][fold]['fit'])|set(m['folds'][fold]['inner'])): raise ValueError('OOF train/selection leakage')
        proofs.append(dict(fold=fold,heldout_hash=digest(sorted(heldout)),model_identities={r:best_checkpoint(cfg,r,fold)[1] for r in generations},
                           no_train_or_selection_overlap=True))
    if len(result)!=len(lookup) or len({r['uid'] for r in result})!=len(lookup): raise ValueError('OOF coverage is not exactly once')
    dest=output(cfg)/'oof'; write_jsonl(dest/'pools.jsonl',result); write_json(dest/'proof.json',dict(data_identity=m['data_identity'],folds=proofs,count=len(result),hash=file_hash(dest/'pools.jsonl')))
    return dict(status='completed',count=len(result))


def finalize(cfg,view='test',adapted=False):
    rows=inputs(cfg,view); labels=[r['label'] for r in completed_stage(cfg,'bami',view)]
    selected=completed_stage(cfg,'reranker',view,adapted=adapted)
    selections=[r['selection'] for r in selected]
    dest=output(cfg)/'exports'/(view+('-adapted' if adapted else '-frozen'))
    report=export([dict(ID=r['ID']) for r in rows],labels,selections,dest)
    report['config_hash']=cfg['_config_hash']; report['code_hash']=code_hash(Path(cfg['_root']))
    write_jsonl(dest/'internal.jsonl',[dict(row,label=l,**sel) for row,l,sel in zip(rows,labels,selections)])
    if view=='dev':
        from .metrics import classification,corrections
        gold=records(path(cfg,cfg['data']['prepared_dir'])/'dev.jsonl')
        report.update(classification=classification([r['label'] for r in gold],labels),internal=corrections(gold,[s['text'] for s in selections]),
                      submitted=corrections(gold,[r['source'] if l==0 else s['text'] for r,l,s in zip(gold,labels,selections)]),official=False)
        report['internal_valid_reference']=corrections(gold,[s['text'] for s in selections],reference_view='valid')
        report['submitted_valid_reference']=corrections(gold,[r['source'] if l==0 else s['text'] for r,l,s in zip(gold,labels,selections)],reference_view='valid')
        report['S_proxy_user_all']=(report['classification']['macro_f1']+report['submitted']['gleu_plus_2016_proxy'])/2
        report['S_proxy_raw_all']=(report['classification']['macro_f1']+report['internal']['gleu_plus_2016_proxy'])/2
    write_json(dest/'report.json',report); return dict(status='completed',destination=str(dest),**report)
