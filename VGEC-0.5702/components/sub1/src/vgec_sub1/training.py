import gc
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
import torch
from .config import output,role_config,path
from .data import select_training,load_manifest
from .io import digest,code_hash,read_json,write_json,records,write_jsonl
from .checkpoints import Store,local_lock,seed_all,rng_state,restore_rng
from .models import Runtime
from .metrics import classification,corrections


def identity(cfg,role,fold,fit,inner,purpose='train'):
    from .io import file_hash
    benchmark_hash=file_hash(output(cfg)/'benchmark.json') if purpose=='train' else None
    return digest(dict(version='sub1-implementation-1',purpose=purpose,role=role,fold=str(fold),
                       config=cfg['_config_hash'],recipe=role_config(cfg,role),fit=fit,inner=inner,
                       code=code_hash(Path(cfg['_root'])),benchmark_hash=benchmark_hash))


def optimizer_for(runtime):
    rc=runtime.rc; groups={}
    for name,p in runtime.net.named_parameters():
        if not p.requires_grad: continue
        lr=rc.get('head_lr',rc['lr']) if runtime.role=='bami' and not name.startswith('encoder.') else rc['lr']
        decay=0.0 if p.ndim<2 or 'norm' in name.lower() or name.endswith('bias') else rc['weight_decay']
        groups.setdefault((lr,decay),[]).append(p)
    return torch.optim.AdamW([dict(params=p,lr=lr,weight_decay=decay) for (lr,decay),p in groups.items()],betas=(.9,.999),eps=1e-8)


def scheduler_for(optimizer,steps,warmup):
    warm=max(1,math.ceil(steps*warmup))
    return torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step: step/warm if step<warm else max(0,(steps-step)/max(steps-warm,1)))


def windows(n,batch,seed,epoch):
    order=list(range(n)); random.Random(seed+epoch).shuffle(order)
    return [order[i:i+batch] for i in range(0,n,batch)]


def train_window(runtime,optimizer,scheduler,rows,micro):
    optimizer.zero_grad(set_to_none=True); total=0.0
    denominator=len(rows)
    for offset in range(0,len(rows),micro):
        with runtime.autocast(): losses=runtime.source_losses(rows[offset:offset+micro])
        if not torch.isfinite(losses).all(): raise ValueError('Nonfinite source loss')
        loss=losses.sum()/denominator
        loss.backward(); total+=float(loss.detach())
    torch.nn.utils.clip_grad_norm_(runtime.net.parameters(),runtime.rc['clip'],error_if_nonfinite=True)
    optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
    return total


def _gradient_comparison(runtime,rows,autocast):
    runtime.net.zero_grad(set_to_none=True)
    subset=rows[:2]
    with autocast(): loss=runtime.source_losses(subset).mean()
    if not torch.isfinite(loss): raise ValueError('Nonfinite reference loss')
    loss.backward()
    reference={n:p.grad.detach().cpu().clone() for n,p in runtime.net.named_parameters() if p.grad is not None}
    runtime.net.zero_grad(set_to_none=True)
    for row in subset:
        with autocast(): loss=runtime.source_losses([row]).sum()/len(subset)
        loss.backward()
    error=norm=other_norm=dot=0.; checked=0
    for n,p in runtime.net.named_parameters():
        if n not in reference: continue
        if p.grad is None or not torch.isfinite(p.grad).all(): raise ValueError('Missing/nonfinite accumulated gradient: '+n)
        a=reference[n].float(); b=p.grad.detach().cpu().float()
        error+=float((a-b).square().sum()); norm+=float(a.square().sum())
        other_norm+=float(b.square().sum()); dot+=float((a*b).sum()); checked+=a.numel()
    relative=math.sqrt(error/max(norm,1e-30))
    cosine=dot/max(math.sqrt(norm*other_norm),1e-30)
    return dict(relative_l2_error=relative,cosine_similarity=min(1.,cosine),checked_parameters=checked)


def gradient_parity_gate(runtime,rows):
    """Separate FP32 source-weight correctness from BF16 batch-shape roundoff."""
    saved=rng_state(); was_training=runtime.net.training
    flags=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    runtime.net.eval()
    try:
        print(dict(stage='gradient_parity',role=getattr(runtime,'role',None),precision='runtime'),flush=True)
        result=_gradient_comparison(runtime,rows,runtime.autocast)
        limit=.10 if getattr(runtime,'device',torch.device('cpu')).type=='cuda' else 1e-3
        cosine_limit=.995 if limit==.10 else .99999
        if not math.isfinite(result['relative_l2_error']) or result['relative_l2_error']>limit or result['cosine_similarity']<cosine_limit:
            raise ValueError('Source-normalized gradient parity failed: '+str(result))
        result.update(status='passed',tolerance_relative=limit,tolerance_cosine=cosine_limit,dropout=False)
        if getattr(runtime,'role',None) in ('bami','bart','vit5'):
            print(dict(stage='gradient_parity',role=runtime.role,precision='float32',tf32=False),flush=True)
            if any(p.dtype!=torch.float32 for p in runtime.net.parameters()):
                raise ValueError('Full fine-tune diagnostic requires FP32 master parameters')
            torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
            fp32=_gradient_comparison(runtime,rows,nullcontext)
            if not math.isfinite(fp32['relative_l2_error']) or fp32['relative_l2_error']>1e-3 or fp32['cosine_similarity']<.99999:
                raise ValueError('FP32 accumulation parity failed; BF16 tolerance cannot override it: '+str(fp32))
            result['fp32_reference']=dict(fp32,status='passed',tf32=False,tolerance_relative=1e-3,tolerance_cosine=.99999)
        return result
    finally:
        runtime.net.zero_grad(set_to_none=True); runtime.net.train(was_training); restore_rng(saved)
        torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=flags


def encoded_cache(cfg,runtime,rows):
    key=digest(dict(rows=rows,recipe=runtime.rc,code=code_hash(Path(cfg['_root']))))
    dest=path(cfg,cfg['runtime']['cache_dir'])/'tokens'/(key+'.jsonl')
    if dest.exists():
        complete=read_json(dest.with_suffix('.complete.json'))
        from .io import file_hash
        if complete['hash']!=file_hash(dest): raise ValueError('Token cache checksum mismatch')
        return records(dest)
    encoded=[runtime.encode(r) for r in rows]
    write_jsonl(dest,encoded)
    from .io import file_hash
    write_json(dest.with_suffix('.complete.json'),dict(hash=file_hash(dest),count=len(encoded)))
    return encoded


def rank_batch(rows,epoch,step,limit=4):
    # At least one explicit audited pair; rotate partners without labeling unknowns.
    result=[]
    for row in rows:
        p=row['pairs'][(epoch+step)%len(row['pairs'])]
        ids=[p['better'],p['worse']]
        rest=[i for i in range(len(row['candidates'])) if i not in ids]
        random.Random(epoch+step).shuffle(rest); ids+=rest[:max(0,limit-2)]
        remap={v:i for i,v in enumerate(ids)}
        pairs=[dict(better=remap[x['better']],worse=remap[x['worse']]) for x in row['pairs'] if x['better'] in remap and x['worse'] in remap]
        result.append(dict(row,candidates=[row['candidates'][i] for i in ids],pairs=pairs))
    return result


@torch.no_grad()
def validate(runtime,rows,batch=4):
    saved=rng_state(); runtime.net.eval()
    try:
        if runtime.role=='bami':
            predicted=[]
            for i in range(0,len(rows),batch): predicted+=runtime.classify([r['source'] for r in rows[i:i+batch]])
            m=classification([r['label'] for r in rows],predicted)
            return (m['macro_f1'],),m
        if runtime.role=='reranker':
            from .candidates import pool,select
            hypotheses=[]; good=total=0; all_scores=[]
            pairs=[(r['source'],c['text']) for r in rows for c in r['candidates']]
            pair_batch=max(1,batch)
            with runtime.autocast():
                for i in range(0,len(pairs),pair_batch): all_scores+=runtime.margins(pairs[i:i+pair_batch]).tolist()
            if not any(math.isfinite(s) for s in all_scores): raise RuntimeError('All reranker validation margins nonfinite')
            cursor=0
            for r in rows:
                scores=all_scores[cursor:cursor+len(r['candidates'])]; cursor+=len(r['candidates'])
                candidates=pool(r['source'],{})
                # Annotated candidates retain the complete OOF pool/provenance.
                candidates=r['candidates']
                hypotheses.append(select(r['source'],candidates,scores)['text'])
                for p in r['pairs']:
                    good+=scores[p['better']]>scores[p['worse']]; total+=1
            m=corrections(rows,hypotheses,reference_view='valid'); m['pair_accuracy']=good/max(total,1)
            return (m['gleu_plus_2016_proxy'],m['pair_accuracy']),m
        hypotheses=[]; invalid=0
        for i in range(0,len(rows),batch):
            group=rows[i:i+batch]; generated=runtime.generate([r['source'] for r in group],two=False)
            invalid+=sum(not x for x in generated)
            hypotheses.extend(x[0] if x else '' for x in generated)
            print(dict(stage='validation',role=runtime.role,source_done=min(i+batch,len(rows)),total=len(rows),invalid=invalid),flush=True)
        m=corrections(rows,hypotheses,reference_view='valid'); m['invalid_count']=invalid
        if invalid==len(rows): raise RuntimeError('All '+runtime.role+' validation outputs invalid: EOS/parser gate failed')
        return (m['gleu_plus_2016_proxy'],m['exact_match']),m
    finally:
        runtime.net.train(); restore_rng(saved)


@torch.no_grad()
def smoke_outputs(runtime,rows):
    saved=rng_state(); was_training=runtime.net.training; runtime.net.eval()
    try:
        if runtime.role=='bami':
            values=runtime.classify([r['source'] for r in rows])
            if len(values)!=len(rows) or any(type(v) is not int or v not in range(5) for v in values):
                raise ValueError('Smoke classifier coverage/label failure')
            return dict(status='passed',labels=values)
        if runtime.role=='reranker':
            pairs=[(r['source'],c['text']) for r in rows for c in r['candidates']][:4]
            with runtime.autocast():
                batched=runtime.margins(pairs).float()
                singles=torch.cat([runtime.margins([p]).float() for p in pairs])
            if not torch.isfinite(batched).all(): raise ValueError('Nonfinite smoke rank margins')
            torch.testing.assert_close(batched,singles,rtol=.05,atol=.05)
            return dict(status='passed',pairs=len(pairs),max_batch_single_error=float((batched-singles).abs().max()))
        values=runtime.generate([r['source'] for r in rows],two=True)
        if len(values)!=len(rows) or not any(values): raise ValueError('Smoke family has zero valid candidate outputs')
        traces=runtime.last_generation_trace
        if len(traces)!=len(rows): raise ValueError('Smoke generation trace coverage mismatch')
        if runtime.role=='qwen_corrector' and {t['mode'] for ts in traces for t in ts}!={'greedy','sampling'}:
            raise ValueError('Smoke must exercise greedy and sampling')
        return dict(status='passed',sources=len(rows),valid_sources=sum(bool(v) for v in values),
                    traces=[dict(source=r['source'],outputs=v,trace=t) for r,v,t in zip(rows,values,traces)])
    finally:
        runtime.net.train(was_training); restore_rng(saved)


def train(cfg,role,fold='final',commit=lambda:None,purpose='train',smoke_steps=20):
    if role=='bami' and str(fold)!='final': raise ValueError('Bami is trained once, without OOF folds')
    manifest=load_manifest(cfg)
    if role=='reranker' and purpose=='smoke':
        from .preferences import smoke_fixtures
        fixture=smoke_fixtures(); fit,inner=fixture[:3],fixture[3:]
    elif role=='reranker':
        from .preferences import training_preferences
        fit,inner=training_preferences(cfg)
        baseline=read_json(output(cfg)/'exports/dev-frozen/report.json')
        if baseline.get('config_hash')!=cfg['_config_hash'] or baseline.get('code_hash')!=code_hash(Path(cfg['_root'])):
            raise ValueError('Evaluate current frozen baseline before GEC reranker fine-tuning')
    else: fit,inner=select_training(cfg,role,fold,manifest)
    rc=role_config(cfg,role)
    benchmark=read_json(output(cfg)/'benchmark.json') if purpose=='train' else None
    if benchmark:
        if benchmark['config_hash']!=cfg['_config_hash'] or benchmark['code_hash']!=code_hash(Path(cfg['_root'])):
            raise ValueError('Benchmark does not match current code/config')
        micro=benchmark['roles'][role]['micro_batch']
    else: micro=1
    if purpose=='smoke': fit=fit[:min(8,len(fit))]; inner=inner[:min(4,len(inner))]
    ident=identity(cfg,role,fold,fit,inner,purpose)
    folder=output(cfg)/('checkpoints' if purpose=='train' else 'smoke')/f'{role}-{fold}'
    with local_lock(folder):
        store=Store(folder,ident,commit); last=store.resolve('last')
        if last and read_json(last/'progress.json').get('done') and purpose=='train':
            report=dict(status='completed',role=role,fold=fold,identity=ident,state=read_json(last/'progress.json'),
                        parameter_report=read_json(last/'identity.json')['parameter_report'])
            write_json(folder/'report.json',report); commit(); return report
        seed_all(rc['seed']); load_start=time.monotonic()
        runtime=Runtime(cfg,role,training=True,checkpoint=last)
        load_seconds=time.monotonic()-load_start
        encoded=encoded_cache(cfg,runtime,fit) if role!='reranker' else fit
        parity=gradient_parity_gate(runtime,encoded) if purpose=='smoke' else None
        steps_epoch=math.ceil(len(encoded)/rc['global_batch'])
        total=smoke_steps if purpose=='smoke' else steps_epoch*rc['epochs']
        optimizer=optimizer_for(runtime); scheduler=scheduler_for(optimizer,total,rc['warmup'])
        state=dict(epoch=0,next_window=0,step=0,best=None,bad_epochs=0,done=False,
                   timing=dict(train=0.,validation=0.,save=0.,load=0.),sources_seen=0)
        if last: state=store.load_training(last,optimizer,scheduler)
        state['timing']['load']+=load_seconds
        if state['done'] and purpose=='train':
            report=dict(status='completed',role=role,fold=fold,identity=ident,state=state,parameter_report=runtime.parameter_report)
            write_json(folder/'report.json',report); commit(); return report
        begin=time.monotonic(); last_checkpoint_at=begin; last_log_at=begin
        soft=cfg['runtime']['training_slice_soft_seconds']
        max_epochs=math.ceil(smoke_steps/steps_epoch) if purpose=='smoke' else rc['epochs']
        for epoch in range(state['epoch'],max_epochs):
            order=windows(len(encoded),rc['global_batch'],rc['seed'],epoch)
            for wi in range(state['next_window'],len(order)):
                rows=[encoded[i] for i in order[wi]]
                if role=='reranker': rows=rank_batch(rows,epoch,state['step'],cfg['models']['reranker']['candidates_per_source_batch'])
                start=time.monotonic(); loss=train_window(runtime,optimizer,scheduler,rows,micro)
                state['timing']['train']+=time.monotonic()-start
                state.update(epoch=epoch,next_window=wi+1,step=state['step']+1,sources_seen=state['sources_seen']+len(rows))
                if state['step']%10==0 or state['step']==1 or time.monotonic()-last_log_at>=30:
                    rate=state['timing']['train']/state['step']
                    print(dict(stage=purpose,role=role,fold=fold,epoch=epoch+1,step=state['step'],planned_steps=total,
                               loss=loss,lr=scheduler.get_last_lr(),sources_seen=state['sources_seen'],eta_train_upper_seconds=rate*max(total-state['step'],0)),flush=True)
                    last_log_at=time.monotonic()
                if purpose=='smoke' and state['step']>=smoke_steps: break
                if time.monotonic()-begin>=soft:
                    state['timing']['save']+=store.save(runtime,optimizer,scheduler,state)
                    return dict(status='needs_resume',role=role,fold=fold,identity=ident,state=state)
                if time.monotonic()-last_checkpoint_at>=cfg['runtime']['checkpoint_interval_seconds']:
                    state['timing']['save']+=store.save(runtime,optimizer,scheduler,state); last_checkpoint_at=time.monotonic()
            if purpose=='smoke' and state['step']<smoke_steps:
                # Tiny fit sets have one optimizer window per epoch. Run all
                # twenty steps before validation/checkpointing, instead of
                # serializing the whole optimizer after every single step.
                state.update(epoch=epoch+1,next_window=0)
                continue
            eval_batch=benchmark['roles'][role]['inference_batch'] if benchmark else min(micro,4)
            start=time.monotonic(); score,metrics=validate(runtime,inner,batch=eval_batch)
            state['timing']['validation']+=time.monotonic()-start
            better=state['best'] is None or tuple(score)>tuple(state['best'])
            if better: state.update(best=list(score),bad_epochs=0)
            else: state['bad_epochs']+=1
            state.update(epoch=epoch+1,next_window=0,last_metrics=metrics)
            state['done']=state['epoch']>=max_epochs or (purpose=='train' and state['bad_epochs']>=rc['patience']) or (purpose=='smoke' and state['step']>=smoke_steps)
            if purpose=='smoke' and state['done']:
                output_check=smoke_outputs(runtime,inner)
                write_json(folder/'output_check.json',output_check)
            state['timing']['save']+=store.save(runtime,optimizer,scheduler,state,is_best=better)
            last_checkpoint_at=time.monotonic()
            if state['done']: break
            if time.monotonic()-begin>=soft: return dict(status='needs_resume',role=role,fold=fold,identity=ident,state=state)
        report=dict(status='completed',role=role,fold=fold,identity=ident,state=state,parameter_report=runtime.parameter_report,
                    code_hash=code_hash(Path(cfg['_root'])),config_hash=cfg['_config_hash'])
        if purpose=='smoke':
            runtime.net.eval()
            with torch.no_grad(),runtime.autocast(): before=runtime.source_losses(encoded[:1]).detach().float()
            # A no-op/partial reload must fail: overwrite every trained tensor first.
            with torch.no_grad():
                for p in runtime.net.parameters():
                    if p.requires_grad: p.add_(.125)
            runtime.reload_weights(store.resolve('last'))
            with torch.no_grad(),runtime.autocast(): after=runtime.source_losses(encoded[:1]).detach().float()
            torch.testing.assert_close(before,after,rtol=1e-5,atol=1e-6)
            report.update(gradient_parity=parity,output_check=read_json(folder/'output_check.json'),
                          reload_loss_parity='passed',actual_optimizer_steps=state['step'],gate_complete=True)
        write_json(folder/'report.json',report); commit()
        return report


def best_checkpoint(cfg,role,fold='final'):
    if role=='reranker':
        from .preferences import training_preferences
        fit,inner=training_preferences(cfg)
    else: fit,inner=select_training(cfg,role,fold)
    ident=identity(cfg,role,fold,fit,inner)
    root=output(cfg)/'checkpoints'/f'{role}-{fold}'
    store=Store(root,ident)
    last=store.resolve('last')
    if not last: raise FileNotFoundError('Train this model first: '+str(root))
    state=read_json(last/'progress.json')
    if not state['done']: raise ValueError('Training incomplete; resume before generation')
    best=store.resolve('best')
    if not best: raise ValueError('Missing validated best checkpoint')
    return best,ident
