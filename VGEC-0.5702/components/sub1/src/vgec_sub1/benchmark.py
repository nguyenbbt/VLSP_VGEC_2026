"""Real optimizer, decode and save measurements, isolated from full training."""
import gc
import math
import time
import tempfile
from pathlib import Path
import torch
from .models import Runtime,causal_sft,parse_correction
from .config import output,role_config
from .data import select_training,load_manifest,execution_plan
from .io import write_json,read_json,code_hash,digest,file_hash
from .training import optimizer_for,scheduler_for,train_window,validate
from .checkpoints import seed_all,Store


def benchmark_role(cfg,role,commit=lambda:None):
    if not torch.cuda.is_available(): raise RuntimeError('Performance benchmark requires the actual A100; CPU estimates are forbidden')
    props=torch.cuda.get_device_properties(0)
    if 'A100' not in props.name: raise ValueError('Benchmark must run on A100')
    smoke=read_json(output(cfg)/'smoke'/f'{role}-final/report.json')
    if smoke['status']!='completed' or not smoke.get('gate_complete') or smoke.get('actual_optimizer_steps')!=20:
        raise ValueError('Run and complete the 20-step per-family GPU smoke first')
    if smoke.get('code_hash')!=code_hash(Path(cfg['_root'])) or smoke.get('config_hash')!=cfg['_config_hash']:
        raise ValueError('GPU smoke belongs to different code/config; rerun smoke first')
    if role in ('bami','bart','vit5') and smoke.get('gradient_parity',{}).get('fp32_reference',{}).get('status')!='passed':
        raise ValueError('Full-precision accumulation reference must pass before benchmarking')
    if role=='reranker':
        from .preferences import smoke_fixtures
        fit=smoke_fixtures()
    else: fit,_=select_training(cfg,role)
    seed_all(cfg['seed']); begin=time.monotonic(); runtime=Runtime(cfg,role,training=True)
    if role=='reranker':
        # Deliberately synthetic performance fixture at the configured context bound;
        # never enters production audited preferences or a training checkpoint.
        for row in fit:
            base=row['source']
            while len(runtime.tokenizer.encode(row['source']+' '+base,add_special_tokens=False))+len(runtime.tokenizer.encode(row['candidates'][0]['text'],add_special_tokens=False))<runtime.rc['max_length']-240:
                row['source']+=' '+base
            row['candidates']=row['candidates']*2
    load_time=time.monotonic()-begin
    # Tokenize ALL training sources before launching full jobs. No hidden truncation.
    start=time.monotonic(); encoded=[runtime.encode(r) for r in fit] if role!='reranker' else fit
    lengths=[]
    for r in encoded:
        if role=='bami': lengths.append(len(r['input_ids']))
        elif role in ('bart','vit5'): lengths.append(len(r['input_ids'])+max(map(len,r['refs'])))
        elif role=='qwen_corrector': lengths.append(max(len(x['input_ids']) for x in r['refs']))
        else: lengths.append(max(len(runtime.rank_ids(r['source'],c['text'])) for c in r['candidates']))
    token_seconds=time.monotonic()-start
    # A short source with many references expands into a larger tensor batch.
    # Include that worst case instead of considering only longest single text.
    loads=[lengths[i]*max(1,len(r.get('refs',r.get('candidates',[None])))) for i,r in enumerate(encoded)]
    longest=max(range(len(encoded)),key=lambda i:lengths[i])
    indices=sorted(range(len(encoded)),key=lambda i:loads[i],reverse=True)
    indices=[longest]+[i for i in indices if i!=longest]
    heavy=[encoded[i] for i in indices[:8]]
    baseline={n:p.detach().cpu().clone() for n,p in runtime.net.named_parameters() if p.requires_grad}
    def reset_weights():
        with torch.no_grad():
            for n,p in runtime.net.named_parameters():
                if n in baseline: p.copy_(baseline[n])
        runtime.net.zero_grad(set_to_none=True); seed_all(cfg['seed'])
    trials=[]; best=None
    matrix=[1,2,4,8,16,32]
    for micro in matrix:
        if micro>runtime.rc['global_batch']: continue
        reset_weights()
        optimizer=optimizer_for(runtime); scheduler=scheduler_for(optimizer,1000,runtime.rc['warmup'])
        torch.cuda.reset_peak_memory_stats()
        elapsed=[]
        try:
            for step in range(35):
                rows=[heavy[(step+j)%len(heavy)] for j in range(runtime.rc['global_batch'])]
                torch.cuda.synchronize(); start=time.monotonic()
                train_window(runtime,optimizer,scheduler,rows,micro)
                torch.cuda.synchronize()
                if step>=5: elapsed.append(time.monotonic()-start)
            seconds=sum(elapsed)/len(elapsed); peak=torch.cuda.max_memory_allocated()
            trial=dict(micro_batch=micro,seconds_per_optimizer_step=seconds,source_per_second=runtime.rc['global_batch']/seconds,
                       stable_steps=len(elapsed),warmup_steps=5,peak_allocated_bytes=peak,peak_reserved_bytes=torch.cuda.max_memory_reserved(),status='ok')
            if peak<props.total_memory*.9 and (best is None or trial['source_per_second']>best['source_per_second']): best=trial
        except torch.OutOfMemoryError:
            trial=dict(micro_batch=micro,status='oom')
            runtime.net.zero_grad(set_to_none=True)
        trials.append(trial); print(dict(role=role,benchmark=trial),flush=True)
        del optimizer,scheduler; gc.collect(); torch.cuda.empty_cache()
        if trial['status']=='oom': break
    if best is None: raise RuntimeError('No safe microbatch passed; full train blocked')
    reset_weights(); del baseline
    runtime.net.eval(); inference_trials=[]
    raw_heavy=[fit[i] for i in indices[:min(len(fit),8)]]
    for batch in [1,2,4,8,16,32]:
        sample=[raw_heavy[i%len(raw_heavy)] for i in range(batch)]
        try:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize(); start=time.monotonic()
            if role=='bami': runtime.classify([r['source'] for r in sample])
            elif role=='reranker':
                with torch.no_grad(),runtime.autocast(): runtime.margins([(r['source'],r['candidates'][0]['text']) for r in sample])
            else:
                values=runtime.generate([r['source'] for r in sample],two=True)
                if role=='qwen_corrector' and all(not x for x in values):
                    print('Untrained benchmark Qwen has invalid output; output contract is gated by smoke, not this timing run.',flush=True)
            torch.cuda.synchronize(); seconds=time.monotonic()-start
            peak=torch.cuda.max_memory_allocated()
            if peak<props.total_memory*.9:
                inference_trials.append(dict(batch=batch,seconds=seconds,source_per_second=batch/max(seconds,1e-9),peak_allocated_bytes=peak))
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache(); break
    if not inference_trials: raise RuntimeError('No inference microbatch passed')
    decode=max(inference_trials,key=lambda r:r['source_per_second'])
    save_dir=output(cfg)/'benchmark-snapshots'/role
    optimizer=optimizer_for(runtime); scheduler=scheduler_for(optimizer,10,.06)
    runtime.net.train(); train_window(runtime,optimizer,scheduler,[heavy[0]],1)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profiler:
        train_window(runtime,optimizer,scheduler,[heavy[0]],1)
    operators=sorted(profiler.key_averages(),key=lambda e:e.device_time_total,reverse=True)[:20]
    kernel_profile=[dict(operator=e.key,device_time_us=e.device_time_total,calls=e.count) for e in operators]
    store=Store(save_dir,digest(dict(purpose='benchmark',role=role)),commit)
    save_seconds=store.save(runtime,optimizer,scheduler,dict(step=1),is_best=True)
    report=dict(**best,inference_batch=decode['batch'],inference_source_per_second=decode['source_per_second'],
                decode_trials=inference_trials,trials=trials,load_seconds=load_time,tokenize_seconds=token_seconds,save_hash_commit_seconds=save_seconds,
                max_token_length=max(lengths),p99_token_length=sorted(lengths)[min(len(lengths)-1,int(len(lengths)*.99))],
                measured_sources=len(encoded),gpu=props.name,gpu_total_bytes=props.total_memory,parameter_report=runtime.parameter_report,
                torch_version=torch.__version__,config_hash=cfg['_config_hash'],code_hash=code_hash(Path(cfg['_root'])),
                reranker_length_coverage='synthetic_near_context_limit_four_candidates' if role=='reranker' else 'actual_training_sources',kernel_profile=kernel_profile,
                kernel_flags=dict(fla_available=__import__('importlib.util',fromlist=['find_spec']).find_spec('fla') is not None))
    write_json(output(cfg)/'benchmarks'/(role+'.json'),report); commit(); return report


def summarize(cfg):
    from .config import ROLES
    roles={r:read_json(output(cfg)/'benchmarks'/(r+'.json')) for r in ROLES}
    code=code_hash(Path(cfg['_root']))
    for r,v in roles.items():
        if v['config_hash']!=cfg['_config_hash'] or v['code_hash']!=code: raise ValueError('Stale benchmark: '+r)
    plan=execution_plan(cfg); estimated=0.
    for run in plan['runs']:
        b=roles[run['role']]
        if run['role']=='reranker': continue
        estimated+=run['optimizer_steps_upper_bound']*b['seconds_per_optimizer_step']
        estimated+=run['max_epochs']*(run['validation_sources']/b['inference_source_per_second']+b['save_hash_commit_seconds'])+b['load_seconds']
    # OOF, test generation and scoring are charged phases too.
    m=load_manifest(cfg); n=sum(len(m['folds'][str(k)]['heldout']) for k in range(3))
    for role in ('bart','vit5','qwen_corrector'): estimated+=n/roles[role]['inference_source_per_second']+3*roles[role]['load_seconds']
    result=dict(config_hash=cfg['_config_hash'],code_hash=code,roles=roles,estimated_base_train_oof_seconds=estimated,
                estimated_hours_range=[estimated/3600,estimated*1.6/3600],
                excludes=['human_annotation_time','reranker_training_on_actual_preferences','private_test_and_dev','image_builds','CPU_workflow'],
                money_estimate_requires_current_hourly_rate=True,full_finetunes=14)
    write_json(output(cfg)/'benchmark.json',result); return result
