import argparse
import json
import sys
from pathlib import Path
from .config import load,output,path
from .data import prepare,execution_plan,load_manifest
from .io import read_json,write_json,records,code_hash


def preflight(cfg):
    manifest=load_manifest(cfg); plan=execution_plan(cfg)
    required=['prompts/correct_v1.txt','prompts/rank_v1.txt','requirements-core.lock','requirements-llm.lock']
    for filename in required:
        if not path(cfg,filename).is_file(): raise FileNotFoundError(filename)
    from .data import source_rows
    test=source_rows(records(path(cfg,cfg['data']['root'])/cfg['data']['test']),'private')
    result=dict(status='passed_local_preflight',config_hash=cfg['_config_hash'],code_hash=code_hash(Path(cfg['_root'])),
                counts=manifest['counts'],private_test_rows=len(test),finetunes=plan['full_finetunes'],
                gold_source=cfg['data']['gold'],eligibility=manifest['constrained_eligibility'],
                pending_gpu_gates=['per_model_parameters','EOS_actual_tokenizer','smoke_each_family','A100_optimizer_decode_benchmark','kernel_profile'],
                pending_human_gate='audited_preferences_100_sources_200_pairs')
    write_json(output(cfg)/'reports/preflight.json',result); return result


def status(cfg):
    root=output(cfg); result=dict(output_dir=str(root),artifacts=[])
    for p in root.rglob('report.json'):
        result['artifacts'].append(dict(file=str(p.relative_to(root)),report=read_json(p)))
    if (root/'benchmark.json').exists(): result['benchmark']=read_json(root/'benchmark.json')
    if (root/'preferences/audit_report.json').exists(): result['preferences']=read_json(root/'preferences/audit_report.json')
    if (root/'workflow_ledger.json').exists(): result['workflow_ledger']=read_json(root/'workflow_ledger.json')
    result['last_checkpoints']=[]
    for branch in ('checkpoints','smoke'):
        for p in (root/branch).glob('*/pointers.json'):
            info=read_json(p); last=p.parent/info['last']/'progress.json'
            result['last_checkpoints'].append(dict(job=str(p.parent.relative_to(root)),progress=read_json(last)))
    return result


def main():
    parser=argparse.ArgumentParser(description='VGEC sub1: constrained 3-fold OOF + independent classifier + reranker')
    parser.add_argument('command',choices=['audit','preflight','plan','status','audit-template','import-audit','export','local-train','local-stage'])
    parser.add_argument('--config',default='configs/sub1.yaml'); parser.add_argument('--role',default='bart')
    parser.add_argument('--fold',default='final',choices=['final','0','1','2']); parser.add_argument('--view',default='test',choices=['test','dev','oof'])
    parser.add_argument('--annotations'); parser.add_argument('--adapted',action='store_true')
    args=parser.parse_args(); cfg=load(args.config)
    if args.command=='audit':
        m=prepare(cfg); result=dict(status='audited',counts=m['counts'],data_identity=m['data_identity'],exclusions=len(m['excluded']),manifest=str(path(cfg,cfg['data']['prepared_dir'])/'manifest.json'))
    elif args.command=='preflight': result=preflight(cfg)
    elif args.command=='plan': result=execution_plan(cfg)
    elif args.command=='status': result=status(cfg)
    elif args.command=='audit-template':
        from .preferences import audit_template
        result=audit_template(cfg)
    elif args.command=='import-audit':
        from .preferences import import_audit
        if not args.annotations: parser.error('--annotations is required')
        result=import_audit(cfg,Path(args.annotations))
    elif args.command=='export':
        from .inference import finalize
        result=finalize(cfg,args.view,args.adapted)
    elif args.command=='local-train':
        import torch
        if not torch.cuda.is_available(): raise RuntimeError('Use Modal A100; CPU full train is blocked')
        from .training import train
        result=train(cfg,args.role,args.fold)
    else:
        from .inference import stage,prepare_pools
        if args.role=='reranker': prepare_pools(cfg,args.view,args.fold)
        result=stage(cfg,args.role,args.view,args.fold,args.adapted)
    print(json.dumps(result,ensure_ascii=False,indent=2))
