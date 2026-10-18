"""Development comparisons reuse fixed dev outputs, never tune on private_test."""
from .config import output,path
from .io import records,write_json
from .inference import completed_stage
from .metrics import corrections,classification
from .candidates import select


def run(cfg,adapted=False,bootstrap=False):
    gold=records(path(cfg,cfg['data']['prepared_dir'])/'dev.jsonl')
    labels=[r['label'] for r in completed_stage(cfg,'bami','dev')]
    rank=completed_stage(cfg,'reranker','dev',adapted=adapted)
    cls=classification([r['label'] for r in gold],labels)
    report=dict(official=False,classification=cls,variants={})
    submitted_views={}
    for variant,allowed in dict(copy={'copy'},bart_copy={'bart','copy'},vit5_copy={'vit5','copy'},qwen_copy={'qwen','copy'},full_pool={'bart','vit5','qwen','copy'}).items():
        texts=[]
        for row,scored in zip(gold,rank):
            cs=[c for c in scored['candidates'] if any(p['family'] in allowed for p in c['provenance'])]
            texts.append(select(row['source'],cs,[c['rank_margin'] if c['rank_margin'] is not None else float('nan') for c in cs])['text'])
        submitted=[row['source'] if l==0 else text for row,l,text in zip(gold,labels,texts)]
        submitted_views[variant]=submitted
        metrics=corrections(gold,submitted)
        report['variants'][variant]=dict(submitted=metrics,internal=corrections(gold,texts),S_proxy_user_all=(cls['macro_f1']+metrics['gleu_plus_2016_proxy'])/2)
    if bootstrap:
        from .bootstrap import paired
        report['paired_bootstrap']={}
        for name in ('copy','bart_copy','vit5_copy','qwen_copy'):
            report['paired_bootstrap']['full_pool_minus_'+name]=paired(gold,submitted_views['full_pool'],submitted_views[name],cfg['scoring']['bootstrap_samples'],cfg['scoring']['bootstrap_seed'])
        if adapted:
            frozen=completed_stage(cfg,'reranker','dev',adapted=False)
            baseline=[r['source'] if l==0 else f['selection']['text'] for r,l,f in zip(gold,labels,frozen)]
            report['paired_bootstrap']['adapted_minus_frozen']=paired(gold,submitted_views['full_pool'],baseline,cfg['scoring']['bootstrap_samples'],cfg['scoring']['bootstrap_seed'])
    write_json(output(cfg)/'reports'/('ablations-adapted.json' if adapted else 'ablations-frozen.json'),report)
    return report
