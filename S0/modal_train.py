"""S0 training only: BARTpho attention and ViT5-base."""
import gc
import json
import os
import sys
from pathlib import Path
import modal

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from vgec.modal_runtime import make_image, artifacts, cache, project_root, GPU
from vgec.training.routing import resolve_job, validate_run_id, verify_data, preflight, run_sequence

ROOT = project_root()
app = modal.App('vgec-s0-train')
image = make_image(training=True)
VOLUMES = {'/artifacts': artifacts, '/hf-cache': cache}


@app.function(image=image, gpu=GPU, timeout=24*60*60, volumes=VOLUMES, scaledown_window=2)
def finetune(model: str, run_id: str, smoke_steps: int = 0, resume: bool = False):
    import math
    import torch
    from vgec.training.full_ft import train, validation
    from vgec.models.seq2seq import load_checkpoint
    from vgec.io import read_records, write_json, write_jsonl
    os.chdir('/workspace')
    artifacts.reload()
    validate_run_id(run_id)
    if smoke_steps < 0:
        raise ValueError('smoke_steps must be nonnegative')
    report = verify_data(ROOT)
    config, prepared, cfg = resolve_job(model, ROOT)
    out = Path('/artifacts') / run_id
    try:
        result = train(config, ROOT/'configs/tracks/experimental.yaml',
                       prepared/'train.jsonl', prepared/'dev.jsonl', out,
                       smoke_steps, resume, raw_train_hash=report['raw_hashes']['train'],
                       commit=artifacts.commit)
        gc.collect()
        torch.cuda.empty_cache()
        if smoke_steps:
            if result['optimizer_steps'] != smoke_steps or not math.isfinite(result['last_loss']):
                raise RuntimeError('Smoke must finish with finite loss and the requested steps')
            net, tok, saved = load_checkpoint(out/'last', torch.device('cuda'))
            if saved['model_id'] != cfg['model_id'] or saved.get('classification_pooling', 'mean') != cfg.get('classification_pooling', 'mean'):
                raise RuntimeError('Smoke reload architecture mismatch')
            del net, tok
        else:
            checkpoints = ['best_classification', 'best_correction'] if cfg['joint'] else ['best_correction']
            for name in checkpoints:
                net, tok, saved = load_checkpoint(out/name, torch.device('cuda'))
                with torch.no_grad():
                    pred, metrics = validation(net, tok, saved, read_records(prepared/'dev.jsonl'), torch.device('cuda'))
                write_jsonl(out/f'dev_predictions_{name}.jsonl', pred)
                write_json(out/f'dev_metrics_{name}.json', metrics)
                del net, tok
                gc.collect()
                torch.cuda.empty_cache()
        return result
    finally:
        artifacts.commit()
        cache.commit()


@app.function(image=image, timeout=24*60*60)
def workflow():
    """CPU orchestration: smoke both models, then full train both sequentially."""
    os.chdir('/workspace')
    preflight(ROOT)
    return run_sequence(finetune.remote)


@app.local_entrypoint()
def main(model: str = 'bartpho_attention', run_id: str = '', smoke_steps: int = 0, resume: bool = False):
    preflight(ROOT)
    resolve_job(model, ROOT)
    if smoke_steps < 0:
        raise ValueError('smoke_steps must be nonnegative')
    if not run_id:
        prefix = 's0-bartpho-attention' if model == 'bartpho_attention' else 's0-vit5-base'
        run_id = f"{prefix}-{'smoke' if smoke_steps else 'full'}-42"
    validate_run_id(run_id)
    print(json.dumps(finetune.remote(model, run_id, smoke_steps, resume), ensure_ascii=False, indent=2))
