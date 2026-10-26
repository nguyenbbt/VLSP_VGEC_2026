"""S0 inference with existing BARTpho/ViT5 checkpoints on Modal."""
import sys
from pathlib import Path
import modal

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from vgec.modal_runtime import make_image, artifacts, cache, GPU
from vgec.training.routing import validate_run_id

app = modal.App('vgec-s0-predict')
image = make_image()


def source_payload(records):
    from vgec.schemas import adapt
    seen, payload = set(), []
    for record in records:
        adapt(record, 'input', supervised=False)
        key = (type(record['ID']), record['ID'])
        if key in seen:
            raise ValueError('Duplicate input IDs')
        seen.add(key)
        payload.append(dict(ID=record['ID'], Input=record['Input']))
    if not payload:
        raise ValueError('Input must be nonempty')
    return payload


@app.function(image=image, gpu=GPU, timeout=6*60*60,
              volumes={'/artifacts': artifacts, '/hf-cache': cache}, scaledown_window=2)
def infer(payload: list, run_id: str, vit5_run: str = '', batch_size: int = 4):
    import torch
    from vgec.schemas import adapt
    from vgec.models.seq2seq import load_checkpoint
    from vgec.inference.pipeline import predict_with_models
    from vgec.training.checkpoints import checkpoint_path
    validate_run_id(run_id)
    if vit5_run:
        validate_run_id(vit5_run)
    if batch_size < 1:
        raise ValueError('batch-size must be positive')
    rows = [adapt(r, 'inference', supervised=False) for r in source_payload(payload)]
    artifacts.reload()
    device = torch.device('cuda')
    paths = [(Path('/artifacts')/run_id/'best_correction', 'bartpho_corrector')]
    if vit5_run:
        paths.append((Path('/artifacts')/vit5_run/'best_correction', 'vit5_corrector'))
    paths.append((Path('/artifacts')/run_id/'best_classification', 'bartpho_classifier'))
    models = []
    for path, producer in paths:
        if path.with_suffix('.json').exists():
            path = checkpoint_path(path.parent, path.name)
        if not path.is_dir():
            raise FileNotFoundError(f'Missing S0 checkpoint: {path}. Select the Modal profile/Volume that owns the weights.')
        net, tok, cfg = load_checkpoint(path, device)
        if producer.startswith('bartpho') and (cfg['kind'] != 'bartpho' or not cfg['joint'] or cfg.get('classification_pooling') != 'attention'):
            raise ValueError('S0 requires BARTpho joint attention checkpoints')
        if producer == 'vit5_corrector' and (cfg['model_id'] != 'VietAI/vit5-base' or cfg['joint']):
            raise ValueError('S0 secondary must be correction-only ViT5-base')
        models.append((net, tok, cfg, producer, device))
    # Preserve the original S0 pool, including the classifier checkpoint's decoder.
    return predict_with_models(rows, models, classifier_index=len(models)-1,
                               selector='s0', batch_size=batch_size, nbest=1)


@app.local_entrypoint()
def main(input: str, output: str = 'outputs/predictions_s0.jsonl',
         run_id: str = 'bartpho-attention-final-full-42',
         vit5_run: str = 'vit5-base-attention-final-full-42',
         batch_size: int = 4, namespace: str = 'test', evaluate: bool = False):
    from vgec.io import read_records, write_jsonl, write_json
    from vgec.schemas import adapt
    from vgec.evaluation.diagnostics import evaluate as evaluate_predictions
    from vgec.evaluation.console import format_metrics
    validate_run_id(run_id)
    if vit5_run:
        validate_run_id(vit5_run)
    if batch_size < 1:
        raise ValueError('batch-size must be positive')
    records = read_records(input)
    payload = source_payload(records)
    references = [adapt(r, namespace, supervised=True) for r in records] if evaluate else None
    predictions = infer.remote(payload, run_id, vit5_run, batch_size)
    for row in predictions:
        row['uid'] = f"{namespace}:{row['id']}"
    write_jsonl(output, predictions)
    print(f'Wrote {len(predictions)} predictions to {output}')
    if evaluate:
        metrics = evaluate_predictions(predictions, references)
        write_json(Path(output).with_suffix('.metrics.json'), metrics)
        print(format_metrics(metrics))
