import re
import math
from pathlib import Path
import yaml
from .io import digest

ROLES = ('bami', 'bart', 'vit5', 'qwen_corrector', 'reranker')
MODEL_IDS = dict(bami='Qualcomm-AI-Research/BamiBERT', bart='vinai/bartpho-syllable',
                 vit5='VietAI/vit5-base', qwen_corrector='Qwen/Qwen3.5-4B', reranker='Qwen/Qwen3-Reranker-4B')


class UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    result = {}
    for key, value in node.value:
        key = loader.construct_object(key, deep=deep)
        if key in result:
            raise ValueError('Duplicate YAML key: '+str(key))
        result[key] = loader.construct_object(value, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _keys(actual, expected, prefix=''):
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise ValueError('Expected config object: '+prefix)
        unknown = set(actual)-set(expected)
        missing = set(expected)-set(actual)
        if unknown or missing:
            raise ValueError(f'Config keys at {prefix}: unknown={sorted(unknown)}, missing={sorted(missing)}')
        for key in expected:
            _keys(actual[key], expected[key], prefix+'.'+key)


def load(path='configs/sub1.yaml', root=None):
    path = Path(path).resolve()
    root = Path(root).resolve() if root else path.parent.parent
    cfg = yaml.load(path.read_text(encoding='utf-8'), Loader=UniqueLoader)
    template = yaml.safe_load(Path(__file__).with_name('defaults.yaml').read_text(encoding='utf-8'))
    _keys(cfg, template)
    if cfg['parameter_limit_exclusive'] != 8_000_000_000:
        raise ValueError('Sub1 requires exclusive 8B parameter gate')
    if cfg['data']['mode'] != 'supplied_final_exploratory':
        raise ValueError('This delivered preset implements supplied-final/processed-gold; use a separate version for other data modes')
    if cfg['data']['grouped_folds'] != 3 or cfg['data']['inner_refit']:
        raise ValueError('Default delivery requires 3 folds with inner_refit=false')
    if cfg['generation']['family_limits'] != dict(bart=2, vit5=2, qwen=2, copy=1):
        raise ValueError('Sub1 pool must be 2+2+2+1')
    if cfg['verification']['learned_edit_confidence'] or cfg['verification']['edit_recombination']:
        raise ValueError('Sub1 verifier is deterministic; no learned confidence/recombination')
    if cfg['generation']['predicted_label_to_correctors'] or cfg['generation']['label_zero_short_circuit']:
        raise ValueError('Correctors must be source-only and process label zero')
    if cfg['selection']['copy_bonus'] != 0 or cfg['selection']['absolute_score_threshold'] is not None:
        raise ValueError('Sub1 selection uses raw score without bonus/threshold')
    if cfg['training_profile'] != 'fresh_train' or cfg['runtime']['world_size'] != 1:
        raise ValueError('Delivered Modal preset is fresh training on one A100 per worker')
    if cfg['runtime']['gpu'] not in ('A100-40GB', 'A100-80GB'):
        raise ValueError('Select explicit A100-40GB or A100-80GB')
    if cfg['runtime']['training_slice_soft_seconds'] >= cfg['runtime']['training_slice_hard_seconds']:
        raise ValueError('Soft deadline must precede hard timeout')
    if cfg['runtime']['checkpoint_interval_seconds']<=0:
        raise ValueError('Checkpoint interval must be positive')
    if cfg['runtime']['training_slice_hard_seconds']!=21600 or cfg['runtime']['training_slice_soft_seconds']>19800 or cfg['runtime']['gpus_per_worker']!=1:
        raise ValueError('Modal workers use one GPU, hard 6h and soft at most 5h30')
    if cfg['runtime']['micro_batch_per_gpu'] is not None or cfg['runtime']['dataloader_num_workers']!=0:
        raise ValueError('Microbatch comes from the A100 benchmark; cached sources use the single-process loader')
    if cfg['runtime']['precision'] not in ('auto','bf16') or cfg['runtime']['device'] not in ('auto','cuda'):
        raise ValueError('A100 preset uses BF16 autocast and automatic/CUDA device selection')
    if cfg['verification']['unit']!='unicode_codepoint' or cfg['verification']['soft_risks_are_hard_filters']:
        raise ValueError('Codepoint verifier and soft-only risks are required')
    if cfg['selection']['allow_degraded'] or cfg['selection']['method']!='reranker_margin_top1':
        raise ValueError('Explicitly create a separate ablation for degraded/alternative selection')
    for role,c in cfg['models'].items():
        if c.get('truncate',False) or c.get('gold_polish_enabled',False): raise ValueError('Truncation/gold-polish is not enabled in this fresh preset')
        if c.get('checkpoint_dir') or c.get('adapter_dir'): raise ValueError('This fresh preset does not inherit old checkpoints')
    if cfg['models']['bami']['rdrop_lambda']!=0 or cfg['models']['bami']['calibration']!='none':
        raise ValueError('Default classifier has no R-Drop/calibration')
    if cfg['models']['reranker']['learned_calibration'] or cfg['models']['qwen_corrector']['enable_thinking']:
        raise ValueError('Calibration/thinking is outside this preset')
    if cfg['models']['bami']['pooling']!='masked_attention' or cfg['models']['bami']['num_labels']!=5:
        raise ValueError('Bami requires masked attention and five labels')
    for role in ('bart','vit5'):
        c=cfg['models'][role]
        if c['do_sample'] or c['num_return_sequences']!=2 or c['num_beams']<2:
            raise ValueError('Seq2seq correction pool requires two beam outputs')
    c=cfg['models']['qwen_corrector']
    if c['packing'] or not c['assistant_only_loss'] or c['candidate_count']!=2 or c['decoding_modes']!=['greedy','sampling']:
        raise ValueError('Qwen uses unpacked assistant-only SFT and greedy + sampling candidates')
    if cfg['scoring']['gleu_order']!=4 or cfg['scoring']['gleu_reference_samples']!=500 or cfg['scoring']['gleu_seed']!=13:
        raise ValueError('Default scorer is locked to GLEU4/500 samples/seed13')
    for role in ROLES:
        model = cfg['models'][role]
        key = 'rebuild_pretrained' if role in ('bart', 'vit5') else 'pretrained'
        rev = model['rebuild_revision' if role in ('bart', 'vit5') else 'revision']
        if model[key] != MODEL_IDS[role] or not isinstance(rev, str) or not re.fullmatch('[a-f0-9]{40}', rev):
            raise ValueError('Wrong model or unpinned revision: '+role)
    cfg['_root'] = str(root)
    cfg['_config_path'] = str(path.relative_to(root))
    cfg['_config_hash'] = digest({k: v for k, v in cfg.items() if not k.startswith('_')})
    for key in ('final_inner_fraction','inner_validation_fraction'):
        if not 0<cfg['data'][key]<1: raise ValueError('Invalid data fraction: '+key)
    if not 0<cfg['models']['reranker']['rank_fit_fraction']<1: raise ValueError('Invalid rank fit fraction')
    if cfg['runtime']['training_slice_soft_seconds']<=0: raise ValueError('Soft deadline must be positive')
    if cfg['models']['qwen_corrector']['schema_retry_count'] not in (0,1): raise ValueError('Schema retry count must be zero or one')
    if cfg['models']['reranker']['min_audited_sources']<100 or cfg['models']['reranker']['min_audited_ordered_pairs']<200:
        raise ValueError('Human audit minimums cannot be weakened')
    for role in ROLES:
        rc=role_config(cfg,role)
        for key in ('global_batch','epochs','patience','max_length'):
            if type(rc[key]) is not int or rc[key]<=0: raise ValueError('Invalid '+role+'/'+key)
        for key in ('lr','clip'):
            if not isinstance(rc[key],(int,float)) or not math.isfinite(rc[key]) or rc[key]<=0:
                raise ValueError('Invalid '+role+'/'+key)
        if not 0<=rc['warmup']<1: raise ValueError('Invalid warmup')
    return cfg


def path(cfg, value):
    base = Path(cfg['_root']).resolve()
    result = (base/value).resolve()
    if not result.is_relative_to(base):
        raise ValueError('Path escapes project: '+str(value))
    return result


def output(cfg):
    return path(cfg, cfg['output_dir'])


def role_config(cfg, role):
    c = cfg['models'][role]
    common = dict(role=role, model_id=MODEL_IDS[role],
                  revision=c.get('rebuild_revision', c.get('revision')),
                  limit=cfg['parameter_limit_exclusive'], seed=cfg['seed'],
                  weight_decay=cfg['runtime']['weight_decay'], warmup=cfg['runtime']['warmup_ratio'],
                  clip=cfg['runtime']['grad_clip_norm'], quantize=False,
                  checkpointing=cfg['runtime']['gradient_checkpointing'])
    if role == 'bami':
        common.update(lr=c['encoder_lr'], head_lr=c['head_lr'], global_batch=c['global_batch'],
                      epochs=c['max_epochs'], patience=c['patience'], max_length=c['source_max_tokens'])
    elif role in ('bart', 'vit5'):
        common.update(lr=c['lr'], global_batch=c['global_batch'],
                      epochs=c['max_epochs'], patience=c['patience'],
                      max_length=c['source_max_tokens'], target_length=c['target_max_tokens'],
                      prefix='' if role == 'bart' else c['rebuild_source_prefix'], beams=c['num_beams'])
    elif role == 'qwen_corrector':
        common.update(lr=c['provided_final_lr'], global_batch=c['provided_final_global_batch'],
                      epochs=c['provided_final_max_epochs'], patience=c['patience'], max_length=c['max_sequence_tokens'],
                      max_new_tokens=c['max_new_tokens'], lora_r=c['lora_rank'], lora_alpha=c['lora_alpha'],
                      lora_dropout=c['lora_dropout'], prompt=path(cfg, 'prompts/correct_v1.txt').read_text(encoding='utf-8'))
    else:
        common.update(lr=c['lr'], global_batch=c['global_source_batch'], epochs=c['max_epochs'],
                      patience=c['patience'], max_length=c['max_pair_tokens'], lora_r=c['lora_rank'],
                      lora_alpha=c['lora_alpha'], lora_dropout=c['lora_dropout'], binary_weight=c['binary_loss_weight'],
                      prompt=path(cfg, 'prompts/rank_v1.txt').read_text(encoding='utf-8'))
    return common
