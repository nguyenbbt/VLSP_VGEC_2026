# Model information and recorded run metadata — VGEC 0.5802

This document identifies the **full-data refit** of the earlier 0.5702 system. **0.5702 is the reported score before merging development data into training; 0.5802 is the reported score after merging all 90 development sources into the 7,153-source training pool.** The repository is a source-code handoff for training from pretrained models on the organizers' own Modal account. The authors' fine-tuned weights are not required for that workflow. Recorded checkpoint identities below are historical provenance, not required downloads.

## 1. Models and pinned versions

| Component | Pretrained model ID | Pretrained revision | Approximate model size |
|---|---|---|---|
| BamiBERT classifier | `Qualcomm-AI-Research/BamiBERT` | `57bc1340debbe4e348ec549047a763caebe4a977` | 102M parameters, including pooling/head |
| BARTpho joint classifier | `vinai/bartpho-syllable` | `36eee8b4d648dd99da56462edcda3c5c97f7f3de` | 396M parameters, including decoder and classification modules |
| BARTpho dedicated corrector | `vinai/bartpho-syllable` | `36eee8b4d648dd99da56462edcda3c5c97f7f3de` | 396M parameters |

Sizes are approximate architecture sizes, not counts of only trainable adapters. Historical parameter reports for the matching classifier architectures record 102,365,189 parameters for BamiBERT and 395,821,061 for joint BARTpho. Those reports are not newly measured counts of the refit snapshots. The refit loaders implement parameter counting and model-size guards; final run-level count reports should accompany the checkpoint release where available.

Pretrained resources:

- [BamiBERT at the pinned revision](https://huggingface.co/Qualcomm-AI-Research/BamiBERT/tree/57bc1340debbe4e348ec549047a763caebe4a977).
- [BARTpho-syllable at the pinned revision](https://huggingface.co/vinai/bartpho-syllable/tree/36eee8b4d648dd99da56462edcda3c5c97f7f3de).

The training workflow downloads these pinned pretrained resources and fine-tunes them. The pretrained links do not contain the authors' fine-tuned weights. BamiBERT and the dedicated BARTpho Runtime also resolve their pinned pretrained configuration/base when loading a locally produced state dictionary; retain access to the pinned resources or their Hugging Face cache.

BamiBERT uses encoder last hidden states, learned attention pooling excluding padding and special tokens, and a five-class head. Joint BARTpho uses encoder states, learned attention pooling excluding padding, and a five-class head. Its decoder participates in correction supervision during training; classification inference uses the encoder, pooling, and head. The dedicated corrector is a separate full encoder–decoder model. The two BARTpho components do not share their fine-tuned weights.

## 2. Exact refit checkpoints

Paths below are relative to the artifact root inside the Modal worker:

```text
/artifacts/vgec-full-data-refit-05702/
```

| Role | Run ID | Checkpoint relative to artifact root | Epoch | Optimizer steps |
|---|---|---|---:|---:|
| BamiBERT | `bami-full-05702` | `bami-full-05702/checkpoints/bami-full/step-00003136-951a1a599098` | 14 | 3,136 |
| Joint BARTpho | `s0-full-05702` | `s0-full-05702/best_classification` | 10 | 4,480 |
| Dedicated BARTpho | `bart-full-05702` | `bart-full-05702/checkpoints/bart-full/step-00002664-7dd5bd1c75fb` | 6 | 2,664 |

The artifact volume is `vgec-05702-refit-artifacts`. Its filesystem root corresponds to `/artifacts` on the worker. The pretrained cache volume is `vgec-05702-refit-hf-cache`.

These paths identify recorded artifacts in the authors' Modal workspace. They are not public download URLs and are not automatically accessible from another Modal account.

## 3. Reproduce by training from pretrained models

No author-provided fine-tuned checkpoint download is required. Follow [`README.md`](README.md), obtain the training data from the Google Drive package supplied to the organizers, and train roles `bami`, `s0`, and `bart` on your own Modal account. The training jobs create the checkpoint files and run metadata needed by inference in your account's artifact volume.

The historical snapshot names in Section 2 identify the original run. Newly trained runs may produce different snapshot names and weight hashes; the refit inference workflow validates and uses the completed checkpoints of those new runs rather than hardcoding the historical snapshot names. The architecture, data checksums, stopping schedule, configuration, and ensemble/export policy remain as documented. Retraining does not guarantee byte-identical weights or an identical leaderboard score.

## 4. Checkpoint layout created by training

The training jobs create run-level metadata, weight files, tokenizer files, and trained classifier/pooling modules. The inference entrypoint checks those files before loading a model. The layout below shows the original run under `vgec-full-data-refit-05702/` in the artifact volume; newly trained snapshot names may differ:

```text
bami-full-05702/
  run_manifest.json
  checkpoints/bami-full/
    pointers.json
    step-00003136-951a1a599098/
      complete.json
      identity.json
      progress.json
      model.pt
      tokenizer/
        tokenizer.json
        tokenizer_config.json
        special_tokens_map.json

s0-full-05702/
  run_manifest.json
  identity.json
  best_classification/
    model.safetensors
    classifier.pt
    attention_pool.pt
    config.json
    generation_config.json
    resolved_config.json
    run_manifest.json
    sentencepiece.bpe.model
    dict.txt
    tokenizer_config.json
    special_tokens_map.json

bart-full-05702/
  run_manifest.json
  checkpoints/bart-full/
    pointers.json
    step-00002664-7dd5bd1c75fb/
      complete.json
      identity.json
      progress.json
      model.pt
      tokenizer/
        sentencepiece.bpe.model
        dict.txt
        tokenizer_config.json
        special_tokens_map.json
```

Keep the files produced by training in place. For BamiBERT and the dedicated corrector, optimizer state `training.pt` is skipped by inference verification and is not loaded for prediction; resuming training requires the relevant training state.

Do not manually replace checkpoint metadata, configuration, or pointers after training. S0's trained `classifier.pt` and `attention_pool.pt` are required; the loader must not substitute randomly initialized modules. See [`checkpoint_info`](experiments/full_data_refit_05702/private_infer.py) for the actual validation contract.

## 5. Recorded integrity information

The complete recorded inference identity is stored at:

```text
experiments/full_data_refit_05702/outputs/ensemble50-full-private-v1/identity.json
```

The corresponding output manifest is:

```text
experiments/full_data_refit_05702/outputs/ensemble50-full-private-v1/manifest.json
```

The identity contains all recorded checkpoint/tokenizer file hashes, the S0 training identity object, and inference code hashes. Selected weight hashes are reproduced below:

| Component/file | SHA256 |
|---|---|
| BamiBERT `model.pt` | `a496d84d11f59ecf4ef782d72522ff606943ac4583d5b492b068a72b122660ff` |
| S0 `model.safetensors` | `8098e5df511327506df2c7a8bb8cca4bd96919e97370759162cff88ee23d8cf3` |
| S0 `classifier.pt` | `b10772996a4158a6e5fdd86b7b13f2d21285a43757f0be554c68982c75627f67` |
| S0 `attention_pool.pt` | `de03f6e590039dd8878ce96bbb3cb305ff187430cc2eb2dbf3f6af112d1fabb5` |
| Dedicated BARTpho `model.pt` | `5c33d04f8a136dcf4d60a6780a9bb61c0dd42c352302d44b151e885cab32ad1e` |

These are hashes recorded by the completed inference job. This documentation task does not independently download or rehash the remote weights.

Training data checksums:

| File | SHA256 |
|---|---|
| `the_final_data (1).json` | `69caf7b8b5508176ebb0f169d24bd7c68104b2a047eaf36e2dbd1d6087041840` |
| `s0_full.jsonl` | `262eb0d41683d27992e9e9bab075d19d1851545b769cc491750fe47576cfac71` |
| `sub1_full.jsonl` | `f4c1e413812db321c77fb355d25d6963c58d8d891272912dd6166fbecc487df2` |
| `sub1_bart_full.jsonl` | `2d7e03aaebdc27b3a5c253f82c71a52587b0068bdc812317aa3fb45e1a837943` |

Recorded private submission output:

| Item | Value |
|---|---|
| Sources | 2,034 |
| Input file SHA256 | `0762703e83e59feac963b2e63f2986a351340cf32da07ce2db7d151a8abe6045` |
| `results.json` SHA256 | `87688cde428418d2ff24594c2645110790f24362e42c33185a345bb2e9e87985` |
| `ensemble50_full_refit_bart_top1.zip` SHA256 | `c9e0499d5021f91e2d911f166f2a737720b463ff8aaa080a7d3cc29c13c83639` |

The ZIP contains only root-level `results.json`. The export runner records `leaderboard_submitted=false` because it does not submit automatically; that field does not describe any subsequent manual upload by the authors.

## 6. Distinguishing the historical baseline

[`VGEC-0.5702/checkpoints.json`](VGEC-0.5702/checkpoints.json) identifies the **pre-refit** system:

- BamiBERT: `step-00002632-551882ef4edc`.
- Joint classifier: `bartpho-attention-final-full-42/best_classification`.
- Dedicated corrector: `step-00002238-09c10cc9af21`.

Those are not the 0.5802 refit checkpoints. Use the refit workflow in the root README for reproduction of the reported full-data system.
