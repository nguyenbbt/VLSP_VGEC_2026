# VGEC 2026 Subtask 1 — full-data refit

This repository contains a Vietnamese grammatical error classification and correction system. The two reported submission scores correspond to different training settings:

| Reported leaderboard score | Training setting |
|---|---|
| **0.5702** | Historical system **before merging the 90 development sources into the training pool**. |
| **0.5802** | Full-data refit **after including all 90 development sources in training**, with 7,153 unique training sources in total. |

The 0.5802 system keeps the same three-component architecture and 50/50 classifier ensemble as the earlier system, but trains the three models again from pretrained initialization. These are submission scores reported by the authors, not scores calculated from the local development scorer. After the merge, the historical development set is no longer held out.

The implementation of the 0.5802 refit is in [`experiments/full_data_refit_05702/`](experiments/full_data_refit_05702/). The directory name retains `05702` because the refit reuses the earlier system architecture. [`VGEC-0.5702/`](VGEC-0.5702/) contains the shared model implementation and the historical inference bundle; its standalone checkpoint specification refers to the earlier system.

Use this root README as the current reproduction guide. Nested README files retained from earlier handoffs may describe the historical baseline or the initial, not-yet-run refit setup.

**This is a source-code handoff for retraining from pretrained models. The authors' fine-tuned checkpoints are not required.** Training data are supplied separately through Google Drive; the organizers should use the data package/link provided with the submission materials. The instructions below explain where to place the data, train all three models on the organizers' own Modal account, and run inference using the checkpoints produced by those training jobs. Model revisions and historical run metadata are documented in [`MODEL_CHECKPOINTS.md`](MODEL_CHECKPOINTS.md).

## 1. System

The pipeline contains three independently fine-tuned components:

| Role | Pretrained model | Fine-tuning objective |
|---|---|---|
| BamiBERT classifier | `Qualcomm-AI-Research/BamiBERT` | Five-class classification with learned attention pooling |
| BARTpho joint classifier | `vinai/bartpho-syllable` | Correction plus 0.2 times classification loss; encoder attention pooling and a five-class head |
| Dedicated BARTpho corrector | `vinai/bartpho-syllable` | Correction-only training |

Classification averages probabilities in label order `[0, 1, 2, 3, 4]`:

```text
p_final = 0.5 * p_bami + 0.5 * p_bartpho_joint
label = argmax(p_final)
```

Temperature is 1. An exact tie selects the first maximum in label order. There is no additional calibration, class bias, or label-0 threshold.

The dedicated corrector receives only the source sentence, independently of the predicted label. It uses beam search with `num_beams=4`, `num_return_sequences=2`, `do_sample=False`, `length_penalty=1.0`, and `max_new_tokens=256`. The selector takes the first valid candidate in generator order and otherwise copies the raw Input. Validation and selection are implemented in [`infer_components.py`](VGEC-0.5702/scripts/infer_components.py) and the shared candidate code; they are format/completeness checks, not a semantic correctness judge.

Export follows these rules:

- Predicted label 0: `"Fixed Sentence": []`.
- Predicted labels 1–4: `"Fixed Sentence": [selected_text]`.
- A nonzero prediction with Copy fallback remains nonzero and exports `[raw Input]`.
- A correction identical to Input does not change the classification label.

This refit does not use R-Drop, Qwen, ViT5, or a reranker. Modules for other experiments remain in the shared source tree, but the refit entrypoints call only the three roles above.

## 2. Required repository layout

Run commands from the repository root. Keep these paths unchanged:

The refit directory must be `experiments/full_data_refit_05702/`, directly under the repository root. It must not be nested at `VGEC-0.5702/full_data_refit_05702/`: the documented commands and the inference entrypoint's local file mounts use the former path. Move the directory into the correct location without modifying the Python source files.

```text
README.md
MODEL_CHECKPOINTS.md
VGEC-0.5702/
  components/s0/
  components/sub1/
  scripts/
  checkpoints.json
  requirements.txt
experiments/full_data_refit_05702/
  audit_data.py
  prepare_full_data.py
  refit.py
  modal_refit.py
  private_infer.py
  modal_private.py
  prepared_manifest.json
  data_audit.json
  test_refit_preflight.py
  test_private_infer.py
  data/                         # supplied or generated prepared views
```

`VGEC-0.5702/checkpoints.json` must remain available because shared helpers import it. Its historical checkpoint identities must not be substituted for the refit checkpoint identities. Use `modal_private.py` for the refit inference workflow.

## 3. Data placement and preparation

For training, obtain the following three prepared views from the separately supplied Google Drive data package and place them under `experiments/full_data_refit_05702/data/`. If matching copies are already included in the checkout, they can be used directly:

| Prepared file | Sources | Consumer |
|---|---:|---|
| `s0_full.jsonl` | 7,153 | Joint BARTpho classifier |
| `sub1_full.jsonl` | 7,153 | BamiBERT classifier |
| `sub1_bart_full.jsonl` | 7,097 | Dedicated BARTpho corrector |

Keep `prepared_manifest.json` in the parent experiment directory. The trainer checks the actual file hashes and source counts before launching a training job. Prepared files must match that manifest; preserve their original encoding and line endings when downloading them.

To additionally reproduce the raw-data audit and preprocessing, place these original inputs at the following paths. This additional audit route is not needed to train from the already supplied, checksum-verified prepared views:

| Path relative to repository root | Purpose |
|---|---|
| `the_final_data (1).json` | Full training source: 7,153 unique sources, already including the 90 development sources |
| `dev.json` | Historical 90-source development set; used to audit the merge |
| `data/prepared_final/train.jsonl` | Historical 7,063-source pool excluding development sources; required by the merge audit |

The original inputs must match the checksums in [`data_audit.json`](experiments/full_data_refit_05702/data_audit.json) and [`prepared_manifest.json`](experiments/full_data_refit_05702/prepared_manifest.json). Preserve file bytes and encoding when downloading them. The 7,063-source JSONL is an audit dependency, not an additional dataset to concatenate onto the final training file.

**Do not concatenate `dev.json` onto the final JSON again.** All 90 development sources are already present exactly once. Their IDs differ between the two files, so the audit compares source content, labels, and references.

The branches use their original, different reference filters. Joint BARTpho masks correction supervision for 108 sources without an eligible reference, while retaining classification supervision. The dedicated corrector excludes 56 sources without an eligible target. Label-0 examples provide identity targets according to each branch's preprocessing implementation.

The three model branches use the author-supplied final dataset. Dataset origin, synthetic generation sources, and usage details belong to the separate data documentation; this README does not infer that provenance from file names or model code. The raw V1/V2 gold files used for EDA are not replacements for the 7,153-source training file or its prepared views.

## 4. Environment

The current training and inference entrypoints explicitly request one **A10-24GB** GPU per worker. This is the hardware setting implemented in the supplied Python code; an A10-24GB setting is not the documented reproduction configuration. Local preparation and export run on CPU. The Modal image specifies:

```text
Python 3.11
torch==2.8.0
transformers==4.57.1
sentencepiece==0.2.1
safetensors==0.6.2
numpy==2.2.6
PyYAML==6.0.3
```

The following commands use Windows PowerShell. Create a Python 3.11 environment from the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r VGEC-0.5702/requirements.txt "modal>=1.6,<2"
python -m pip install -e VGEC-0.5702/components/s0 -e VGEC-0.5702/components/sub1
$env:PYTHONIOENCODING = 'utf-8'
```

Authenticate Modal using your own account. In the commands below, replace `YOUR_MODAL_PROFILE` with that account's configured profile:

```powershell
$vgecProfile = 'YOUR_MODAL_PROFILE'
```

The entrypoints use two account-local volumes:

- `vgec-05702-refit-artifacts`: trained checkpoints and inference artifacts.
- `vgec-05702-refit-hf-cache`: Hugging Face pretrained cache.

Use your own Modal account and run all three training jobs below. The runner creates the volumes as needed and writes your new checkpoints there. No access to the authors' Modal account, existing volumes, or fine-tuned checkpoint files is needed for this training workflow. Internet access is needed to download the pinned pretrained Hugging Face models unless they are already cached.

## 5. CPU preflight

After placing the three prepared training views and their manifest, run the CPU tests:

```powershell
python -B -X utf8 -m unittest discover -s experiments/full_data_refit_05702 -p 'test_*.py' -v
```

The tests verify the real prepared data hashes/counts and fixed stopping epochs, as well as fixture-based checks for the development merge, ID handling, probability averaging, candidate selection, fallback, checkpoint validation, and export. They do not launch model inference or training. Training subsequently verifies the prepared file hashes and expected source counts again.

Optional: if the three original audit inputs listed in Section 3 have also been supplied, audit and regenerate the views first:

```powershell
python -X utf8 experiments/full_data_refit_05702/audit_data.py
python -X utf8 experiments/full_data_refit_05702/prepare_full_data.py
python -B -X utf8 -m unittest discover -s experiments/full_data_refit_05702 -p 'test_*.py' -v
```

## 6. Train the full-data refit

Each model starts independently from its pinned pretrained revision. The fixed stopping schedule preserves the epochs chosen before refitting:

| Role | Stop epoch | Scheduler horizon | Expected optimizer steps |
|---|---:|---:|---:|
| `bami` | 14 | 15 | 3,136 |
| `s0` | 10 | 12 | 4,480 |
| `bart` | 6 | 8 | 2,664 |

There is no new development-based checkpoint selection after merging development sources into training. Pointers named `best` or `best_classification` refer to the completed fixed-epoch checkpoint in this refit.

First run the two-step smoke checks sequentially. Each must return `checkpoint_reload: passed`:

```powershell
modal run --profile $vgecProfile experiments/full_data_refit_05702/modal_refit.py --role bami --run-id smoke-bami-05702 --smoke-steps 2
modal run --profile $vgecProfile experiments/full_data_refit_05702/modal_refit.py --role s0 --run-id smoke-s0-05702 --smoke-steps 2
modal run --profile $vgecProfile experiments/full_data_refit_05702/modal_refit.py --role bart --run-id smoke-bart-05702 --smoke-steps 2
```

After all smoke checks pass, run the full jobs sequentially, waiting for each command to complete successfully:

```powershell
modal run --profile $vgecProfile experiments/full_data_refit_05702/modal_refit.py --role bami --run-id bami-full-05702
modal run --profile $vgecProfile experiments/full_data_refit_05702/modal_refit.py --role s0 --run-id s0-full-05702
modal run --profile $vgecProfile experiments/full_data_refit_05702/modal_refit.py --role bart --run-id bart-full-05702
```

To resume an interrupted run, repeat its command with the same run ID and unchanged code, configuration, and data. The runner restores an existing valid checkpoint of that experiment. Smoke checkpoints are separate from full checkpoints. Do not run duplicate jobs for the same role/run ID concurrently.

Full training uses fixed seeds and the recorded configuration. It reproduces the training procedure and system architecture without the authors' fine-tuned weights. A newly trained checkpoint has its own snapshot identity; retraining does not guarantee byte-identical weights, logits, or the same leaderboard score across environments.

## 7. Inference and export

After all three full training jobs finish on your Modal account, place the evaluation input at `private_test.json`. It is a JSON array containing typed `ID` values and raw `Input` strings. Gold fields are not needed by inference. The entrypoint reads the newly trained checkpoints from your account's artifact volume; the authors' original checkpoints are not required.

```powershell
modal run --profile $vgecProfile experiments/full_data_refit_05702/modal_private.py
```

For a different source-only input file and a new output namespace:

```powershell
modal run --profile $vgecProfile experiments/full_data_refit_05702/modal_private.py --input-path path/to/input.json --output-id reproduction-001
```

The CPU preflight checks completion, data/config/code identity, checkpoint metadata, tokenizer/weight hashes, and trained S0 pooling/head files before requesting a GPU. Inference processes BamiBERT, joint BARTpho, and the dedicated corrector sequentially. Matching complete caches are reused.

The default local output directory is:

```text
experiments/full_data_refit_05702/outputs/ensemble50-full-private-v1/
  bami.json
  s0.json
  bart.json
  predictions.jsonl
  identity.json
  manifest.json
  results.json
  ensemble50_full_refit_bart_top1.zip
```

The ZIP contains only `results.json` at its root. Export preserves typed IDs, source order, and coverage. It validates source hashes, classifier probabilities, and the correction trace. Scripts do not submit results to the leaderboard.

## 8. Evaluation and release status

The historical development set contains 90 sources. All are optimizer-exposed in the full-data refit, so scores on those sources must not be presented as held-out evaluation of the refitted models. Local GLEU/ROUGE implementations are diagnostic proxies unless matched to the organizers' scorer.

The reported 0.5802 submission has 2,034 rows. Its recorded JSON and ZIP hashes are in `MODEL_CHECKPOINTS.md` and the saved output manifest. The score-to-submission association is author-reported; the local metadata alone is not an upload or scoring log.

The training data are distributed separately via Google Drive, as provided to the organizers. The source-code handoff does not require distribution of the authors' fine-tuned weights: training creates the weights needed for subsequent inference. Before starting, confirm the correct repository directory layout, the three prepared data files, a valid Modal profile, and access to the pinned pretrained models.

If the development workspace's experiment-level `.gitignore` is also copied into the repository, it excludes `data/`, `prepared_manifest.json`, `data_audit.json`, and `outputs/`. Explicitly track the required manifests and intended prepared views, or distribute the views with the data package. Do not assume copying files into an ignored directory makes them part of a Git commit.
