# VGEC 2026 Subtask 1 — full-data refit

This repository contains a Vietnamese grammatical error classification and correction system. The full-data refit obtained a **reported leaderboard score of 0.5802**. The earlier, pre-refit system obtained **0.5702**. These are submission scores reported by the authors, not scores calculated from the local development scorer.

The implementation of the 0.5802 refit is in [`experiments/full_data_refit_05702/`](experiments/full_data_refit_05702/). The directory name retains `05702` because the refit reuses the earlier system architecture. [`VGEC-0.5702/`](VGEC-0.5702/) contains the shared model implementation and the historical inference bundle; its standalone checkpoint specification refers to the earlier system.

**Data will be provided separately by the authors.** The instructions below explain where to place the supplied files and how to train and run the system. Checkpoint identities, model revisions, and checkpoint availability are documented in [`MODEL_CHECKPOINTS.md`](MODEL_CHECKPOINTS.md).

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

The authors will supply the data separately. To reproduce the audit and preprocessing, place the following files at these exact paths:

| Path relative to repository root | Purpose |
|---|---|
| `the_final_data (1).json` | Full training source: 7,153 unique sources, already including the 90 development sources |
| `dev.json` | Historical 90-source development set; used to audit the merge |
| `data/prepared_final/train.jsonl` | Historical 7,063-source pool excluding development sources; required by the merge audit |

The supplied files must match the checksums in [`data_audit.json`](experiments/full_data_refit_05702/data_audit.json) and [`prepared_manifest.json`](experiments/full_data_refit_05702/prepared_manifest.json). Preserve file bytes and encoding when downloading them.

**Do not concatenate `dev.json` onto the final JSON again.** All 90 development sources are already present exactly once. Their IDs differ between the two files, so the audit compares source content, labels, and references.

Preprocessing creates three training views:

| Prepared file under `experiments/full_data_refit_05702/data/` | Sources | Consumer |
|---|---:|---|
| `s0_full.jsonl` | 7,153 | Joint BARTpho classifier |
| `sub1_full.jsonl` | 7,153 | BamiBERT classifier |
| `sub1_bart_full.jsonl` | 7,097 | Dedicated BARTpho corrector |

The branches use their original, different reference filters. Joint BARTpho masks correction supervision for 108 sources without an eligible reference, while retaining classification supervision. The dedicated corrector excludes 56 sources without an eligible target. Label-0 examples provide identity targets according to each branch's preprocessing implementation.

The three model branches use the author-supplied final dataset. Dataset origin, synthetic generation sources, and usage details will accompany the separate data release; this README does not infer that provenance from file names or model code. The raw V1/V2 gold files used for EDA are not replacements for the 7,153-source training file.

## 4. Environment

Training and model inference use Modal with one **A100-40GB** GPU per worker. Local preparation and export run on CPU. The Modal image specifies:

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

An account with newly created empty volumes does not have the authors' checkpoints. Either train all three models or restore the supplied checkpoint packages as described in `MODEL_CHECKPOINTS.md`.

## 5. CPU preflight

After receiving and placing the data, run:

```powershell
python -X utf8 experiments/full_data_refit_05702/audit_data.py
python -X utf8 experiments/full_data_refit_05702/prepare_full_data.py
python -B -X utf8 -m unittest discover -s experiments/full_data_refit_05702 -p 'test_*.py' -v
```

These commands audit source coverage and annotations, prepare the branch-specific views, and test ID handling, probability averaging, candidate selection, fallback, and export. They do not launch model inference or training. Training subsequently verifies prepared file hashes and expected source counts.

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

Full training uses fixed seeds and the recorded configuration. A newly trained checkpoint has its own snapshot identity; retraining does not guarantee identical weights, logits, or leaderboard scores across environments. The original checkpoint packages are the route to reproducing the recorded trained system.

## 7. Inference and export

After all three full runs finish, or after restoring the original complete checkpoints, place the evaluation input at `private_test.json`. It is a JSON array containing typed `ID` values and raw `Input` strings. Gold fields are not needed by inference.

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

At the time these instructions were written, separate data download information and trained checkpoint download URLs had not yet been added. These must be supplied before an external reviewer can run the complete workflow. The commands above document the implementation; they are not a claim that the GitHub release already contains all required assets.

The experiment's `.gitignore` excludes `data/`, `prepared_manifest.json`, `data_audit.json`, and `outputs/`. When publishing the handoff, explicitly track the intended manifests or distribute them with the data package. Do not assume copying them into an ignored directory makes them part of a Git commit.
