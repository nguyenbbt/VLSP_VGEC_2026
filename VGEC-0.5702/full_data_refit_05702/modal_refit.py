"""Modal entrypoint for isolated, fresh 0.5702 full-data refits.

Importing this module and local preflight do not allocate a GPU.
"""

import json
import re
from pathlib import Path

import modal

def resolve_paths(entry_file: str, is_local: bool) -> tuple[Path, Path]:
    # Modal imports the entrypoint from /root/modal_refit.py in the container;
    # the project files themselves are mounted under /workspace.
    here = (Path(entry_file).resolve().parent if is_local else
            Path("/workspace/experiments/full_data_refit_05702"))
    root = here.parents[1]
    return here, root


HERE, ROOT = resolve_paths(__file__, modal.is_local())
ARTIFACT_ROOT = Path("/artifacts/vgec-full-data-refit-05702")
# Volumes belong to the Modal workspace selected by --profile. Give this refit
# its own volumes so a newly added account does not depend on historical runs.
artifacts = modal.Volume.from_name("vgec-05702-refit-artifacts", create_if_missing=True)
cache = modal.Volume.from_name("vgec-05702-refit-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0", "transformers==4.57.1", "sentencepiece==0.2.1",
                 "safetensors==0.6.2", "numpy==2.2.6", "PyYAML==6.0.3")
    .env({"HF_HOME": "/hf-cache", "TOKENIZERS_PARALLELISM": "false",
          "PYTHONPATH": "/workspace/VGEC-0.5702/components/s0/src:/workspace/VGEC-0.5702/components/sub1/src:/workspace/experiments/full_data_refit_05702"})
    .add_local_dir(ROOT / "VGEC-0.5702/components/s0/src", remote_path="/workspace/VGEC-0.5702/components/s0/src")
    .add_local_dir(ROOT / "VGEC-0.5702/components/s0/configs", remote_path="/workspace/VGEC-0.5702/components/s0/configs")
    .add_local_dir(ROOT / "VGEC-0.5702/components/sub1/src", remote_path="/workspace/VGEC-0.5702/components/sub1/src")
    .add_local_dir(ROOT / "VGEC-0.5702/components/sub1/configs", remote_path="/workspace/VGEC-0.5702/components/sub1/configs")
    .add_local_dir(ROOT / "VGEC-0.5702/components/sub1/prompts", remote_path="/workspace/VGEC-0.5702/components/sub1/prompts")
    .add_local_dir(HERE, remote_path="/workspace/experiments/full_data_refit_05702")
)

app = modal.App("vgec-05702-full-data-refit")


def check_args(role, run_id, smoke_steps):
    if role not in {"bami", "s0", "bart"}:
        raise ValueError("Choose exactly one 0.5702 component: bami, s0, bart")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("Unsafe run ID")
    if smoke_steps < 0:
        raise ValueError("Negative smoke steps")
    if smoke_steps and not run_id.startswith("smoke-"):
        raise ValueError("Smoke and full refit must have different run IDs")
    if not smoke_steps and run_id.startswith("smoke-"):
        raise ValueError("Full refit cannot use smoke run ID")


@app.function(image=image, gpu="A100-40GB", timeout=24 * 60 * 60,
              volumes={"/artifacts": artifacts, "/hf-cache": cache}, scaledown_window=2)
def fit(role: str, run_id: str, smoke_steps: int = 0):
    import gc
    import sys
    import torch

    check_args(role, run_id, smoke_steps)
    sys.path.insert(0, "/workspace/experiments/full_data_refit_05702")
    from refit import inputs, train_s0, train_sub1

    artifacts.reload()
    inputs(role)  # Recheck the exact prepared bytes inside the remote image.
    out = ARTIFACT_ROOT / run_id
    out.mkdir(parents=True, exist_ok=True)
    try:
        state = (train_s0(out, smoke_steps, artifacts.commit) if role == "s0"
                 else train_sub1(role, out, smoke_steps, artifacts.commit))
        if not state["done"]:
            raise RuntimeError("Refit stopped before its fixed epoch")
        gc.collect()
        torch.cuda.empty_cache()
        if role == "s0":
            from vgec.models.seq2seq import load_checkpoint
            model, tokenizer, cfg = load_checkpoint(out / "best_classification", torch.device("cuda"))
            if cfg["classification_pooling"] != "attention" or model.attention_pool is None:
                raise RuntimeError("S0 attention pooling reload failed")
            del model, tokenizer
        else:
            from vgec_sub1.checkpoints import Store
            from vgec_sub1.config import load
            from vgec_sub1.models import Runtime
            manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
            store = Store(out / "checkpoints" / f"{role}-full", manifest["identity"])
            chosen = store.resolve("best")
            if chosen is None:
                raise RuntimeError("Sub1 trained checkpoint missing")
            cfg = load(Path("/workspace/VGEC-0.5702/components/sub1/configs/sub1.yaml"))
            runtime = Runtime(cfg, role, training=False, checkpoint=chosen)
            del runtime
        artifacts.commit()
        cache.commit()
        return dict(role=role, run_id=run_id, epoch=state["epoch"],
                    optimizer_steps=state["step"], checkpoint_reload="passed",
                    smoke_only=bool(smoke_steps))
    finally:
        artifacts.commit()
        cache.commit()


@app.local_entrypoint()
def main(role: str, run_id: str, smoke_steps: int = 0):
    import sys

    check_args(role, run_id, smoke_steps)
    sys.path.insert(0, str(HERE))
    from refit import inputs

    inputs(role)  # Local hash/count preflight before requesting a GPU.
    print(json.dumps(fit.remote(role, run_id, smoke_steps), ensure_ascii=False, indent=2))
