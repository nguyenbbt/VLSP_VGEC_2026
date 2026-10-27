"""Validate completed refits on CPU, infer private with one GPU, export locally."""
import json
from pathlib import Path
import sys

import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/workspace")
HERE = ROOT / "experiments/full_data_refit_05702"
sys.path.insert(0, str(HERE))

volume = modal.Volume.from_name("vgec-05702-refit-artifacts", create_if_missing=False)
hf_cache = modal.Volume.from_name("vgec-05702-refit-hf-cache", create_if_missing=False)
ARTIFACT_ROOT = Path("/artifacts/vgec-full-data-refit-05702")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.8.0", "transformers==4.57.1", "sentencepiece==0.2.1",
                      "safetensors==0.6.2", "numpy==2.2.6", "PyYAML==6.0.3")
         .env({"HF_HOME": "/hf-cache", "TOKENIZERS_PARALLELISM": "false",
               "PYTHONPATH": "/workspace/VGEC-0.5702/components/s0/src:/workspace/VGEC-0.5702/components/sub1/src:/workspace/experiments/full_data_refit_05702"})
         .add_local_dir(ROOT / "VGEC-0.5702/components/s0/src", remote_path="/workspace/VGEC-0.5702/components/s0/src")
         .add_local_dir(ROOT / "VGEC-0.5702/components/s0/configs", remote_path="/workspace/VGEC-0.5702/components/s0/configs")
         .add_local_dir(ROOT / "VGEC-0.5702/components/sub1/src", remote_path="/workspace/VGEC-0.5702/components/sub1/src")
         .add_local_dir(ROOT / "VGEC-0.5702/components/sub1/configs", remote_path="/workspace/VGEC-0.5702/components/sub1/configs")
         .add_local_dir(ROOT / "VGEC-0.5702/components/sub1/prompts", remote_path="/workspace/VGEC-0.5702/components/sub1/prompts")
         .add_local_dir(ROOT / "VGEC-0.5702/scripts", remote_path="/workspace/VGEC-0.5702/scripts")
         .add_local_file(ROOT / "VGEC-0.5702/checkpoints.json", remote_path="/workspace/VGEC-0.5702/checkpoints.json"))
# Mount code and the training identity manifest only, not gold training/dev rows.
for name in ("private_infer.py", "modal_private.py", "refit.py", "prepared_manifest.json"):
    image = image.add_local_file(HERE / name, remote_path=f"/workspace/experiments/full_data_refit_05702/{name}")
app = modal.App("vgec-05702-full-refit-private")


@app.function(image=image, volumes={"/artifacts": volume}, cpu=2, memory=4096, timeout=1200)
def preflight(sources: list, input_sha256: str, output_id: str):
    from private_infer import (ROLES, RUNS, cached_rows, checkpoint_info, digest,
                               job_identity, read_json, safe_id, source_only, write_json)
    volume.reload()
    sources = source_only(sources)
    proofs = {role: checkpoint_info(ARTIFACT_ROOT, role, RUNS[role]) for role in ROLES}
    identity = job_identity(sources, input_sha256, proofs)
    directory = ARTIFACT_ROOT / "private-inference" / safe_id(output_id)
    lock = directory / "identity.json"
    if lock.exists() and read_json(lock) != identity:
        raise ValueError("Inference output ID already belongs to different checkpoints/input/code")
    write_json(lock, identity)
    volume.commit()
    cached = {role: cached_rows(directory, role, identity, sources) for role in ROLES}
    print(json.dumps(dict(stage="cpu_preflight_passed", rows=len(sources), identity=digest(identity),
                          checkpoints={r: p["checkpoint"] for r, p in proofs.items()})), flush=True)
    return dict(identity=identity, cached=cached if all(v is not None for v in cached.values()) else None)


@app.function(image=image, gpu="A100-40GB", cpu=4, memory=24576, timeout=7200,
              volumes={"/artifacts": volume, "/hf-cache": hf_cache},
              max_containers=1, retries=0, scaledown_window=2)
def infer(sources: list, identity: dict, output_id: str):
    import gc
    import torch
    from private_infer import (ROLES, RUNS, cached_rows, checkpoint_info, digest, infer_rows,
                               job_identity, read_json, safe_id, source_only, write_json)
    volume.reload()
    sources = source_only(sources)
    # Check the complete job, including source-only input and code hashes, before loading weights.
    if job_identity(sources, identity["input_file_sha256"], identity["checkpoint_proofs"]) != identity:
        raise ValueError("Remote inference job/input/code identity mismatch")
    directory = ARTIFACT_ROOT / "private-inference" / safe_id(output_id)
    if read_json(directory / "identity.json") != identity:
        raise ValueError("CPU preflight identity missing or changed")
    records = {}
    for role in ROLES:
        proof = checkpoint_info(ARTIFACT_ROOT, role, RUNS[role])
        if proof != identity["checkpoint_proofs"][role]:
            raise ValueError("Checkpoint changed after CPU preflight")
        rows = cached_rows(directory, role, identity, sources)
        if rows is None:
            rows = infer_rows(role, sources, proof["checkpoint"])
            write_json(directory / f"{role}.json", dict(identity=digest(identity), rows=rows, rows_sha256=digest(rows)))
            volume.commit()
            gc.collect()
            torch.cuda.empty_cache()
        records[role] = rows
    hf_cache.commit()
    return records


@app.local_entrypoint()
def main(input_path: str = "private_test.json", output_id: str = "ensemble50-full-private-v1"):
    from private_infer import (export_private, job_identity, read_json, safe_id, sha_file, source_only)
    path = Path(input_path).resolve()
    sources = source_only(read_json(path))
    input_sha256 = sha_file(path)
    safe_id(output_id)
    prepared = preflight.remote(sources, input_sha256, output_id)
    identity = prepared["identity"]
    if identity != job_identity(sources, input_sha256, identity["checkpoint_proofs"]):
        raise ValueError("Remote/local checkpoint job/input/code identity mismatch")
    records = prepared["cached"]
    if records is None:
        records = infer.remote(sources, identity, output_id)
    if sha_file(path) != input_sha256:
        raise ValueError("Private input changed during inference")
    result = export_private(HERE / "outputs" / output_id, sources, records, identity)
    print(json.dumps(dict(status="completed", **result), ensure_ascii=False, indent=2))
