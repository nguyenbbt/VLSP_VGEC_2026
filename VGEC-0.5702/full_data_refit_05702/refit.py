"""Fresh, train-only refits of the three 0.5702 components.

This is a new experiment. It never selects a checkpoint on the 90 now-exposed
dev examples and never resumes the historical models.
"""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "VGEC-0.5702/components/s0/src"))
sys.path.insert(0, str(ROOT / "VGEC-0.5702/components/sub1/src"))

EPOCHS = {"bami": 14, "s0": 10, "bart": 6}
COUNTS = {"bami": 7153, "s0": 7153, "bart": 7097}
DATA = {"bami": "sub1_full.jsonl", "s0": "s0_full.jsonl", "bart": "sub1_bart_full.jsonl"}


def code_identity(role):
    files = [Path(__file__)]
    if role == "s0":
        files += [ROOT / "VGEC-0.5702/components/s0/src/vgec/training/batching.py",
                  ROOT / "VGEC-0.5702/components/s0/src/vgec/models/seq2seq.py"]
    else:
        files += [ROOT / "VGEC-0.5702/components/sub1/src/vgec_sub1/training.py",
                  ROOT / "VGEC-0.5702/components/sub1/src/vgec_sub1/models.py"]
    return {path.relative_to(ROOT).as_posix(): sha256(path.read_bytes()).hexdigest() for path in files}


def inputs(role):
    from vgec.io import file_hash, read_records
    manifest = json.loads((HERE / "prepared_manifest.json").read_text(encoding="utf-8"))
    name = DATA[role]
    path = HERE / "data" / name
    if file_hash(path) != manifest["prepared_sha256"][name]:
        raise ValueError("Prepared data hash mismatch")
    rows = read_records(path)
    if len(rows) != COUNTS[role] or len({r["uid"] for r in rows}) != len(rows):
        raise ValueError("Full-data count or UID coverage mismatch")
    return rows, manifest


def train_sub1(role, out, smoke_steps=0, commit=lambda: None):
    import torch
    from vgec_sub1.config import load, role_config
    from vgec_sub1.models import Runtime
    from vgec_sub1.training import optimizer_for, scheduler_for, windows, train_window
    from vgec_sub1.checkpoints import Store, seed_all
    from vgec_sub1.io import digest, write_json

    rows, data_manifest = inputs(role)
    cfg = load(ROOT / "VGEC-0.5702/components/sub1/configs/sub1.yaml")
    rc = role_config(cfg, role)
    if cfg["models"]["bami"]["rdrop_lambda"] != 0 or role not in {"bami", "bart"}:
        raise ValueError("Unexpected component")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Sub1 refit requires a BF16 GPU")
    horizon_epochs = rc["epochs"]  # Preserve the original LR schedule horizon.
    stop_epoch = EPOCHS[role]
    micro = 32 if role == "bami" else 16  # Original A100 benchmark choices.
    seed_all(rc["seed"])
    identity = digest(dict(experiment="full-data-refit-05702-v1", role=role,
                           data_sha256=data_manifest["prepared_sha256"][DATA[role]],
                           config_hash=cfg["_config_hash"], stop_epoch=stop_epoch,
                           horizon_epochs=horizon_epochs, micro=micro,
                           code_sha256=code_identity(role)))
    out = Path(out)
    store = Store(out / "checkpoints" / f"{role}-full", identity, commit)
    previous = store.resolve("last")
    runtime = Runtime(cfg, role, training=True, checkpoint=previous)
    encoded = [runtime.encode(r) for r in rows]
    steps_per_epoch = math.ceil(len(encoded) / rc["global_batch"])
    optimizer = optimizer_for(runtime)
    scheduler = scheduler_for(optimizer, steps_per_epoch * horizon_epochs, rc["warmup"])
    state = dict(epoch=0, step=0, done=False, losses=[])
    if previous:
        state = store.load_training(previous, optimizer, scheduler)
    if state["done"]:
        best = store.resolve("best")
        if best is None:
            raise RuntimeError("Completed refit is missing its best checkpoint")
        write_json(out / "run_manifest.json", dict(identity=identity, role=role,
                   checkpoint=best.name, stop_epoch=stop_epoch,
                   horizon_epochs=horizon_epochs, sources=len(rows),
                   optimizer_steps=state["step"], code_sha256=code_identity(role),
                   data_sha256=data_manifest["prepared_sha256"][DATA[role]],
                   selection="fixed prior selected epoch; dev is optimizer-exposed",
                   smoke_only=bool(smoke_steps), complete=True))
        commit()
        return state
    for epoch in range(state["epoch"], stop_epoch):
        for indices in windows(len(encoded), rc["global_batch"], rc["seed"], epoch):
            loss = train_window(runtime, optimizer, scheduler, [encoded[i] for i in indices], micro)
            state["step"] += 1
            if state["step"] % 50 == 0:
                print(json.dumps(dict(role=role, epoch=epoch + 1, step=state["step"], loss=loss)), flush=True)
            if smoke_steps and state["step"] >= smoke_steps:
                break
        state["epoch"] = epoch + 1
        state["losses"].append(loss)
        state["done"] = bool(smoke_steps) or state["epoch"] == stop_epoch
        store.save(runtime, optimizer, scheduler, state, is_best=state["done"])
        write_json(out / "run_manifest.json", dict(identity=identity, role=role,
                   checkpoint=store.resolve("best").name if state["done"] else None,
                   stop_epoch=stop_epoch, horizon_epochs=horizon_epochs,
                   sources=len(rows), optimizer_steps=state["step"],
                   code_sha256=code_identity(role),
                   data_sha256=data_manifest["prepared_sha256"][DATA[role]],
                   selection="fixed prior selected epoch; dev is optimizer-exposed",
                   smoke_only=bool(smoke_steps), complete=state["done"]))
        commit()
        if state["done"]:
            break
    return state


def train_s0(out, smoke_steps=0, commit=lambda: None):
    import numpy as np
    import shutil
    import torch
    from vgec.io import load_config, write_json
    from vgec.models.seq2seq import load_pretrained, load_checkpoint, save_model
    from vgec.training.batching import collate, audit_lengths

    rows, data_manifest = inputs("s0")
    cfg = load_config(ROOT / "VGEC-0.5702/components/s0/configs/models/bartpho_attention.yaml")
    if cfg["model_id"] != "vinai/bartpho-syllable" or not cfg["joint"] or cfg["classification_pooling"] != "attention":
        raise ValueError("Wrong S0 classifier architecture")
    if not torch.cuda.is_available():
        raise RuntimeError("S0 refit requires a GPU")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    stage, previous, last = out / "staging", out / "previous", out / "last"
    if last.exists() and not (last / "training_state.pt").is_file():
        if not (previous / "training_state.pt").is_file():
            raise RuntimeError("Incomplete S0 last checkpoint and no resumable previous checkpoint")
        shutil.rmtree(last)
        previous.rename(last)
    if not last.exists():
        if (stage / "training_state.pt").is_file():
            stage.rename(last)
        elif previous.exists():
            previous.rename(last)
    if stage.exists():
        shutil.rmtree(stage)
    identity = dict(experiment="full-data-refit-05702-v1", role="s0",
                    data_sha256=data_manifest["prepared_sha256"][DATA["s0"]],
                    config=cfg, code_sha256=code_identity("s0"), stop_epoch=EPOCHS["s0"],
                    horizon_epochs=cfg["max_epochs"], selection="fixed prior selected epoch")
    identity_path = out / "identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("S0 resume identity mismatch")
    else:
        write_json(identity_path, identity)
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    if (last / "training_state.pt").exists():
        model, tokenizer, saved_cfg = load_checkpoint(last, device)
        if saved_cfg != cfg:
            raise ValueError("S0 saved config mismatch")
    else:
        model, tokenizer = load_pretrained(cfg)
        model.to(device)
    audit_lengths(rows, tokenizer, cfg)
    model.backbone.gradient_checkpointing_enable()
    model.backbone.config.use_cache = False
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        (no_decay if param.ndim < 2 or "norm" in name.lower() or name.endswith("bias") else decay).append(param)
    optimizer = torch.optim.AdamW([dict(params=decay, weight_decay=cfg["weight_decay"]),
                                   dict(params=no_decay, weight_decay=0.)], lr=cfg["learning_rate"],
                                  betas=(.9, .999), eps=1e-8)
    micro, accum = cfg["micro_batch"], cfg["accumulation"]
    steps_per_epoch = math.ceil(math.ceil(len(rows) / micro) / accum)
    total_steps = steps_per_epoch * cfg["max_epochs"]
    warmup = min(total_steps - 1, round(total_steps * cfg["warmup_ratio"]))

    def schedule(step):
        if step < warmup:
            return step / max(1, warmup)
        return max(0., (total_steps - step) / max(1, total_steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    state = dict(epoch=0, step=0, done=False, losses=[])
    if (last / "training_state.pt").exists():
        prior = torch.load(last / "training_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(prior["optimizer"])
        scheduler.load_state_dict(prior["scheduler"])
        scaler.load_state_dict(prior["scaler"])
        torch.set_rng_state(prior["torch_rng"])
        torch.cuda.set_rng_state_all(prior["cuda_rng"])
        state = prior["state"]
    if state["done"]:
        best = out / "best_classification"
        if not all((best / name).is_file() for name in ("model.safetensors", "classifier.pt", "attention_pool.pt", "resolved_config.json")):
            save_model(model, tokenizer, out / "best_classification", cfg, identity)
            commit()
        write_json(out / "run_manifest.json", dict(identity=identity, sources=len(rows),
                   optimizer_steps=state["step"], epoch=state["epoch"],
                   smoke_only=bool(smoke_steps), complete=True))
        commit()
        return state
    for epoch in range(state["epoch"], EPOCHS["s0"]):
        order = list(rows)
        random.Random(seed + epoch).shuffle(order)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, len(order), micro * accum):
            chunk = order[start:start + micro * accum]
            for offset in range(0, len(chunk), micro):
                samples = chunk[offset:offset + micro]
                batch = collate(samples, tokenizer, cfg, epoch, device)
                with torch.autocast("cuda", dtype=dtype):
                    loss, _ = model(**batch, classification_weight=cfg["classification_weight"])
                    weighted = loss * len(samples) / len(chunk)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite S0 loss")
                scaler.scale(weighted).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            if not torch.isfinite(norm):
                raise ValueError("Nonfinite S0 gradients")
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            state["step"] += 1
            if state["step"] % 50 == 0:
                print(json.dumps(dict(role="s0", epoch=epoch + 1, step=state["step"], loss=float(loss.detach()))), flush=True)
            if smoke_steps and state["step"] >= smoke_steps:
                break
        state["epoch"] = epoch + 1
        state["losses"].append(float(loss.detach()))
        state["done"] = bool(smoke_steps) or state["epoch"] == EPOCHS["s0"]
        # Keep resumable state only in this new run; never write to the historical run.
        stage.mkdir(exist_ok=False)
        save_model(model, tokenizer, stage, cfg, identity)
        torch.save(dict(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                        scaler=scaler.state_dict(), torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all(), state=state), stage / "training_state.pt")
        old = previous
        if old.exists():
            shutil.rmtree(old)
        if last.exists():
            last.rename(old)
        stage.rename(last)
        commit()
        if state["done"]:
            save_model(model, tokenizer, out / "best_classification", cfg, identity)
            commit()
        if state["done"]:
            break
    write_json(out / "run_manifest.json", dict(identity=identity, sources=len(rows),
               optimizer_steps=state["step"], epoch=state["epoch"],
               smoke_only=bool(smoke_steps), complete=state["done"]))
    commit()
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=tuple(EPOCHS))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke-steps", type=int, default=0)
    args = parser.parse_args()
    if args.smoke_steps < 0:
        parser.error("smoke steps must be nonnegative")
    result = train_s0(args.out, args.smoke_steps) if args.role == "s0" else train_sub1(args.role, args.out, args.smoke_steps)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
