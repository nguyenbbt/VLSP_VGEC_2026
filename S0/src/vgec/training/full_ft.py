import contextlib
import json
import math
import platform
import random
import time
from pathlib import Path
import torch
from ..io import load_config, read_records, write_json, write_jsonl, file_hash, object_hash
from ..data.split import inner_split
from ..evaluation.diagnostics import evaluate
from ..models.seq2seq import load_pretrained, save_model, load_checkpoint
from ..models.registry import parameter_report
from ..inference.pipeline import generate_candidates, predict_with_models
from .batching import collate, audit_lengths


def enforce_policy(rows, track):
    if track.get("require_verified_lineage"):
        if any(r.get("lineage_status") != "verified" for r in rows):
            raise ValueError("Constrained run blocked: training lineage is not verified. Use experimental track for the user-authorized unverified dataset.")
        for row in rows:
            if row.get("origin") not in {"gold","synthetic"}:
                raise ValueError("Unknown data origin")
            if row.get("origin")=="synthetic":
                closure=row.get("lineage_models",[])
                if not closure or row.get("dev_exposed") is not False:
                    raise ValueError("Synthetic ancestry/development exposure unverified")
                for item in closure:
                    count=item.get("total_parameters")
                    if type(count) is not int or count>track["max_parameters_per_model"] or not item.get("revision"):
                        raise ValueError("Synthetic producer/judge has unverified or excessive parameters")


def validation(model,tokenizer,cfg,rows,device):
    if cfg.get("task") == "classification":
        from ..inference.pipeline import encode
        from ..evaluation.diagnostics import classification
        pred=[]
        model.eval()
        for start in range(0,len(rows),cfg["micro_batch"]):
            chunk=rows[start:start+cfg["micro_batch"]]
            inputs=encode([r["source_raw"] for r in chunk],tokenizer,cfg,device)
            probabilities=model.predict_proba(inputs).cpu().tolist()
            for r,probs in zip(chunk,probabilities,strict=True):
                pred.append(dict(uid=r["uid"],id=r["id"],source_raw=r["source_raw"],
                                 predicted_label=max(range(5),key=lambda k:probs[k]),class_probs=probs))
        metrics=dict(n=len(rows),task="classification",status="classification_only",
                     classification=classification([r["label"] for r in rows],[p["predicted_label"] for p in pred]),
                     correction=None,joint_accuracy=None)
        return pred,metrics
    if cfg["joint"]:
        pred=predict_with_models(rows,[(model,tokenizer,cfg,"incumbent",device)],batch_size=cfg["micro_batch"])
    else:
        pred=[]
        for start in range(0,len(rows),cfg["micro_batch"]):
            chunk=rows[start:start+cfg["micro_batch"]]
            pools=generate_candidates(model,tokenizer,cfg,chunk,device,"corrector")
            for r,cs in zip(chunk,pools,strict=True):
                c=cs[0]
                pred.append(dict(uid=r["uid"],id=r["id"],source_raw=r["source_raw"],predicted_label=0,
                                 selected_text=c["text_raw"] if c["valid"] else r["source_raw"],candidates=cs))
    metrics=evaluate(pred,rows)
    if not cfg["joint"]:
        metrics["classification"]=None
        metrics["joint_accuracy"]=None
    return pred,metrics


def train(config,track_config,train_path,dev_path,output,smoke_steps=0,resume=False,raw_train_hash=None,commit=None):
    import numpy as np
    import transformers
    cfg,track=load_config(config),load_config(track_config)
    classification_only=cfg.get("task")=="classification"
    rows,dev=read_records(train_path),read_records(dev_path)
    enforce_policy(rows,track)
    if {r["group_id"] for r in rows}&{r["group_id"] for r in dev}:
        raise ValueError("Train/dev overlap")
    if not cfg["joint"]:
        rows=[r for r in rows if r["correction_references"]]
    training,inner=inner_split(rows,cfg["seed"])
    out=Path(output)
    existing=out.exists() and any(out.iterdir())
    if existing and not resume:
        raise ValueError("Run directory is nonempty: use a new run ID or --resume")
    out.mkdir(parents=True,exist_ok=True)
    seed=cfg["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type=="cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    cfg=dict(cfg)
    identity=dict(config=cfg,track=track,train_hash=file_hash(train_path),dev_hash=file_hash(dev_path),smoke_steps=smoke_steps)
    if raw_train_hash is not None:
        identity["raw_train_hash"]=raw_train_hash
    identity_hash=object_hash(identity)
    transactional=cfg.get('transactional_checkpoints',False)
    last_path=out/'last'
    if transactional and (out/'last.json').exists():
        from .checkpoints import checkpoint_path
        last_path=checkpoint_path(out,'last',identity_hash)
    checkpoint_resume=False
    if resume and existing:
        previous=json.loads((out/"run_manifest.json").read_text(encoding="utf-8"))
        if previous["identity_hash"]!=identity_hash:
            raise ValueError("Resume config/data identity mismatch")
        if previous.get("status")=="finished":
            if not (last_path/"training_state.pt").is_file():
                raise ValueError("Finished run is missing its last checkpoint")
            return previous
        checkpoint_resume=(last_path/"training_state.pt").is_file()
    if checkpoint_resume:
        model,tokenizer,_=load_checkpoint(last_path,device)
    else:
        model,tokenizer=load_pretrained(cfg)
        model.to(device)
    eligibility=parameter_report(model,track["max_parameters_per_model"])
    if eligibility["total_parameters"]!=eligibility["trainable_parameters"]:
        raise ValueError("Full FT requires all parameters trainable")
    lengths=audit_lengths(training+inner,tokenizer,cfg)
    model.backbone.gradient_checkpointing_enable()
    model.backbone.config.use_cache=False
    dtype=torch.bfloat16 if device.type=="cuda" and torch.cuda.is_bf16_supported() else torch.float16
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda" and dtype==torch.float16)
    def amp():
        return torch.autocast("cuda",dtype=dtype) if device.type=="cuda" else contextlib.nullcontext()
    decay,no_decay=[],[]
    for name,param in model.named_parameters():
        (no_decay if param.ndim<2 or "norm" in name.lower() or name.endswith("bias") else decay).append(param)
    optimizer=torch.optim.AdamW([dict(params=decay,weight_decay=cfg["weight_decay"]),dict(params=no_decay,weight_decay=0.)],
                               lr=cfg["learning_rate"],betas=(.9,.999),eps=1e-8)
    micro,accum=cfg["micro_batch"],cfg["accumulation"]
    batches=math.ceil(len(training)/micro)
    total_steps=smoke_steps or math.ceil(batches/accum)*cfg["max_epochs"]
    warmup=min(total_steps-1,round(total_steps*cfg["warmup_ratio"]))
    def schedule(step):
        if step<warmup:
            return step/max(1,warmup)
        return max(0.,(total_steps-step)/max(1,total_steps-warmup))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
    manifest=dict(identity_hash=identity_hash,**identity,parameters=eligibility,token_lengths=lengths,
                  split_hashes=dict(optimizer=object_hash([r["uid"] for r in training]),
                                    inner=object_hash([r["uid"] for r in inner]),
                                    dev=object_hash([r["uid"] for r in dev])),
                  model_revision_resolved=getattr(model.backbone.config,"_commit_hash",None),
                  tokenizer_revision_resolved=getattr(tokenizer,"init_kwargs",{}).get("_commit_hash") or getattr(model.backbone.config,"_commit_hash",None),
                  experiment="bamibert_attention_five_class_full_ft" if classification_only else
                             "new_full_ft_encoder_head_ablation" if cfg["joint"] else "new_correction_only_full_ft",
                  not_historical_reproduction=True,training_records=len(training),inner_records=len(inner),
                  effective_batch=micro*accum,world_size=1,dtype=str(dtype) if device.type=="cuda" else "float32",
                  gpu=torch.cuda.get_device_name() if device.type=="cuda" else "CPU",
                  dependencies=dict(torch=torch.__version__,transformers=transformers.__version__,numpy=np.__version__,python=platform.python_version()),
                  scorer_status="provisional_no_official_scorer",constrained_eligible=track["track"]=="constrained",
                  ancestor_leakage_verified=False)
    from ..inference.staged import atomic_json
    (atomic_json if transactional else write_json)(out/"run_manifest.json",manifest)
    write_json(out/"resolved_config.json",cfg)
    write_json(out/"model_eligibility.json",eligibility)
    write_json(out/"split_manifest.json",dict(train=[r["uid"] for r in training],inner=[r["uid"] for r in inner],dev=[r["uid"] for r in dev]))
    epoch_start,step,best_cls,best_corr,stale=0,0,-1.,-1.,0
    last_loss=None
    if checkpoint_resume:
        # Load on CPU: optimizer.load_state_dict moves its own tensors to the
        # parameter device. Avoid placing the whole serialized state on CUDA.
        state=torch.load(last_path/"training_state.pt",map_location="cpu",weights_only=False)
        optimizer.load_state_dict(state["optimizer"]);scheduler.load_state_dict(state["scheduler"]);scaler.load_state_dict(state["scaler"])
        epoch_start,step,best_cls,best_corr,stale=state["next_epoch"],state["step"],state["best_cls"],state["best_corr"],state["stale"]
        last_loss=state.get("last_loss",previous.get("last_loss"))
        random.setstate(state["python_rng"]);np.random.set_state(state["numpy_rng"]);torch.set_rng_state(state["torch_rng"].cpu())
        if device.type=="cuda" and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])
        del state
        # A worker may stop after saving the final checkpoint but before the
        # finished manifest. Respect all stopping criteria on resume as well.
        if (smoke_steps and step>=smoke_steps) or stale>=cfg["patience"]:
            epoch_start=cfg["max_epochs"]
    started=time.monotonic()
    resumed_step=step
    def snapshot(name,state=None):
        if not transactional:
            save_model(model,tokenizer,out/name,cfg,manifest)
            if state is not None: torch.save(state,out/name/'training_state.pt')
            return
        from ..inference.staged import publish_directory,atomic_json
        def writer(path):
            save_model(model,tokenizer,path,cfg,manifest)
            if state is not None: torch.save(state,path/'training_state.pt')
        path=publish_directory(out/'checkpoints',dict(run_identity=identity_hash,name=name,step=step),writer)
        atomic_json(out/(name+'.json'),dict(key=path.name))
    for epoch in range(epoch_start,cfg["max_epochs"]):
        order=list(training);random.Random(seed+epoch).shuffle(order)
        model.train();optimizer.zero_grad(set_to_none=True)
        epoch_loss=[]
        for window in range(0,len(order),micro*accum):
            chunk=order[window:window+micro*accum]
            for start in range(0,len(chunk),micro):
                samples=chunk[start:start+micro]
                batch=collate(samples,tokenizer,cfg,epoch,device)
                with amp():
                    loss,parts=model(**batch,classification_weight=cfg["classification_weight"])
                    # Last accumulation window keeps equal weighting by source.
                    weighted=loss*len(samples)/len(chunk)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                scaler.scale(weighted).backward()
                epoch_loss.append(float(loss.detach()))
                last_loss=epoch_loss[-1]
            scaler.unscale_(optimizer)
            grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["max_grad_norm"])
            if not torch.isfinite(grad_norm):
                raise ValueError("Nonfinite full FT gradients")
            scaler.step(optimizer);scaler.update();scheduler.step();optimizer.zero_grad(set_to_none=True)
            step+=1
            print(json.dumps(dict(epoch=epoch+1,step=step,loss=epoch_loss[-1],lr=scheduler.get_last_lr()[0]),ensure_ascii=False),flush=True)
            if smoke_steps and step>=smoke_steps:
                break
        model.eval()
        with amp():
            preds,metrics=validation(model,tokenizer,cfg,inner[:min(25,len(inner))] if smoke_steps else inner,device)
        cls=metrics["classification"]["macro_f1"] if cfg["joint"] else -1.
        corr=-1. if classification_only else metrics["correction"]["em_raw"]
        improved=False
        if cfg["joint"] and cls>best_cls:
            best_cls=cls;improved=True
            snapshot('best_classification')
        if corr>best_corr:
            best_corr=corr;improved=True
            snapshot('best_correction')
        stale=0 if improved else stale+1
        write_json(out/f"inner_metrics_epoch_{epoch+1}.json",metrics)
        write_jsonl(out/f"inner_predictions_epoch_{epoch+1}.jsonl",preds)
        snapshot('last',dict(optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),next_epoch=epoch+1,
            step=step,last_loss=last_loss,best_cls=best_cls,best_corr=best_corr,stale=stale,python_rng=random.getstate(),numpy_rng=np.random.get_state(),
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=="cuda" else None))
        if commit: commit()
        if smoke_steps or stale>=cfg["patience"]:
            break
    wall_seconds=time.monotonic()-started
    manifest.update(optimizer_steps=step,wall_seconds=wall_seconds,last_loss=last_loss,
                    optimizer_steps_per_second=(step-resumed_step)/wall_seconds if wall_seconds else None,
                    peak_vram_bytes=torch.cuda.max_memory_allocated() if device.type=="cuda" else None,
                    smoke_only=bool(smoke_steps),status="finished")
    (atomic_json if transactional else write_json)(out/"run_manifest.json",manifest)
    return manifest
