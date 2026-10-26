import math
import time
import torch
from ..schemas import model_view
from .candidates import deduplicate
from .verifier import candidate, select_s0


def encode(sources, tokenizer, cfg, device):
    inputs = tokenizer([cfg["prefix"]+model_view(s) for s in sources], padding=True, truncation=False,
                       return_tensors="pt",return_token_type_ids=False)
    if inputs["input_ids"].shape[1] > cfg["source_max_length"]:
        raise ValueError("Source too long; refusing silent truncation")
    return {k:v.to(device) for k,v in inputs.items()}


@torch.no_grad()
def generate_candidates(model, tokenizer, cfg, rows, device, producer, nbest=1):
    if getattr(model,"classification_only",False):
        raise ValueError("Encoder classifier cannot generate corrections")
    model.eval()
    inputs = encode([r["source_raw"] for r in rows],tokenizer,cfg,device)
    sequences = model.backbone.generate(**inputs, do_sample=False, use_cache=True, num_beams=cfg["beams"],
                    num_return_sequences=nbest,max_new_tokens=cfg["target_max_length"],length_penalty=1.0)
    texts = tokenizer.batch_decode(sequences,skip_special_tokens=True)
    eos = model.backbone.generation_config.eos_token_id
    eos = set(eos if isinstance(eos,list) else [eos])
    result = []
    for i,row in enumerate(rows):
        pool=[]
        for rank in range(nbest):
            seq = sequences[i*nbest+rank].tolist()
            # Ignore decoder start token, which can itself be EOS (mBART).
            generated = seq[1:]
            stop=next((idx+1 for idx,token in enumerate(generated) if token in eos),None)
            # forced_eos_token_id can add EOS at the budget boundary; it does not prove completeness.
            complete=stop is not None and stop<cfg["target_max_length"]
            pool.append(candidate(row["source_raw"],texts[i*nbest+rank],producer,rank,truncated=not complete))
        result.append(pool)
    return result


@torch.no_grad()
def common_score(model,tokenizer,cfg,source,candidates,device):
    from torch.nn.functional import cross_entropy
    inputs = encode([source]*len(candidates),tokenizer,cfg,device)
    target = tokenizer(text_target=[model_view(c["text_raw"]) for c in candidates],padding=True,
                       return_tensors="pt",truncation=False)
    if target["input_ids"].shape[1] > cfg["target_max_length"]:
        return False
    labels=target["input_ids"].masked_fill(target["attention_mask"]==0,-100).to(device)
    logits=model.backbone(**inputs,labels=labels,use_cache=False).logits.float()
    loss=cross_entropy(logits.transpose(1,2),labels,ignore_index=-100,reduction="none")
    scores=(-loss.sum(1)/(labels!=-100).sum(1).clamp_min(1)).tolist()
    for c,score in zip(candidates,scores,strict=True):
        c["common_logprob"]=score
    return True


def predict_with_models(rows, models, classifier_index=0, selector="s0", batch_size=4, nbest=1):
    if selector not in {"s0","s1"}:
        raise ValueError("Unknown selector")
    if not models or getattr(models[0][0],"classification_only",False):
        raise ValueError("First model must be a correction generator; supply the encoder as classifier")
    classifier,ctok,ccfg,_,device=models[classifier_index]
    if not classifier.joint:
        raise ValueError("A trained source classifier is required, including for public-test input")
    classifier.eval()
    probabilities=[]
    with torch.no_grad():
        for start in range(0,len(rows),batch_size):
            inputs=encode([r["source_raw"] for r in rows[start:start+batch_size]],ctok,ccfg,device)
            probabilities.extend(classifier.predict_proba(inputs).cpu().tolist())
    pools=[[candidate(r["source_raw"],r["source_raw"],"copy")] for r in rows]
    for model,tokenizer,cfg,producer,device in models:
        if getattr(model,"classification_only",False):
            continue
        for start in range(0,len(rows),batch_size):
            generated=generate_candidates(model,tokenizer,cfg,rows[start:start+batch_size],device,producer,nbest)
            for offset,cs in enumerate(generated):
                pools[start+offset].extend(cs)
    predictions=[]
    for row,probs,pool in zip(rows,probabilities,pools,strict=True):
        pool=deduplicate(pool)
        for c in pool:
            c["independent_agreement"]=len(set(c["agreeing_producers"])-{"copy"})
        # S0 prefers the incumbent even if its text is identical to the copy candidate.
        valid=[c for c in pool if c["valid"]]
        incumbent=models[0][3]
        selected=next((c for c in valid if incumbent in c["agreeing_producers"]),next(c for c in valid if c["is_copy"]))
        if selector=="s1":
            scorer,stok,scfg,_,sdevice=models[0]
            if common_score(scorer,stok,scfg,row["source_raw"],valid,sdevice):
                selected=max(valid,key=lambda c:c["common_logprob"])
        predictions.append(dict(uid=row["uid"],id=row["id"],source_raw=row["source_raw"],
            predicted_label=max(range(5),key=lambda k:probs[k]),class_probs=probs,selected_text=selected["text_raw"],
            candidates=pool,selected_producer=selected["producer_id"],route=selector,flags=selected["flags"]))
    return predictions
