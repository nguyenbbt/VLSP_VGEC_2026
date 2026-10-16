import random
import torch
from ..schemas import model_view


def choose_reference(row, seed, epoch):
    refs = row["correction_references"]
    if not refs:
        return row["source_raw"]  # Only shape filler, correction_mask is zero.
    return random.Random(f"{seed}:{epoch}:{row['uid']}").choice(refs)


def collate(rows, tokenizer, cfg, epoch, device):
    sources = [cfg["prefix"] + model_view(r["source_raw"]) for r in rows]
    if cfg.get("task") == "classification":
        inputs = tokenizer(sources, padding=True, truncation=False, return_tensors="pt", return_token_type_ids=False)
        if inputs["input_ids"].shape[1] > cfg["source_max_length"]:
            raise ValueError("Source exceeds classifier token limit; truncation is disabled")
        return dict(input_ids=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device),
                    class_labels=torch.tensor([r["label"] for r in rows], device=device))
    targets = [model_view(choose_reference(r,cfg["seed"],epoch)) for r in rows]
    inputs = tokenizer(sources, padding=True, truncation=False, return_tensors="pt", return_token_type_ids=False)
    target = tokenizer(text_target=targets, padding=True, truncation=False, return_tensors="pt")
    if inputs["input_ids"].shape[1] > cfg["source_max_length"] or target["input_ids"].shape[1] > cfg["target_max_length"]:
        raise ValueError("Token length exceeds audited limit; truncation is disabled")
    labels = target["input_ids"].masked_fill(target["attention_mask"] == 0, -100)
    return dict(input_ids=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device),
                labels=labels.to(device), class_labels=torch.tensor([r["label"] for r in rows],device=device),
                correction_mask=torch.tensor([bool(r["correction_references"]) for r in rows], device=device,dtype=torch.float32))


def audit_lengths(rows, tokenizer, cfg):
    source_lengths = [len(tokenizer(cfg["prefix"]+model_view(r["source_raw"]))["input_ids"]) for r in rows]
    target_lengths = ([] if cfg.get("task") == "classification" else
                      [len(tokenizer(text_target=model_view(ref))["input_ids"]) for r in rows for ref in r["correction_references"]])
    def report(lengths):
        ordered = sorted(lengths)
        return dict(max=max(ordered,default=0),p99=ordered[min(len(ordered)-1,int(.99*len(ordered)))] if ordered else 0)
    summary = dict(source=report(source_lengths),target=report(target_lengths))
    if summary["source"]["max"] > cfg["source_max_length"] or (target_lengths and summary["target"]["max"] > cfg["target_max_length"]):
        raise ValueError(f"Increase configured token limits before training. Actual lengths: {summary}")
    return summary
