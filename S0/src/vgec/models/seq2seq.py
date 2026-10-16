import torch
from torch import nn
from torch.nn import functional as F


class AttentionPooling(nn.Module):
    """Learn a token weight from encoder states, excluding every padding token."""

    def __init__(self, hidden_size):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, attention_mask):
        if hidden.ndim != 3 or attention_mask.shape != hidden.shape[:2]:
            raise ValueError("Pooling expects hidden [batch, tokens, hidden] and mask [batch, tokens]")
        valid = attention_mask.to(device=hidden.device, dtype=torch.bool)
        if not valid.any(dim=1).all():
            raise ValueError("Attention pooling requires at least one unmasked token per source")
        # FP32 scoring, softmax and reduction avoid overflow in mixed precision.
        # Mask hidden states too: arbitrary padded values must not affect gradients.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            states = hidden.float().masked_fill(~valid.unsqueeze(-1), 0.)
            scores = F.linear(states, self.score.weight.float()).squeeze(-1)
            weights = scores.masked_fill(~valid, float("-inf")).softmax(dim=1)
            pooled = (states * weights.unsqueeze(-1)).sum(dim=1)
        return pooled.to(hidden.dtype)


class JointSeq2Seq(nn.Module):
    """A new source encoder head ablation, not a reproduction of historic runs."""
    def __init__(self, backbone, joint=True, classification_pooling="mean"):
        super().__init__()
        if classification_pooling not in {"mean", "attention"}:
            raise ValueError("classification_pooling must be mean or attention")
        if not joint and classification_pooling != "mean":
            raise ValueError("Attention classification pooling requires a joint model")
        self.backbone = backbone
        self.joint = joint
        self.classification_pooling = classification_pooling
        hidden = getattr(backbone.config, "d_model", getattr(backbone.config, "hidden_size", None))
        self.attention_pool = AttentionPooling(hidden) if joint and classification_pooling == "attention" else None
        self.classifier = nn.Sequential(nn.Dropout(.1), nn.Linear(hidden, 5)) if joint else None

    def pooled_logits(self, hidden, attention_mask):
        if not self.joint:
            raise ValueError("Correction-only model has no classification head")
        if self.attention_pool is not None:
            pooled = self.attention_pool(hidden, attention_mask)
        else:
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        return self.classifier(pooled)

    def forward(self, input_ids, attention_mask, labels, class_labels, correction_mask, classification_weight=.2):
        result = self.backbone(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
        # Equal weight per source regardless of reference length.
        losses = F.cross_entropy(result.logits.float().transpose(1,2), labels, ignore_index=-100, reduction="none")
        per_source = losses.sum(1) / (labels != -100).sum(1).clamp_min(1)
        correction_loss = (per_source * correction_mask).mean()
        class_loss = torch.zeros((), device=input_ids.device)
        if self.joint:
            logits = self.pooled_logits(result.encoder_last_hidden_state, attention_mask)
            class_loss = F.cross_entropy(logits.float(), class_labels)
        loss = correction_loss + classification_weight * class_loss
        return loss, dict(correction_loss=float(correction_loss.detach()), classification_loss=float(class_loss.detach()))

    @torch.no_grad()
    def predict_proba(self, inputs):
        if not self.joint:
            raise ValueError("Correction-only model cannot supply source labels")
        hidden = self.backbone.get_encoder()(**inputs).last_hidden_state
        return self.pooled_logits(hidden, inputs["attention_mask"]).float().softmax(-1)


def load_pretrained(cfg):
    if cfg.get("task") == "classification":
        raise ValueError("S0 uses the BARTpho joint classification head")
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    if cfg["kind"] not in {"bartpho", "vit5"}:
        raise ValueError("Only BARTpho-syllable and ViT5 are enabled before complementarity gates")
    backbone = AutoModelForSeq2SeqLM.from_pretrained(cfg["model_id"], revision=cfg["revision"])
    resolved=getattr(backbone.config,"_commit_hash",None) or cfg["revision"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], revision=resolved, use_fast=False)
    return JointSeq2Seq(backbone, cfg["joint"], cfg.get("classification_pooling", "mean")), tokenizer


def save_model(model, tokenizer, path, cfg, manifest):
    from pathlib import Path
    from ..io import write_json
    path = Path(path)
    cfg = dict(cfg)
    if getattr(model, "classification_only", False):
        raise ValueError("Standalone encoder classifiers are not part of S0")
    elif cfg.get("task") == "classification":
        raise ValueError("Classification checkpoint requires an encoder classifier")
    if cfg["joint"] != model.joint or cfg.get("classification_pooling", "mean") != model.classification_pooling:
        raise ValueError("Checkpoint config does not match model classification architecture")
    path.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(path)
    tokenizer.save_pretrained(path)
    if model.joint:
        torch.save(model.classifier.state_dict(), path / "classifier.pt")
        if model.attention_pool is not None:
            torch.save(model.attention_pool.state_dict(), path / "attention_pool.pt")
    write_json(path / "resolved_config.json", cfg)
    write_json(path / "run_manifest.json", manifest)


def load_checkpoint(path, device):
    from pathlib import Path
    import json
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    path = Path(path)
    cfg = json.loads((path / "resolved_config.json").read_text(encoding="utf-8"))
    if cfg.get("task") == "classification":
        raise ValueError("S0 requires a BARTpho joint classification checkpoint")
    pooling = cfg.get("classification_pooling", "mean")  # Legacy checkpoints used masked mean pooling.
    if pooling == "attention" and not (path / "attention_pool.pt").is_file():
        raise FileNotFoundError(f"Attention checkpoint is missing trained pooling weights: {path / 'attention_pool.pt'}")
    if pooling != "attention" and (path / "attention_pool.pt").exists():
        raise ValueError("Pooling weights/config disagree; refusing to load a mixed checkpoint")
    model = JointSeq2Seq(AutoModelForSeq2SeqLM.from_pretrained(path), cfg["joint"], pooling)
    if cfg["joint"]:
        model.classifier.load_state_dict(torch.load(path / "classifier.pt", map_location="cpu", weights_only=True))
        if model.attention_pool is not None:
            model.attention_pool.load_state_dict(torch.load(path / "attention_pool.pt", map_location="cpu", weights_only=True))
    model.to(device).eval()
    return model, AutoTokenizer.from_pretrained(path, use_fast=False), cfg
