import unicodedata
from .io import object_hash

LABELS = list(range(5))


def model_view(text):
    return unicodedata.normalize("NFC", text)


def source_key(text):
    # Only for grouping/deduplication, never for official EM or export.
    return " ".join(model_view(text).split())


def incomplete_reference(text):
    return not text.strip() or "..." in text or "…" in text


def adapt(row, namespace, supervised=True):
    if "ID" not in row or not isinstance(row.get("Input"), str) or not row["Input"].strip():
        raise ValueError("Record requires ID and nonempty Input")
    if isinstance(row["ID"], bool) or not isinstance(row["ID"], (str, int)):
        raise ValueError("ID must be string or integer")
    label, refs = None, []
    if supervised:
        output = row.get("Output", {})
        label, refs = output.get("Error_Type_Id"), output.get("Fixed Sentence")
        if type(label) is not int or label not in LABELS:
            raise ValueError(f"Invalid label at {row['ID']}")
        if not isinstance(refs, list) or (not refs and label != 0) or any(not isinstance(r, str) or not r.strip() for r in refs):
            raise ValueError(f"Invalid references at {row['ID']}")
    key = source_key(row["Input"])
    return dict(uid=f"{namespace}:{row['ID']}", id=row["ID"], source_raw=row["Input"],
                references_raw=refs, label=label, group_id=object_hash(key)[:20],
                origin="unknown" if namespace == "train" else "gold_dev",
                lineage_status="unknown" if namespace == "train" else "user_supplied_dev",
                parent_gold_ids=[], generator_model_id=None, generator_revision=None,
                correction_references=([row["Input"]] if supervised and label == 0 and not refs else [r for r in refs if not incomplete_reference(r)]),
                flags=(["no_error_empty_references_identity_target"] if supervised and label == 0 and not refs else []) +
                      (["incomplete_reference"] if any(incomplete_reference(r) for r in refs) else []))
