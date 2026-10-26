import random
from collections import defaultdict


def inner_split(rows, seed=42, fraction=.15):
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    by_label = defaultdict(list)
    for group, members in groups.items():
        labels = {r["label"] for r in members}
        if len(labels) != 1:
            raise ValueError("Conflicting group labels")
        by_label[members[0]["label"]].append(group)
    valid_groups = set()
    rng = random.Random(seed)
    for label in sorted(by_label):
        ids = sorted(by_label[label])
        if len(ids) < 2:
            raise ValueError(f"Insufficient train groups for label {label}")
        rng.shuffle(ids)
        valid_groups.update(ids[:max(1, min(len(ids)-1, round(len(ids)*fraction)))])
    train = [r for r in rows if r["group_id"] not in valid_groups]
    valid = [r for r in rows if r["group_id"] in valid_groups]
    return train, valid
