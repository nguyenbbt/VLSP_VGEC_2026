from statistics import mean
from ..schemas import LABELS


def classification(gold, predicted):
    confusion = [[0]*5 for _ in LABELS]
    for y, p in zip(gold, predicted, strict=True):
        if type(p) is not int or p not in LABELS:
            raise ValueError("Predicted label must be integer 0..4")
        confusion[y][p] += 1
    per_class = []
    for label in LABELS:
        tp = confusion[label][label]
        support = sum(confusion[label])
        count = sum(row[label] for row in confusion)
        precision = tp/count if count else 0.
        recall = tp/support if support else 0.
        f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.
        per_class.append(dict(label=label, precision=precision, recall=recall, f1=f1, support=support))
    accuracy = sum(confusion[i][i] for i in LABELS)/len(gold) if gold else 0.
    return dict(macro_f1=mean(r["f1"] for r in per_class), micro_f1=accuracy,
                accuracy=accuracy, per_class=per_class, confusion_matrix=confusion)


def evaluate(predictions, references):
    ref = {r["uid"]: r for r in references}
    pred = {r["uid"]: r for r in predictions}
    if len(pred) != len(predictions) or len(ref) != len(references) or set(ref) != set(pred):
        raise ValueError("Predictions must cover each reference UID exactly once")
    rows = [(ref[uid], pred[uid]) for uid in ref]
    for gold, p in rows:
        if p.get("source_raw") != gold["source_raw"] or p.get("id") != gold["id"]:
            raise ValueError("Prediction source or ID differs from evaluation data")
    result = dict(status="provisional_no_official_scorer", scale="0_to_1", n=len(rows),
                  classification=classification([g["label"] for g, _ in rows], [p["predicted_label"] for _, p in rows]))
    def matching(g, p):
        refs=g["references_raw"] or ([g["source_raw"]] if g["label"]==0 else [])
        return p["selected_text"] in refs
    result["empty_no_error_reference_policy"]="identity_input_as_provisional_reference"
    def stats(subset):
        if not subset:
            return dict(n=0, em_raw=None, changed_rate=None)
        return dict(n=len(subset), em_raw=mean(matching(g,p) for g,p in subset),
                    changed_rate=mean(p["selected_text"] != g["source_raw"] for g,p in subset))
    result["correction"] = stats(rows)
    result["error_only"] = stats([(g,p) for g,p in rows if g["label"] != 0])
    result["no_error"] = stats([(g,p) for g,p in rows if g["label"] == 0])
    result["joint_accuracy"] = mean(p["predicted_label"] == g["label"] and matching(g,p) for g,p in rows) if rows else 0.
    result["GLEU+"] = None
    result["ROUGE-L"] = None
    return result


def analyze_pool(predictions, references):
    evaluate(predictions, references)  # Coverage and source integrity.
    ref = {r["uid"]: r for r in references}
    hits, unique = 0, {}
    for row in predictions:
        gold=ref[row["uid"]]
        refs=gold["references_raw"] or ([gold["source_raw"]] if gold["label"]==0 else [])
        producers = {producer for c in row["candidates"] if c["valid"] and c["text_raw"] in refs for producer in c.get("agreeing_producers",[c["producer_id"]])}
        hits += bool(producers)
        if len(producers) == 1:
            producer = next(iter(producers))
            unique[producer] = unique.get(producer, 0) + 1
    return dict(n=len(predictions), oracle_em_raw=hits/len(predictions) if predictions else 0., exclusive_correct_by_producer=unique,
                status="diagnostic_upper_bound_not_deployable", actual=evaluate(predictions,references))
