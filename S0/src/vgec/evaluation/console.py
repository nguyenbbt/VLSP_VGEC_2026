"""Readable local terminal reports; reference labels are never model inputs."""


def format_metrics(metrics):
    def percent(value):
        return "N/A" if value is None else f"{100 * value:.2f}%"

    cls = metrics["classification"]
    lines = [
        f"Evaluation: {metrics['n']} sentences (provisional metrics)",
        f"Macro-F1: {percent(cls['macro_f1'])}",
        f"Micro-F1 / Accuracy: {percent(cls['micro_f1'])}",
        f"EM: {percent(metrics['correction']['em_raw'])}",
        f"Error-only EM: {percent(metrics['error_only']['em_raw'])} (n={metrics['error_only']['n']})",
        f"No-error EM: {percent(metrics['no_error']['em_raw'])} (n={metrics['no_error']['n']})",
        f"No-error changed rate: {percent(metrics['no_error']['changed_rate'])}",
        f"Joint label + correction: {percent(metrics['joint_accuracy'])}",
        "", "Label    Precision    Recall        F1    Support",
    ]
    for row in cls["per_class"]:
        lines.append(f"{row['label']:>5}    {percent(row['precision']):>9}    {percent(row['recall']):>6}    {percent(row['f1']):>6}    {row['support']:>7}")
    lines += ["", "Confusion matrix (rows=gold, columns=predicted; labels 0..4):"]
    lines += [" ".join(f"{value:>5}" for value in row) for row in cls["confusion_matrix"]]
    lines += ["", "GLEU+ / ROUGE-L: N/A (official BTC scorer not configured)"]
    return "\n".join(lines)


def format_predictions(predictions, references=None):
    gold = {row["uid"]: row for row in references or []}
    lines = []
    for row in predictions:
        reference = gold.get(row["uid"])
        label = f"predicted={row['predicted_label']}"
        if reference is not None:
            label += f", gold={reference['label']}"
        lines += [f"ID {row['id']} | {label}", f"Input: {row['source_raw']}", f"Prediction: {row['selected_text']}"]
        if reference is not None:
            refs = reference["references_raw"] or ([reference["source_raw"]] if reference["label"] == 0 else [])
            lines += [f"Reference {i}: {text}" for i, text in enumerate(refs, 1)]
        lines.append("")
    return "\n".join(lines)
