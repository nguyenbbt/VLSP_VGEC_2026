from ..io import write_json


def export_submission(predictions, inputs, output, schema_verified=False, draft=False):
    if not schema_verified and not draft:
        raise ValueError("Submission schema not verified; use --draft until a BTC template is supplied")
    if len({type(r["ID"]).__name__+":"+str(r["ID"]) for r in inputs})!=len(inputs):
        raise ValueError("Duplicate public-test IDs")
    by_id={type(p["id"]).__name__+":"+str(p["id"]):p for p in predictions}
    keys=[type(r["ID"]).__name__+":"+str(r["ID"]) for r in inputs]
    if len(by_id)!=len(predictions) or set(keys)!=set(by_id):
        raise ValueError("Prediction/input ID coverage differs")
    result=[]
    for row,key in zip(inputs,keys,strict=True):
        p=by_id[key]
        if p["source_raw"]!=row["Input"]:
            raise ValueError("Prediction does not correspond to original raw Input")
        if type(p["predicted_label"]) is not int or p["predicted_label"] not in range(5) or not p["selected_text"].strip():
            raise ValueError("Invalid submission record")
        result.append(dict(ID=row["ID"],Output={"Error_Type_Id":p["predicted_label"],"Fixed Sentence":[p["selected_text"]]}))
    write_json(output,result)
    return dict(records=len(result),draft=not schema_verified)
