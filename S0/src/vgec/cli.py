import argparse
import json
import sys
from pathlib import Path
from .io import read_records, write_json, write_jsonl


def prediction_command(args):
    import torch
    from .models.seq2seq import load_checkpoint
    from .models.registry import parameter_report
    from .inference.pipeline import predict_with_models
    from .schemas import adapt
    rows=read_records(args.input)
    if rows and "source_raw" not in rows[0]:
        rows=[adapt(r,args.namespace,supervised=False) for r in rows]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths=[args.checkpoint]
    if args.secondary:
        paths.append(args.secondary)
    classifier_index=0
    if args.classifier and args.classifier not in paths:
        classifier_index=len(paths);paths.append(args.classifier)
    elif args.classifier:
        classifier_index=paths.index(args.classifier)
    models=[]
    for i,path in enumerate(paths):
        model,tok,cfg=load_checkpoint(path,device)
        parameter_report(model)
        models.append((model,tok,cfg,f"model_{i}",device))
    pred=predict_with_models(rows,models,classifier_index=classifier_index,selector=args.selector,
                             batch_size=args.batch_size,nbest=args.nbest)
    write_jsonl(args.output,pred)
    return dict(records=len(pred),output=args.output)


def main(argv=None):
    parser=argparse.ArgumentParser(prog="vgec")
    sub=parser.add_subparsers(dest="command",required=True)
    a=sub.add_parser("audit");a.add_argument("--config",default="configs/data.yaml")
    t=sub.add_parser("train")
    t.add_argument("--config",required=True);t.add_argument("--track-config",default="configs/tracks/experimental.yaml")
    t.add_argument("--train-path",default="data/prepared/train.jsonl");t.add_argument("--dev-path",default="data/prepared/dev.jsonl")
    t.add_argument("--output",required=True);t.add_argument("--smoke-steps",type=int,default=0);t.add_argument("--resume",action="store_true")
    p=sub.add_parser("predict")
    p.add_argument("--checkpoint",required=True);p.add_argument("--secondary");p.add_argument("--classifier")
    p.add_argument("--input",required=True);p.add_argument("--output",required=True);p.add_argument("--namespace",default="test")
    p.add_argument("--selector",choices=["s0"],default="s0");p.add_argument("--nbest",type=int,choices=[1,2],default=1)
    p.add_argument("--batch-size",type=int,default=4)
    for name in ("evaluate","analyze-pool"):
        e=sub.add_parser(name);e.add_argument("--predictions",required=True);e.add_argument("--references",default="data/prepared/dev.jsonl")
        e.add_argument("--output",required=True)
    e=sub.add_parser("export");e.add_argument("--predictions",required=True);e.add_argument("--input",required=True);e.add_argument("--output",required=True)
    e.add_argument("--draft",action="store_true");e.add_argument("--schema-verified",action="store_true")
    args=parser.parse_args(argv)
    if args.command=="audit":
        from .data.audit import audit
        result=audit(args.config)
        result={k:result[k] for k in ("original_train","dev_records","prepared_train","exclusions_by_reason","dev_incomplete_reference_records","train_without_complete_correction_target","constrained_eligible")}
    elif args.command=="train":
        from .training.full_ft import train
        result=train(args.config,args.track_config,args.train_path,args.dev_path,args.output,args.smoke_steps,args.resume)
    elif args.command=="predict":
        result=prediction_command(args)
    elif args.command in {"evaluate","analyze-pool"}:
        from .evaluation.diagnostics import evaluate,analyze_pool
        fn=evaluate if args.command=="evaluate" else analyze_pool
        result=fn(read_records(args.predictions),read_records(args.references));write_json(args.output,result)
    elif args.command=="export":
        from .export.submission import export_submission
        result=export_submission(read_records(args.predictions),read_records(args.input),args.output,args.schema_verified,args.draft)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
