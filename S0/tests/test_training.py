import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from transformers import T5Config,T5ForConditionalGeneration,PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from vgec.models.seq2seq import AttentionPooling,JointSeq2Seq,save_model,load_checkpoint
from vgec.models.registry import parameter_report
from vgec.training.batching import collate
from vgec.training.full_ft import train,enforce_policy,validation
from vgec.schemas import adapt
from vgec.io import write_jsonl


def tiny_model(pooling="mean"):
    torch.manual_seed(42)
    tok=Tokenizer(WordLevel({"<pad>":0,"</s>":1,"<unk>":2,"source":3,"target":4},unk_token="<unk>"))
    tok.pre_tokenizer=Whitespace()
    tok.post_processor=TemplateProcessing(single="$A </s>",special_tokens=[("</s>",1)])
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=tok,pad_token="<pad>",eos_token="</s>",unk_token="<unk>")
    backbone=T5ForConditionalGeneration(T5Config(vocab_size=5,d_model=16,d_ff=32,num_layers=1,num_decoder_layers=1,num_heads=2,
                                                decoder_start_token_id=0,pad_token_id=0,eos_token_id=1,dropout_rate=0.))
    return JointSeq2Seq(backbone,True,pooling),tokenizer


class Training(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def cfg(self):
        return dict(kind="vit5",model_id="tiny-local-fixture",revision="fixture",joint=True,classification_weight=.2,prefix="",seed=42,
                    learning_rate=.001,micro_batch=2,accumulation=2,max_epochs=2,patience=3,warmup_ratio=.06,weight_decay=.01,
                    max_grad_norm=1.,source_max_length=16,target_max_length=8,beams=2)

    def make_model(self):
        return tiny_model(self.cfg().get("classification_pooling", "mean"))

    def rows(self,namespace="train",count=30):
        return [adapt(dict(ID=i,Input=f"source {namespace} {i}",Output={"Error_Type_Id":i%5,"Fixed Sentence":["target"]}),namespace) for i in range(count)]

    def test_full_ft_gradient_and_pad_mask(self):
        model,tok=self.make_model()
        batch=collate(self.rows()[:2],tok,self.cfg(),0,torch.device("cpu"))
        loss,_=model(**batch)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.backbone.shared.weight.grad)
        self.assertGreater(float(model.backbone.shared.weight.grad.abs().sum()),0.)
        self.assertIsNotNone(model.classifier[1].weight.grad)
        self.assertGreater(float(model.classifier[1].weight.grad.abs().sum()),0.)
        if model.attention_pool is not None:
            self.assertGreater(float(model.attention_pool.score.weight.grad.abs().sum()),0.)
        self.assertEqual(parameter_report(model)["trainable_parameters"],parameter_report(model)["total_parameters"])

    def test_target_padding_and_no_truncation(self):
        model,tok=self.make_model()
        rows=self.rows()[:2];rows[1]["correction_references"]=["target target target"]
        batch=collate(rows,tok,self.cfg(),0,torch.device("cpu"))
        self.assertTrue((batch["labels"]==-100).any())
        cfg=self.cfg();cfg["target_max_length"]=2
        with self.assertRaises(ValueError):
            collate(rows,tok,cfg,0,torch.device("cpu"))

    def test_classification_only_sample_has_zero_correction_loss(self):
        model,tok=self.make_model()
        row=self.rows()[0];row["correction_references"]=[]
        batch=collate([row],tok,self.cfg(),0,torch.device("cpu"))
        loss,parts=model(**batch)
        self.assertEqual(parts["correction_loss"],0.)
        self.assertGreater(parts["classification_loss"],0.)

    def test_parameter_limit_and_unknown_lineage_negative_cases(self):
        model,_=self.make_model()
        with self.assertRaises(ValueError):
            parameter_report(model,max_parameters=1)
        with self.assertRaises(ValueError):
            enforce_policy(self.rows(),dict(require_verified_lineage=True,max_parameters_per_model=8_000_000_000))

    def test_checkpoint_roundtrip_and_public_input_inference(self):
        from vgec.inference.pipeline import predict_with_models
        model,tok=self.make_model();model.eval()
        with tempfile.TemporaryDirectory() as temp:
            save_model(model,tok,temp,self.cfg(),{})
            loaded,ltok,cfg=load_checkpoint(temp,torch.device("cpu"))
            self.assertTrue(torch.equal(model.classifier[1].weight,loaded.classifier[1].weight))
            inputs=tok(["source", "source target"],padding=True,return_tensors="pt",return_token_type_ids=False)
            torch.testing.assert_close(model.predict_proba(inputs),loaded.predict_proba(inputs),rtol=0,atol=0)
            if model.attention_pool is not None:
                self.assertTrue(torch.equal(model.attention_pool.score.weight,loaded.attention_pool.score.weight))
            row=adapt({"ID":7,"Input":"source"},"test",False)
            pred=predict_with_models([row],[(loaded,ltok,cfg,"tiny",torch.device("cpu"))],batch_size=1)
            self.assertEqual(pred[0]["id"],7)
            self.assertIn(pred[0]["predicted_label"],range(5))
            self.assertTrue(pred[0]["selected_text"])

    def test_training_checkpoint_resume(self):
        import yaml
        cfg=self.cfg()
        track=dict(track="experimental_unverified",require_verified_lineage=False,max_parameters_per_model=8_000_000_000)
        calls=[0]
        def interrupted(*args,**kwargs):
            calls[0]+=1
            if calls[0]==2:
                raise RuntimeError("simulated interruption after epoch-one checkpoint")
            return validation(*args,**kwargs)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);config=root/"config.yaml";policy=root/"track.yaml"
            config.write_text(yaml.safe_dump(cfg),encoding="utf-8");policy.write_text(yaml.safe_dump(track),encoding="utf-8")
            t=root/"train.jsonl";d=root/"dev.jsonl";out=root/"run"
            write_jsonl(t,self.rows());write_jsonl(d,self.rows("dev",5))
            with patch("vgec.training.full_ft.load_pretrained",side_effect=lambda cfg:self.make_model()),patch("vgec.training.full_ft.validation",side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError,"simulated interruption"):
                    train(config,policy,t,d,out)
            self.assertTrue((out/"last"/"training_state.pt").exists())
            manifest=train(config,policy,t,d,out,resume=True)
            self.assertEqual(manifest["status"],"finished")
            self.assertEqual(manifest["optimizer_steps"],14)
            self.assertTrue((out/"best_classification"/"classifier.pt").exists())
            # Resume must recover weights AND optimizer/RNG state: compare with
            # the same complete run that was never interrupted.
            uninterrupted=root/"uninterrupted"
            with patch("vgec.training.full_ft.load_pretrained",side_effect=lambda cfg:self.make_model()):
                train(config,policy,t,d,uninterrupted)
            resumed_model,_,_=load_checkpoint(out/"last",torch.device("cpu"))
            reference_model,_,_=load_checkpoint(uninterrupted/"last",torch.device("cpu"))
            for name,value in reference_model.state_dict().items():
                torch.testing.assert_close(value,resumed_model.state_dict()[name],rtol=0,atol=0)
            changed=cfg|{"learning_rate":.002};config.write_text(yaml.safe_dump(changed),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"identity mismatch"):
                train(config,policy,t,d,out,resume=True)

    def test_resume_before_first_checkpoint_restarts_with_matching_identity(self):
        import yaml
        cfg=self.cfg();cfg["max_epochs"]=1
        track=dict(track="experimental_unverified",require_verified_lineage=False,max_parameters_per_model=8_000_000_000)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);config=root/"config.yaml";policy=root/"track.yaml"
            config.write_text(yaml.safe_dump(cfg),encoding="utf-8")
            policy.write_text(yaml.safe_dump(track),encoding="utf-8")
            t=root/"train.jsonl";d=root/"dev.jsonl";out=root/"run"
            write_jsonl(t,self.rows(count=16));write_jsonl(d,self.rows("dev",5))
            with patch("vgec.training.full_ft.load_pretrained",side_effect=lambda cfg:self.make_model()), \
                 patch("vgec.training.full_ft.validation",side_effect=RuntimeError("preempted before save")):
                with self.assertRaisesRegex(RuntimeError,"preempted before save"):
                    train(config,policy,t,d,out,smoke_steps=2,resume=True)
            self.assertTrue((out/"run_manifest.json").exists())
            self.assertFalse((out/"last"/"training_state.pt").exists())
            with patch("vgec.training.full_ft.load_pretrained",side_effect=lambda cfg:self.make_model()):
                result=train(config,policy,t,d,out,smoke_steps=2,resume=True)
            self.assertEqual(result["optimizer_steps"],2)
            self.assertEqual(result["status"],"finished")
            self.assertTrue((out/"last"/"training_state.pt").exists())
            self.assertEqual(train(config,policy,t,d,out,smoke_steps=2,resume=True)["status"],"finished")


class AttentionTraining(Training):
    """Run the whole train/save/resume/public-inference contract with attention."""

    def cfg(self):
        return super().cfg() | {"classification_pooling": "attention"}

    def test_resume_after_terminal_checkpoint_does_not_train_extra_steps(self):
        import yaml
        from vgec.io import write_json
        track=dict(track="experimental_unverified",require_verified_lineage=False,max_parameters_per_model=8_000_000_000)
        for reason,smoke_steps,max_epochs,patience in (("smoke",2,2,3),("max_epochs",0,1,3),("early_stop",0,4,1)):
            with self.subTest(reason=reason),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);out=root/"run";config=root/"config.yaml";policy=root/"track.yaml"
                cfg=self.cfg()|{"max_epochs":max_epochs,"patience":patience}
                config.write_text(yaml.safe_dump(cfg),encoding="utf-8")
                policy.write_text(yaml.safe_dump(track),encoding="utf-8")
                t=root/"train.jsonl";d=root/"dev.jsonl"
                write_jsonl(t,self.rows(count=16));write_jsonl(d,self.rows("dev",5))
                completed=[]
                def fail_final_manifest(path,value):
                    if Path(path)==out/"run_manifest.json" and value.get("status")=="finished":
                        completed.append(dict(value))
                        raise RuntimeError("preempted after terminal checkpoint")
                    return write_json(path,value)
                def flat_validation(*args,**kwargs):
                    pred,metrics=validation(*args,**kwargs)
                    metrics["classification"]["macro_f1"]=.5
                    metrics["correction"]["em_raw"]=.5
                    return pred,metrics
                with patch("vgec.training.full_ft.load_pretrained",side_effect=lambda cfg:self.make_model()), \
                     patch("vgec.training.full_ft.write_json",side_effect=fail_final_manifest), \
                     patch("vgec.training.full_ft.validation",side_effect=flat_validation):
                    with self.assertRaisesRegex(RuntimeError,"preempted after terminal"):
                        train(config,policy,t,d,out,smoke_steps=smoke_steps)
                with patch("vgec.training.full_ft.validation",side_effect=AssertionError("Unexpected extra epoch")):
                    resumed=train(config,policy,t,d,out,smoke_steps=smoke_steps,resume=True)
                self.assertEqual(resumed["optimizer_steps"],completed[0]["optimizer_steps"])
                self.assertEqual(resumed["last_loss"],completed[0]["last_loss"])
                self.assertEqual(resumed["status"],"finished")

    def test_masked_pooling_values_gradients_and_mixed_precision(self):
        pool=AttentionPooling(3)
        with torch.no_grad():
            pool.score.weight.copy_(torch.tensor([[1., 0., 0.]]))
        hidden=torch.tensor([[[0.,2.,0.],[1.,0.,4.],[float("nan"),float("nan"),float("nan")]]],requires_grad=True)
        mask=torch.tensor([[1,1,0]])
        weight=torch.softmax(torch.tensor([0.,1.]),0)
        expected=weight[0]*hidden.detach()[0,0]+weight[1]*hidden.detach()[0,1]
        result=pool(hidden,mask)
        torch.testing.assert_close(result[0],expected)
        result.sum().backward()
        self.assertTrue(torch.isfinite(pool.score.weight.grad).all())
        self.assertGreater(float(pool.score.weight.grad.abs().sum()),0.)
        self.assertTrue(torch.equal(hidden.grad[0,2],torch.zeros(3)))
        with torch.autocast("cpu",dtype=torch.bfloat16):
            low_precision=pool(hidden.detach().to(torch.bfloat16),mask)
        self.assertTrue(torch.isfinite(low_precision).all())
        torch.testing.assert_close(low_precision.float(),result.detach(),rtol=.01,atol=.01)
        with self.assertRaisesRegex(ValueError,"unmasked token"):
            pool(torch.zeros(1,2,3),torch.zeros(1,2))
        with self.assertRaisesRegex(ValueError,"Pooling expects"):
            pool(torch.zeros(1,2,3),torch.ones(1,3))

    def test_padding_does_not_change_five_class_probabilities(self):
        model,tok=self.make_model();model.eval()
        single=tok(["source"],return_tensors="pt",return_token_type_ids=False)
        batch=tok(["source","source target source"],padding=True,return_tensors="pt",return_token_type_ids=False)
        probabilities=model.predict_proba(batch)
        self.assertEqual(tuple(probabilities.shape),(2,5))
        torch.testing.assert_close(probabilities.sum(-1),torch.ones(2))
        torch.testing.assert_close(model.predict_proba(single)[0],probabilities[0],rtol=1e-5,atol=1e-6)

    def test_missing_weights_or_wrong_architecture_cannot_load_silently(self):
        model,tok=self.make_model()
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)
            with self.assertRaisesRegex(ValueError,"config does not match"):
                save_model(model,tok,path,self.cfg()|{"classification_pooling":"mean"},{})
            save_model(model,tok,path,self.cfg(),{})
            cfg_path=path/"resolved_config.json"
            cfg_path.write_text(json.dumps(self.cfg()|{"classification_pooling":"mean"}),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"mixed checkpoint"):
                load_checkpoint(path,torch.device("cpu"))
            cfg_path.write_text(json.dumps(self.cfg()),encoding="utf-8")
            (path/"attention_pool.pt").unlink()
            with self.assertRaisesRegex(FileNotFoundError,"missing trained pooling"):
                load_checkpoint(path,torch.device("cpu"))

    def test_backbone_pretrained_path_receives_attention_config(self):
        from vgec.models.seq2seq import load_pretrained
        model,tok=self.make_model()
        with patch("transformers.AutoModelForSeq2SeqLM.from_pretrained",return_value=model.backbone), \
             patch("transformers.AutoTokenizer.from_pretrained",return_value=tok):
            loaded,_=load_pretrained(self.cfg())
        self.assertIsInstance(loaded.attention_pool,AttentionPooling)
        self.assertEqual(loaded.classifier[1].out_features,5)

    def test_mbart_joint_attention_training_and_checkpoint(self):
        from transformers import MBartConfig,MBartForConditionalGeneration
        torch.manual_seed(42)
        _,tok=self.make_model()
        backbone=MBartForConditionalGeneration(MBartConfig(
            vocab_size=5,d_model=16,encoder_layers=1,decoder_layers=1,
            encoder_attention_heads=2,decoder_attention_heads=2,
            encoder_ffn_dim=32,decoder_ffn_dim=32,max_position_embeddings=32,
            pad_token_id=0,eos_token_id=1,bos_token_id=1,decoder_start_token_id=1,
            dropout=0.,attention_dropout=0.))
        model=JointSeq2Seq(backbone,True,"attention")
        model.backbone.gradient_checkpointing_enable()
        model.train()
        batch=collate(self.rows()[:2],tok,self.cfg(),0,torch.device("cpu"))
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        previous=model.attention_pool.score.weight.detach().clone()
        with torch.autocast("cpu",dtype=torch.bfloat16):
            loss,_=model(**batch)
        loss.backward()
        for parameters in (model.attention_pool.score.weight,model.classifier[1].weight,model.backbone.model.shared.weight):
            self.assertIsNotNone(parameters.grad)
            self.assertTrue(torch.isfinite(parameters.grad).all())
            self.assertGreater(float(parameters.grad.abs().sum()),0.)
        optimizer.step()
        self.assertFalse(torch.equal(previous,model.attention_pool.score.weight))
        model.eval()
        inputs=tok(["source"],return_tensors="pt",return_token_type_ids=False)
        with tempfile.TemporaryDirectory() as temp:
            save_model(model,tok,temp,self.cfg()|{"kind":"bartpho"},{})
            loaded,_,_=load_checkpoint(temp,torch.device("cpu"))
            torch.testing.assert_close(model.predict_proba(inputs),loaded.predict_proba(inputs),rtol=0,atol=0)


if __name__=="__main__":
    unittest.main()
