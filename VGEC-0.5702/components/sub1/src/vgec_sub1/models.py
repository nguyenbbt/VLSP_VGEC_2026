"""Production wrappers; training and generation use the same tokenizer contracts."""
import json
import math
import re
import time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .config import role_config


class AttentionClassifier(nn.Module):
    def __init__(self, encoder, hidden=256, dropout=.1):
        super().__init__(); self.encoder=encoder
        self.attention=nn.Sequential(nn.Linear(encoder.config.hidden_size,hidden),nn.Tanh(),nn.Linear(hidden,1))
        self.dropout=nn.Dropout(dropout); self.head=nn.Linear(encoder.config.hidden_size,5)

    def forward(self,input_ids,attention_mask,special_tokens_mask):
        h=self.encoder(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state
        mask=attention_mask.bool() & ~special_tokens_mask.bool()
        if (~mask.any(-1)).any(): raise ValueError('Classifier source has no nonspecial tokens')
        logits=self.attention(h).squeeze(-1).float().masked_fill(~mask,float('-inf'))
        pooled=(h*logits.softmax(-1).to(h.dtype).unsqueeze(-1)).sum(1)
        return self.head(self.dropout(pooled))


def parse_correction(text):
    def unique(pairs):
        obj={}
        for k,v in pairs:
            if k in obj: raise ValueError('Duplicate JSON key')
            obj[k]=v
        return obj
    obj=json.loads(text,object_pairs_hook=unique)
    if not isinstance(obj,dict) or set(obj)!={'fixed_sentence'} or not isinstance(obj['fixed_sentence'],str) or not obj['fixed_sentence'].strip():
        raise ValueError('Expected exactly one nonempty fixed_sentence string')
    if re.search(r'<\|[^>]+\|>|</?think>',obj['fixed_sentence']): raise ValueError('Control token in correction')
    return obj['fixed_sentence']


def parameter_gate(total,limit):
    if type(total) is not int or not 0<total<limit: raise ValueError(f'Parameter gate failed: {total} >= {limit}')


def causal_sft(tokenizer,prompt,answer,max_length):
    prefix=tokenizer.apply_chat_template(prompt,tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False)
    full=tokenizer.apply_chat_template(prompt+[dict(role='assistant',content=answer)],tokenize=True,return_dict=False,
                                      add_generation_prompt=False,enable_thinking=False)
    if full[:len(prefix)]!=prefix: raise ValueError('Assistant template is not prefix aligned')
    end=tokenizer.convert_tokens_to_ids('<|im_end|>')
    if end is None or end==tokenizer.unk_token_id or end not in full[len(prefix):]: raise ValueError('Missing supervised assistant EOS')
    # Trailing template newline is not an assistant target.
    stop=len(prefix)+full[len(prefix):].index(end)+1; full=full[:stop]
    if len(full)>max_length: raise ValueError('SFT sequence over limit; truncation forbidden')
    return dict(input_ids=full,labels=[-100]*len(prefix)+full[len(prefix):],end_id=end)


class Runtime:
    def __init__(self,cfg,role,training=False,checkpoint=None,device=None):
        started=time.monotonic()
        print(dict(stage='model_loading',role=role,checkpoint=str(checkpoint) if checkpoint else 'pretrained'),flush=True)
        from transformers import AutoConfig,AutoTokenizer,AutoModel,AutoModelForSeq2SeqLM,PreTrainedTokenizerFast
        self.cfg=cfg; self.role=role; self.rc=rc=role_config(cfg,role)
        self.device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        if self.device.type=='cuda' and not torch.cuda.is_bf16_supported(): raise ValueError('BF16 is required on A100')
        config=AutoConfig.from_pretrained(rc['model_id'],revision=rc['revision'],trust_remote_code=False)
        tokenizer_class=PreTrainedTokenizerFast if role=='bami' else AutoTokenizer
        self.tokenizer=(tokenizer_class.from_pretrained(str(Path(checkpoint)/'tokenizer'),local_files_only=True,trust_remote_code=False)
                        if checkpoint else tokenizer_class.from_pretrained(rc['model_id'],revision=rc['revision'],trust_remote_code=False))
        t=self.tokenizer
        if t.pad_token_id is None: t.pad_token=t.eos_token
        if t.pad_token_id is None: raise ValueError('No padding token')
        if role in ('bart','vit5'): cls=AutoModelForSeq2SeqLM
        elif role=='bami': cls=AutoModel
        else:
            from transformers import Qwen3_5ForConditionalGeneration,Qwen3ForCausalLM
            cls=Qwen3_5ForConditionalGeneration if role=='qwen_corrector' else Qwen3ForCausalLM
        with torch.device('meta'):
            meta=cls.from_config(config,**({'add_pooling_layer':False} if role=='bami' else {})) if role in ('bami','bart','vit5') else cls(config)
            if role=='bami': meta=AttentionClassifier(meta,cfg['models']['bami']['attention_hidden_size'],cfg['models']['bami']['dropout'])
            else: meta.tie_weights()
        total=sum(p.numel() for p in meta.parameters()); del meta
        parameter_gate(total,rc['limit'])
        # Full FT uses fp32 master parameters plus BF16 autocast. LoRA base uses BF16.
        dtype=torch.bfloat16 if role in ('qwen_corrector','reranker') and self.device.type=='cuda' else torch.float32
        kw=dict(revision=rc['revision'],trust_remote_code=False,torch_dtype=dtype)
        if role=='bami': kw['add_pooling_layer']=False
        if role in ('qwen_corrector','reranker'): kw['attn_implementation']='sdpa'
        net=cls.from_pretrained(rc['model_id'],**kw)
        if role=='bami': net=AttentionClassifier(net,cfg['models']['bami']['attention_hidden_size'],cfg['models']['bami']['dropout'])
        targets=[]
        if role in ('qwen_corrector','reranker') and (training or checkpoint):
            from peft import LoraConfig,PeftModel,get_peft_model
            if checkpoint:
                net=PeftModel.from_pretrained(net,str(Path(checkpoint)/'adapter'),is_trainable=training)
                targets=sorted(net.peft_config['default'].target_modules)
            else:
                targets=[name for name,m in net.named_modules() if isinstance(m,nn.Linear)
                         and '.layers.' in name and 'visual' not in name and 'vision' not in name
                         and (role!='qwen_corrector' or '.language_model.' in name)]
                if not targets: raise ValueError('No text LoRA linear targets found')
                net=get_peft_model(net,LoraConfig(r=rc['lora_r'],lora_alpha=rc['lora_alpha'],lora_dropout=rc['lora_dropout'],
                                                target_modules=targets,bias='none'))
            if any(p.requires_grad for n,p in net.named_parameters() if ('visual' in n or 'vision' in n)):
                raise ValueError('Vision parameters unexpectedly trainable')
        elif checkpoint:
            net.load_state_dict(torch.load(Path(checkpoint)/'model.pt',map_location='cpu',weights_only=True))
        self.net=net.to(self.device)
        if not training: self.net.requires_grad_(False)
        adapter_count=sum(p.numel() for n,p in net.named_parameters() if 'lora_' in n)
        parameter_gate(total+adapter_count,rc['limit'])
        loaded_total=sum(p.numel() for p in net.parameters())
        if loaded_total!=total+adapter_count: raise ValueError('Loaded logical parameter count differs from meta architecture + adapter')
        self.parameter_report=dict(base_parameters=total,adapter_parameters=adapter_count,total_parameters=total+adapter_count,
                                   loaded_parameters=loaded_total,
                                   trainable_parameters=sum(p.numel() for p in net.parameters() if p.requires_grad),targets=targets,
                                   count_method='unique_unquantized_meta_full_architecture_plus_head_and_adapter')
        if training and rc['checkpointing']:
            target=net.encoder if role=='bami' else net
            target.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            if hasattr(target,'enable_input_require_grads') and role in ('qwen_corrector','reranker'): target.enable_input_require_grads()
            if hasattr(target,'config'): target.config.use_cache=False
        if role=='qwen_corrector':
            self.end_id=t.convert_tokens_to_ids('<|im_end|>')
            if self.end_id is None or self.end_id==t.unk_token_id: raise ValueError('Unknown assistant end token')
            net.generation_config.eos_token_id=self.end_id
            net.generation_config.pad_token_id=t.pad_token_id
            self.token_contract=dict(tokenizer_eos=t.eos_token_id,assistant_end=self.end_id,
                                     generation_eos=net.generation_config.eos_token_id,pad=t.pad_token_id)
        elif role=='reranker':
            self.yes=t.encode('yes',add_special_tokens=False); self.no=t.encode('no',add_special_tokens=False)
            if len(self.yes)!=1 or len(self.no)!=1 or self.yes==self.no: raise ValueError('yes/no must be distinct single tokens')
        self.net.train(training)
        print(dict(stage='model_loaded',role=role,seconds=round(time.monotonic()-started,2),
                   parameters=loaded_total,trainable=self.parameter_report['trainable_parameters']),flush=True)

    def autocast(self):
        return torch.autocast(device_type=self.device.type,dtype=torch.bfloat16,enabled=self.device.type=='cuda')

    def messages(self,source):
        return [dict(role='system',content=self.rc['prompt']),dict(role='user',content=source)]

    def ids(self,text,limit,special=True):
        ids=self.tokenizer.encode(text,add_special_tokens=special)
        if len(ids)>limit: raise ValueError(f'{self.role} token limit exceeded: {len(ids)} > {limit}; no truncation')
        return ids

    def encode(self,row):
        t=self.tokenizer; rc=self.rc; role=self.role
        if role=='bami':
            ids=self.ids(row['source'],rc['max_length'])
            special=t.get_special_tokens_mask(ids,already_has_special_tokens=True)
            return dict(input_ids=ids,special_tokens_mask=special,label=row['label'],uid=row['uid'])
        if role in ('bart','vit5'):
            source=self.ids(rc['prefix']+row['source'],rc['max_length'])
            refs=[self.ids(r,rc['target_length']) for r in row['references']]
            return dict(input_ids=source,refs=refs,uid=row['uid'])
        if role=='qwen_corrector':
            refs=[causal_sft(t,self.messages(row['source']),json.dumps({'fixed_sentence':r},ensure_ascii=False),rc['max_length']) for r in row['references']]
            return dict(refs=refs,uid=row['uid'])
        return row

    def pad(self,seqs,left=False,value=None):
        value=self.tokenizer.pad_token_id if value is None else value
        width=max(map(len,seqs)); out=torch.full((len(seqs),width),value,dtype=torch.long,device=self.device)
        mask=torch.zeros_like(out)
        for k,s in enumerate(seqs):
            start=width-len(s) if left else 0
            out[k,start:start+len(s)]=torch.tensor(s,device=self.device); mask[k,start:start+len(s)]=1
        return out,mask

    def source_losses(self,rows):
        if self.role=='reranker': return self.rank_losses(rows)
        if self.role=='bami':
            ids,mask=self.pad([r['input_ids'] for r in rows]); specials,_=self.pad([r['special_tokens_mask'] for r in rows],value=1)
            logits=self.net(ids,mask,specials)
            return F.cross_entropy(logits.float(),torch.tensor([r['label'] for r in rows],device=self.device),reduction='none')
        examples=[]; owners=[]
        for i,r in enumerate(rows):
            for ref in r['refs']:
                examples.append(dict(input_ids=r['input_ids'],labels=ref) if self.role in ('bart','vit5') else ref)
                owners.append(i)
        ids,mask=self.pad([e['input_ids'] for e in examples]); labels,_=self.pad([e['labels'] for e in examples],value=-100)
        if self.role in ('bart','vit5'):
            decoder_ids=self.net.prepare_decoder_input_ids_from_labels(labels=labels)
            logits=self.net(input_ids=ids,attention_mask=mask,decoder_input_ids=decoder_ids,use_cache=False).logits
        else:
            first=int((labels!=-100).any(0).nonzero()[0]); start=max(0,first-1)
            logits=self.net(input_ids=ids,attention_mask=mask,use_cache=False,logits_to_keep=ids.shape[1]-start).logits[:,:-1]
            labels=labels[:,start+1:]
        active=labels!=-100
        token_loss=F.cross_entropy(logits[active].float(),labels[active],reduction='none')
        example_ids=active.nonzero(as_tuple=True)[0]
        per_ref=torch.zeros(labels.size(0),device=logits.device).scatter_add(0,example_ids,token_loss)/active.sum(1).clamp_min(1)
        return torch.stack([per_ref[torch.tensor([j for j,o in enumerate(owners) if o==i],device=self.device)].mean() for i in range(len(rows))])

    def rank_ids(self,source,candidate):
        cache=getattr(self,'_rank_ids_cache',None)
        if cache is None: self._rank_ids_cache=cache={}
        key=(source,candidate)
        if key in cache: return cache[key]
        prefix='<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        suffix='<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
        body=f'<Instruct>: {self.rc["prompt"]}\n<Query>: {source}\n<Document>: {candidate}'
        ids=sum((self.tokenizer.encode(x,add_special_tokens=False) for x in (prefix,body,suffix)),[])
        if len(ids)>self.rc['max_length']: raise ValueError('Reranker pair exceeds max tokens')
        if len(cache)<32768: cache[key]=ids
        return ids

    def margins(self,pairs):
        ids,mask=self.pad([self.rank_ids(s,c) for s,c in pairs],left=True)
        logits=self.net(input_ids=ids,attention_mask=mask,use_cache=False,logits_to_keep=1).logits[:,-1,:].float()
        return logits[:,self.yes[0]]-logits[:,self.no[0]]

    def rank_losses(self,rows):
        pairs=[(r['source'],c['text']) for r in rows for c in r['candidates']]
        scores=self.margins(pairs); losses=[]; offset=0
        for r in rows:
            s=scores[offset:offset+len(r['candidates'])]; offset+=len(s)
            terms=[F.softplus(-(s[p['better']]-s[p['worse']])) for p in r['pairs']]
            if not terms: raise ValueError('No audited pair in source batch')
            loss=torch.stack(terms).mean()
            binary=[F.binary_cross_entropy_with_logits(s[i],torch.tensor(float(c['quality']),device=s.device)) for i,c in enumerate(r['candidates']) if c.get('quality') is not None]
            if binary: loss=loss+self.rc['binary_weight']*torch.stack(binary).mean()
            losses.append(loss)
        return torch.stack(losses)

    @torch.no_grad()
    def classify(self,sources):
        rows=[self.encode(dict(source=s,label=0,uid=str(i))) for i,s in enumerate(sources)]
        ids,mask=self.pad([r['input_ids'] for r in rows]); special,_=self.pad([r['special_tokens_mask'] for r in rows],value=1)
        with self.autocast(): logits=self.net(ids,mask,special)
        return logits.float().argmax(-1).tolist()

    @torch.no_grad()
    def generate(self,sources,two=True):
        self.last_generation_trace=[[] for _ in sources]
        if self.role in ('bart','vit5'):
            ids,mask=self.pad([self.ids(self.rc['prefix']+s,self.rc['max_length']) for s in sources])
            c=self.cfg['models'][self.role]; count=2 if two else 1
            with self.autocast():
                seqs=self.net.generate(input_ids=ids,attention_mask=mask,num_beams=c['num_beams'],num_return_sequences=count,
                                       max_new_tokens=self.rc['target_length'],do_sample=False,length_penalty=c['length_penalty'],use_cache=True)
            decoded=[]
            eos=self.net.generation_config.eos_token_id; eos=[eos] if isinstance(eos,int) else eos
            for k,seq in enumerate(seqs.tolist()):
                # decoder start may equal EOS; inspect after that token.
                ended=any(v in eos for v in seq[1:])
                text=self.tokenizer.decode(seq,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                first_end=next((i for i,v in enumerate(seq[1:],1) if v in eos),None)
                forced_boundary=first_end==self.rc['target_length'] and self.net.generation_config.forced_eos_token_id is not None
                ended=ended and not forced_boundary
                self.last_generation_trace[k//count].append(dict(mode='beam',beam=k%count,ended=ended,forced_boundary=forced_boundary,text=text,tokens=len(seq)))
                decoded.append(text if ended and text.strip() else None)
            return [[v for v in decoded[i*count:(i+1)*count] if v is not None] for i in range(len(sources))]
        c=self.cfg['models']['qwen_corrector']
        prompts=[self.tokenizer.apply_chat_template(self.messages(s),tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False) for s in sources]
        if any(len(p)+self.rc['max_new_tokens']>self.rc['max_length'] for p in prompts): raise ValueError('Prompt + generation budget exceeds context contract')
        ids,mask=self.pad(prompts,left=True); result=[[] for _ in sources]
        for sample in ([False,True] if two else [False]):
            options=dict(do_sample=sample,max_new_tokens=self.rc['max_new_tokens'],eos_token_id=self.end_id,
                         pad_token_id=self.tokenizer.pad_token_id,repetition_penalty=c['repetition_penalty'],use_cache=True)
            if sample: options.update(temperature=c['sampling_temperature'],top_p=c['sampling_top_p'])
            with self.autocast(): seqs=self.net.generate(input_ids=ids,attention_mask=mask,**options)
            for i,seq in enumerate(seqs[:,ids.shape[1]:].tolist()):
                if self.end_id not in seq:
                    self.last_generation_trace[i].append(dict(mode='sampling' if sample else 'greedy',ended=False,tokens=len(seq),reason='output_budget_exhausted'))
                    continue
                text=self.tokenizer.decode(seq[:seq.index(self.end_id)],skip_special_tokens=False,clean_up_tokenization_spaces=False)
                trace=dict(mode='sampling' if sample else 'greedy',ended=True,tokens=seq.index(self.end_id)+1,raw=text)
                try:
                    trace['text']=parse_correction(text)
                    result[i].append(trace['text']); trace['schema_valid']=True
                except (ValueError,json.JSONDecodeError) as e:
                    trace.update(schema_valid=False,reason=str(e))
                self.last_generation_trace[i].append(trace)
        return result

    def save_model(self,destination):
        destination=Path(destination); destination.mkdir(parents=True,exist_ok=True)
        self.tokenizer.save_pretrained(destination/'tokenizer')
        if self.role in ('qwen_corrector','reranker'): self.net.save_pretrained(destination/'adapter')
        else: torch.save(self.net.state_dict(),destination/'model.pt')

    def reload_weights(self,destination):
        if self.role in ('qwen_corrector','reranker'):
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file
            set_peft_model_state_dict(self.net,load_file(str(Path(destination)/'adapter/adapter_model.safetensors')))
        else: self.net.load_state_dict(torch.load(Path(destination)/'model.pt',map_location=self.device,weights_only=True))
