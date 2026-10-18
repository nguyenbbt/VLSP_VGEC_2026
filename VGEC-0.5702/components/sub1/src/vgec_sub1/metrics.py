"""Reproducible diagnostics. These are proxies, never the official VGEC scorer."""
from collections import Counter
import math
import random
import unicodedata

def words(text):
    return unicodedata.normalize('NFC',text).split()


def classification(gold, predicted):
    if len(gold)!=len(predicted) or not gold: raise ValueError('Metric requires nonempty equal lists')
    f1=[]; by={}
    for k in range(5):
        tp=sum(g==k and p==k for g,p in zip(gold,predicted))
        fp=sum(g!=k and p==k for g,p in zip(gold,predicted))
        fn=sum(g==k and p!=k for g,p in zip(gold,predicted))
        score=2*tp/max(2*tp+fp+fn,1); f1.append(score)
        by[str(k)]=dict(f1=score,support=sum(g==k for g in gold))
    return dict(macro_f1=sum(f1)/5,micro_f1=sum(g==p for g,p in zip(gold,predicted))/len(gold),per_label=by,N=len(gold))


def ngrams(words,n):
    return Counter(tuple(words[i:i+n]) for i in range(len(words)-n+1))


def gleu_counts(source,hypothesis,reference,order=4):
    # GLEU+ 2016: clip against reference and penalize source-only ngrams.
    stats=[len(hypothesis),len(reference)]
    for n in range(1,order+1):
        s,h,r=(ngrams(x,n) for x in (source,hypothesis,reference))
        good=sum((h&r).values())
        source_only=Counter({k:v for k,v in s.items() if k not in r})
        bad=sum((h&source_only).values())
        stats.extend([max(good-bad,0),max(len(hypothesis)-n+1,0)])
    return stats


def gleu(stats):
    if any(x==0 for x in stats): return 0.0
    c,r=stats[:2]
    return math.exp(min(0,1-r/c)+sum(math.log(stats[i]/stats[i+1]) for i in range(2,len(stats),2))/((len(stats)-2)/2))


def corpus_gleu(sources,hypotheses,references,samples=500,seed=13):
    if not sources or not len(sources)==len(hypotheses)==len(references): raise ValueError('Invalid GLEU inputs')
    rng=random.Random(seed); scores=[]
    cached=[[gleu_counts(words(s),words(h),words(r)) for r in refs] for s,h,refs in zip(sources,hypotheses,references)]
    if any(not r for r in cached): raise ValueError('GLEU requires references for every source')
    for _ in range(samples):
        total=[0]*10
        for choices in cached:
            stat=choices[rng.randrange(len(choices))]
            total=[a+b for a,b in zip(total,stat)]
        scores.append(gleu(total))
    return sum(scores)/len(scores)


def rouge_l(hypothesis,reference):
    a,b=words(hypothesis),words(reference); prev=[0]*(len(b)+1)
    for x in a:
        row=[0]
        for j,y in enumerate(b): row.append(prev[j]+1 if x==y else max(row[-1],prev[j+1]))
        prev=row
    return 2*prev[-1]/max(len(a)+len(b),1)


def corrections(rows,hypotheses,samples=500,seed=13,reference_view='raw'):
    if len(rows)!=len(hypotheses) or not rows: raise ValueError('Invalid correction metric coverage')
    refs=[r['references_raw' if reference_view=='raw' else 'references'] if r['label'] else [r['source']] for r in rows]
    chosen=[i for i,r in enumerate(refs) if r]
    rows=[rows[i] for i in chosen]; hypotheses=[hypotheses[i] for i in chosen]; refs=[refs[i] for i in chosen]
    if not rows: raise ValueError('No correction-valid references for metric')
    return dict(N=len(rows),exact_match=sum(h in rr for h,rr in zip(hypotheses,refs))/len(rows),
                gleu_plus_2016_proxy=corpus_gleu([r['source'] for r in rows],hypotheses,refs,samples,seed),
                rouge_l_proxy=sum(max(rouge_l(h,r) for r in rr) for h,rr in zip(hypotheses,refs))/len(rows))
