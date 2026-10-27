"""Optional paired source bootstrap with corpus GLEU and shared reference draws."""
import random
import numpy as np
from .metrics import gleu_counts,words


def _corpus_scores(stats):
    valid=(stats>0).all(axis=1); safe=np.maximum(stats,1)
    bp=np.minimum(0.,1.-safe[:,1]/safe[:,0])
    precision=np.log(safe[:,2::2]/safe[:,3::2]).mean(axis=1)
    return np.where(valid,np.exp(bp+precision),0.)


def paired(rows,a,b,samples=2000,seed=42,reference_samples=500,reference_seed=13):
    if not rows or len(rows)!=len(a) or len(a)!=len(b): raise ValueError('Bootstrap coverage mismatch')
    refs=[r['references_raw'] if r['label'] else [r['source']] for r in rows]
    rng=random.Random(reference_seed)
    caches=[[[gleu_counts(words(r['source']),words(h),words(ref)) for ref in rr] for r,h,rr in zip(rows,hyp,refs)] for hyp in (a,b)]
    draws=[[rng.randrange(len(rr)) for rr in refs] for _ in range(reference_samples)]
    arrays=[np.array([[cache[i][draw[i]] for i in range(len(rows))] for draw in draws],dtype=np.int64) for cache in caches]
    exact=np.array([int(x in rr)-int(y in rr) for x,y,rr in zip(a,b,refs)])
    sampler=np.random.default_rng(seed); deltas=[]; em=[]
    for _ in range(samples):
        ids=sampler.integers(0,len(rows),size=len(rows))
        ga,gb=(_corpus_scores(x[:,ids,:].sum(axis=1)).mean() for x in arrays)
        deltas.append(float(ga-gb)); em.append(float(exact[ids].mean()))
    def summary(values):
        return dict(mean=float(np.mean(values)),ci95=[float(v) for v in np.quantile(values,[.025,.975])])
    return dict(N=len(rows),samples=samples,seed=seed,reference_samples=reference_samples,reference_seed=reference_seed,
                paired_delta_gleu=summary(deltas),paired_delta_exact_match=summary(em),
                paired_delta_S_proxy_with_fixed_classifier=summary([x/2 for x in deltas]),bootstrap_fraction_positive=float(np.mean(np.array(deltas)>0)),official=False)
