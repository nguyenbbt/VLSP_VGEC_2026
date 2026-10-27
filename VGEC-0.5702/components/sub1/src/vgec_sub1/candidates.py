"""Raw-text candidates, deterministic codepoint edits and independent serialization."""
import math
import re
import hashlib
from .io import digest, write_json


def alignment(source, target):
    n, m = len(source), len(target)
    if max(n, m) > 8000:
        raise ValueError('Alignment input exceeds audited resource limit')
    from array import array
    dp = [array('I', range(m+1))]
    for i in range(1, n+1):
        row = array('I', [i]+[0]*m)
        for j in range(1, m+1):
            row[j] = min(dp[i-1][j-1]+(source[i-1]!=target[j-1]), dp[i-1][j]+1, row[j-1]+1)
        dp.append(row)
    # Backtrack with the documented priority at each DP cell.
    ops = []
    i, j = n, m
    while i or j:
        if i and j and source[i-1] == target[j-1] and dp[i][j] == dp[i-1][j-1]:
            ops.append(('match', i-1, j-1)); i-=1; j-=1
        elif i and j and dp[i][j] == dp[i-1][j-1]+1:
            ops.append(('substitute', i-1, j-1)); i-=1; j-=1
        elif i and dp[i][j] == dp[i-1][j]+1:
            ops.append(('delete', i-1, j)); i-=1
        else:
            ops.append(('insert', i, j-1)); j-=1
    ops.reverse()
    hunks = []
    si = ti = 0
    start = None
    for op, _, _ in ops+[('match', n, m)]:
        if op == 'match':
            if start is not None:
                a,b = start
                hunks.append(dict(start=a, end=si, replacement=target[b:ti]))
                start=None
            si+=1; ti+=1
        else:
            if start is None: start=(si,ti)
            si += op != 'insert'
            ti += op != 'delete'
    rebuilt = apply_edits(source, hunks)
    if rebuilt != target: raise AssertionError('Edit roundtrip failed')
    risks=[]
    if re.findall(r'\d+(?:[.,]\d+)*', source) != re.findall(r'\d+(?:[.,]\d+)*', target): risks.append('number_changed')
    negatives = r'\b(?:không|chưa|chẳng|chả|đừng)\b'
    if re.findall(negatives, source.lower()) != re.findall(negatives, target.lower()): risks.append('negation_changed')
    if dp[n][m]/max(n,1) > .3: risks.append('high_edit_ratio')
    return dict(distance=int(dp[n][m]), hunks=hunks, roundtrip=True, soft_risks=risks)


def apply_edits(source, hunks):
    cursor=0; pieces=[]
    for h in hunks:
        if not cursor <= h['start'] <= h['end'] <= len(source): raise ValueError('Invalid edit offsets')
        pieces.extend([source[cursor:h['start']],h['replacement']]); cursor=h['end']
    return ''.join(pieces)+source[cursor:]


def pool(source, generations):
    limits=dict(bart=2,vit5=2,qwen=2)
    merged={}
    for family in (*limits,'copy'):
        texts=[source] if family=='copy' else generations.get(family,[])
        if len(texts) > limits.get(family,1): raise ValueError('Candidate family overflow')
        for k,item in enumerate(texts):
            text=item.get('text') if isinstance(item,dict) else item
            if not isinstance(text,str) or not text.strip(): continue
            if any(marker in text for marker in ('<|im_start|>','<|im_end|>','<think>','</think>')): continue
            if text not in merged:
                merged[text]=dict(text=text, provenance=[], **alignment(source,text))
            provenance=dict(family=family,index=item.get('index',k) if isinstance(item,dict) else k)
            if isinstance(item,dict) and item.get('mode'): provenance['mode']=item['mode']
            merged[text]['provenance'].append(provenance)
    for c in merged.values():
        c['agreement']=len({p['family'] for p in c['provenance'] if p['family']!='copy'})
        c['text_hash']=hashlib.sha256(c['text'].encode('utf-8')).hexdigest()
    return list(merged.values())


def select(source, candidates, scores, tolerance=1e-6):
    if len(candidates)!=len(scores): raise ValueError('Reranker score count mismatch')
    valid=[(c,float(s)) for c,s in zip(candidates,scores) if math.isfinite(float(s)) and c['roundtrip']]
    if not valid: return dict(text=source, fallback='no_finite_valid_candidate')
    top=max(s for _,s in valid)
    tied=[(c,s) for c,s in valid if top-s <= tolerance]
    c,s=min(tied,key=lambda x:(-x[0]['agreement'],x[0]['distance'],x[0]['text_hash']))
    return dict(**c, score=s, fallback=None)


def serialize(inputs, labels, selections):
    if len(inputs)!=len(labels) or len(inputs)!=len(selections): raise ValueError('Output coverage mismatch')
    seen=set(); result=[]
    for row,label,selected in zip(inputs,labels,selections):
        idkey=(type(row['ID']).__name__,row['ID'])
        if type(row['ID']) not in (int,str) or idkey in seen: raise ValueError('Duplicate/invalid typed ID')
        seen.add(idkey)
        if type(label) is not int or label not in range(5): raise ValueError('Invalid output label')
        text=selected['text']
        if not isinstance(text,str) or not text.strip(): raise ValueError('Invalid selected correction')
        result.append(dict(ID=row['ID'],Output={'Error_Type_Id':label,'Fixed Sentence':[] if label==0 else [text]}))
    return result


def export(inputs, labels, selections, destination):
    import zipfile
    from pathlib import Path
    destination=Path(destination); destination.mkdir(parents=True,exist_ok=True)
    result=serialize(inputs,labels,selections)
    write_json(destination/'results.json',result)
    tmp=destination/'submission.zip.tmp'
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED) as z:
        z.write(destination/'results.json','results.json')
    tmp.replace(destination/'submission.zip')
    return dict(rows=len(result),results_hash=digest(result))
