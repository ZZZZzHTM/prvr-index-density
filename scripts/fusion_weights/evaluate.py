import json
import hashlib
from pathlib import Path
import numpy as np
import torch

def twin_secondary(root, output):
    files=[root/'test'/f'clip{i}_scores.npy' for i in (1,2)]
    scores=[np.load(p,mmap_mode='r') for p in files]
    ids=json.loads((root/'test/id_order.json').read_text(encoding='utf8'));gt=ids['q2v_gt']
    if isinstance(gt,dict):gt=[gt[str(i)] if str(i) in gt else gt[i] for i in range(len(ids['query_ids']))]
    gt=np.asarray(gt).reshape(-1);assert len(gt)==scores[0].shape[0]
    ranks=[]
    for start in range(0,len(gt),256):
        # Same fp32 order as official score fusion; no checkpoint evaluation.
        a=torch.from_numpy(np.array(scores[0][start:start+256],copy=True));b=torch.from_numpy(np.array(scores[1][start:start+256],copy=True))
        s=.7*a+.3*b;order=torch.argsort(-s,dim=1,stable=False)
        ranks.extend((order==torch.as_tensor(gt[start:start+len(order)])[:,None]).nonzero()[:,1].add(1).tolist())
    assert len(ranks)==len(gt);ranks=np.asarray(ranks);out=Path(output);np.save(out,ranks)
    metrics={f'R@{k}':float(100*(ranks<=k).mean()) for k in (1,5,10,100)};metrics['SumR']=sum(metrics.values())
    return metrics,dict(formula='fp32 .7*clip1+.3*clip2; CPU torch.argsort(-scores), stable=False; percent hits at1/5/10/100',
        score_files=[ref(p,'full_score_matrix') for p in files],id_order=ref(root/'test/id_order.json','q2v_gt'),ranks=ref(out,'all positive ranks'),
        CPU_ranking_may_differ_from_GPU_on_ties=True)


def ref(path, field):
    path = Path(path)
    return {'file': path.name, 'field':field, 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
