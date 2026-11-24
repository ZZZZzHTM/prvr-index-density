"""Numerical implementation used by the reported experiment."""
COARSE_KEEPALIVE=[]
import ctypes
import faiss
import math
import numpy as np
import subprocess
from threadpoolctl import threadpool_limits
import time
import torch
from pathlib import Path
import json
import os
HERE=Path(__file__).resolve().parent
RUNTIME=Path(os.environ.get('PRVR_WORK','work'))
def read(p):
    return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,v):
    p=Path(p)
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf8')


def metric(ranks):
    m={f'R@{k}':float(100*np.mean(ranks<=k)) for k in (1,5,10,100)}
    m['SumR']=sum(m.values());return m


def rank_scores(matrix,truth):
    order=torch.argsort(-torch.from_numpy(np.ascontiguousarray(matrix)),dim=1,stable=False)
    return (order==torch.from_numpy(truth)[:,None]).long().argmax(1).numpy()+1


def arr(p):return np.load(p,mmap_mode='r',allow_pickle=False)


def cluster_draws(truth):
    _,inv=np.unique(truth,return_inverse=True);n=int(inv.max()+1)
    counts=np.bincount(inv,minlength=n);rng=np.random.default_rng(20261023)
    weights=np.empty((2000,n),np.float64)
    for b in range(2000):weights[b]=np.bincount(rng.integers(0,n,size=n),minlength=n)
    return weights,counts,inv


def paired(ranks,base,draws):
    weights,counts,inv=draws
    delta=sum((ranks<=k).astype(np.float64)-(base<=k) for k in [1,5,10,100])
    by_video=np.bincount(inv,weights=delta,minlength=len(counts))
    samples=100*(weights@by_video)/(weights@counts)
    ci=np.quantile(samples,[.025,.975],method='linear')
    return dict(point=float(100*delta.mean()),ci_low=float(ci[0]),ci_high=float(ci[1]),
        n_query=len(ranks),n_video=len(counts),bootstrap='paired positive-video cluster,2000,default_rng(20261023),linear[.025,.975]')


def build_adc():
    lib=RUNTIME/'adc_max.so'
    subprocess.run(['g++','-O3','-mavx2','-fopenmp','-fPIC','-shared','-ffp-contract=off',str(HERE/'adc_max.cpp'),'-o',str(lib)],check=True)
    loaded=ctypes.CDLL(str(lib));fn=loaded.adc_max
    fn.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int64]*3+[ctypes.c_int]*2+[ctypes.c_void_p]
    fn.restype=None;return loaded,fn


def index_model(folder,name,kind,m,train_meta,test_meta):
    target=folder/(name+'.faiss');record=folder/(name+'.json')
    if target.exists() and record.exists():return faiss.read_index(str(target)),read(record)
    train=arr(train_meta['vectors']);test=arr(test_meta['vectors']);start=time.perf_counter()
    if kind=='pq':index=faiss.IndexPQ(384,m,8,faiss.METRIC_INNER_PRODUCT)
    else:
        nlist=int(math.floor(4*math.sqrt(len(test))+.5))
        coarse=faiss.IndexFlatIP(384)
        index=faiss.IndexIVFPQ(coarse,384,nlist,m,8,faiss.METRIC_INNER_PRODUCT)
        # FAISS 1.15 disallows arbitrary attributes on SWIG index objects.
        COARSE_KEEPALIVE.append(coarse)
        index.cp.seed=20261023;index.cp.niter=25;index.cp.max_points_per_centroid=256
    index.pq.cp.seed=20261023;index.pq.cp.niter=25;index.pq.cp.max_points_per_centroid=256
    print('train-only codebook',name,'training vectors',len(train),'test vectors',len(test),flush=True)
    index.train(train)
    training_seconds=time.perf_counter()-start
    index.add(test);faiss.write_index(index,str(target))
    meta=dict(kind=kind,m=m,nbits=8,metric='inner product',dimension=384,train_only=True,
        n_train_vectors=len(train),n_test_vectors=len(test),n_train_video=train_meta['n_video'],n_test_video=test_meta['n_video'],
        train_source=train_meta['vectors'],test_source=test_meta['vectors'],index_path=str(target),
        serialized_bytes=target.stat().st_size,PQ_code_bytes=int(len(test)*m),
        PQ_codebook_bytes=int(faiss.vector_to_array(index.pq.centroids).nbytes),
        coarse_codebook_bytes=int(index.nlist*384*4) if kind=='ivfpq' else 0,
        nlist=int(index.nlist) if kind=='ivfpq' else None,by_residual=bool(index.by_residual) if kind=='ivfpq' else None,
        IVF_vector_ID_bytes=int(len(test)*8) if kind=='ivfpq' else 0,
        codebook_training_seconds=training_seconds,seed=20261023,niter=25,max_points_per_centroid=256,
        native_PQ_training_sample_limit=256*256,training_test_vectors_used=0)
    write(record,meta);return index,meta


def adc_branch(index,meta,queries,test_meta,output,fn,zero_mask=None):
    if output.exists():return arr(output)
    m=index.pq.M;codes=faiss.vector_to_array(index.codes).reshape(-1,m)
    codes=np.ascontiguousarray(codes.T);offsets=np.array(arr(test_meta['offsets']),dtype=np.int64,copy=True)
    nv=len(offsets)-1
    result=np.lib.format.open_memmap(output,mode='w+',dtype=np.float32,shape=(len(queries),nv))
    for begin in range(0,len(queries),32):
        q=np.ascontiguousarray(queries[begin:begin+32]);tables=np.empty((len(q),m,256),np.float32)
        index.pq.compute_inner_prod_tables(len(q),faiss.swig_ptr(q),faiss.swig_ptr(tables))
        scores=np.empty((len(q),nv),np.float32)
        fn(tables.ctypes.data,codes.ctypes.data,offsets.ctypes.data,len(q),nv,codes.shape[1],m,4,scores.ctypes.data)
        if zero_mask is not None:scores[:,zero_mask]=np.maximum(scores[:,zero_mask],0)
        result[begin:begin+len(q)]=scores
        if begin%1024==0:print('ADC',output.name,begin,'/',len(queries),flush=True)
    result.flush();return result


def storage(branches,test,nv,m,ivf=False):
    vectors=sum(test[b]['n_vectors'] for b in ['clip','frame']);codebytes=sum(v['PQ_code_bytes'] for v in branches.values())
    codebook=sum(v['PQ_codebook_bytes'] for v in branches.values());coarse=sum(v['coarse_codebook_bytes'] for v in branches.values())
    result=dict(mean_effective_vectors=vectors/nv,PQ_codes_bytes_per_video=codebytes/nv,PQ_codes_total_bytes=codebytes,
        PQ_codebook_bytes=codebook,coarse_codebook_bytes=coarse,codebooks_total_bytes=codebook+coarse,
        offsets_bytes_per_video=sum(Path(test[b]['offsets']).stat().st_size-128 for b in ['clip','frame'])/nv,
        serialized_indices_total_bytes=sum(v['serialized_bytes'] for v in branches.values()),
        IVF_vector_ID_bytes_per_video=sum(v['IVF_vector_ID_bytes'] for v in branches.values())/nv,
        vector_to_video_mapping_bytes_per_video=(vectors*8/nv if ivf else 0),
        fp32_reranking_vectors_bytes_per_video=(vectors*384*4/nv if ivf else 0),
        PQ_plus_fp32_reranking_bytes_per_video=(codebytes+vectors*384*4)/nv if ivf else codebytes/nv)
    return result


def pq_configuration(ds,start,m,inputs,folder,queries,truth,base,draws,fn):
    key=f'{start}_pq_m{m}';done=folder/(key+'_RESULT.json')
    if done.exists():return read(done)
    kind='raw_frame' if start=='raw' else 'ward8_frame';models={};metadata={};scores={};test={};t0=time.perf_counter()
    for branch,branch_kind in [('clip','clip'),('frame',kind)]:
        train_meta=inputs['train'][branch_kind];test_meta=inputs['test'][branch_kind];test[branch]=test_meta
        name=f'{branch_kind}_pq_m{m}';models[branch],metadata[branch]=index_model(folder,name,'pq',m,train_meta,test_meta)
        zero_mask=(arr(inputs['test']['lengths'])<128) if start=='raw' and branch=='frame' else None
        scores[branch]=adc_branch(models[branch],metadata[branch],queries,test_meta,folder/(name+'_ADC_scores.npy'),fn,zero_mask)
    ranks=np.empty(len(truth),np.int32)
    for begin in range(0,len(truth),64):
        end=min(begin+64,len(truth));ranks[begin:end]=rank_scores(.7*scores['clip'][begin:end]+.3*scores['frame'][begin:end],truth[begin:end])
    rankpath=folder/(key+'_ranks.npy');np.save(rankpath,ranks,allow_pickle=False)
    result=dict(dataset=ds,start=start,configuration=f'pq_m{m}',metrics=metric(ranks),delta_vs_uncompressed_fp32=paired(ranks,base,draws),
        index_slots=160 if start=='raw' else 40,storage=storage(metadata,test,len(arr(test['clip']['offsets']))-1,m),
        branch_indices=metadata,sources=dict(ranks=str(rankpath),ADC_scores={k:str(folder/(f'{("clip" if k=="clip" else kind)}_pq_m{m}_ADC_scores.npy')) for k in ['clip','frame']}),
        seconds=time.perf_counter()-t0,ADC_implementation='FAISS IndexPQ codes + compute_inner_prod_tables; exhaustive LUT accumulation and per-video max; no reconstruction or renormalization')
    write(done,result);print(key,result['metrics'],flush=True);return result


def ivf_configuration(ds,nprobe,inputs,folder,queries,truth,base,draws):
    key=f'ward8_ivfpq_nprobe{nprobe}';done=folder/(key+'_RESULT.json')
    if done.exists():return read(done)
    models={};metadata={};test={};vecs={};offsets={};owners={};t0=time.perf_counter()
    for branch,kind in [('clip','clip'),('frame','ward8_frame')]:
        test[branch]=inputs['test'][kind]
        models[branch],metadata[branch]=index_model(folder,kind+'_ivfpq_m48','ivfpq',48,inputs['train'][kind],test[branch])
        models[branch].nprobe=nprobe
        vecs[branch]=arr(test[branch]['vectors']);offsets[branch]=np.asarray(arr(test[branch]['offsets']))
        owners[branch]=np.repeat(np.arange(len(offsets[branch])-1,dtype=np.int64),np.diff(offsets[branch]))
    ranks=np.full(len(truth),np.inf,np.float64);counts=np.zeros(len(truth),np.int64);hit=np.zeros(len(truth),bool);elapsed=np.zeros(len(truth))
    # Entire measured per-query workload is single-threaded, including BLAS and FAISS.
    faiss.omp_set_num_threads(1);torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        for qi,q in enumerate(queries):
            ts=time.perf_counter();candidates=[]
            for branch in ['clip','frame']:
                _,indices=models[branch].search(q[None,:],min(1000,metadata[branch]['n_test_vectors']))
                candidates.append(owners[branch][indices[0,indices[0]>=0]])
            candidates=np.unique(np.concatenate(candidates));counts[qi]=len(candidates);hit[qi]=np.any(candidates==truth[qi])
            branch_scores=[]
            for branch in ['clip','frame']:
                off=offsets[branch];sizes=off[candidates+1]-off[candidates]
                all_ids=np.concatenate([np.arange(off[v],off[v+1]) for v in candidates])
                sim=np.asarray(vecs[branch][all_ids])@q
                starts=np.r_[0,np.cumsum(sizes[:-1])]
                branch_scores.append(np.maximum.reduceat(sim,starts))
            fused=.7*branch_scores[0]+.3*branch_scores[1]
            order=torch.argsort(-torch.from_numpy(fused),stable=False).numpy()
            if hit[qi]:ranks[qi]=np.flatnonzero(candidates[order]==truth[qi])[0]+1
            elapsed[qi]=time.perf_counter()-ts
            if qi%1000==0:print(ds,'IVF',nprobe,qi,'/',len(truth),flush=True)
    torch.set_num_threads(4);faiss.omp_set_num_threads(4)
    rankpath=folder/(key+'_ranks.npy');np.save(rankpath,ranks,allow_pickle=False)
    for name,value in [('candidate_counts',counts),('positive_in_candidates',hit),('query_seconds',elapsed)]:np.save(folder/(key+'_'+name+'.npy'),value,allow_pickle=False)
    result=dict(dataset=ds,start='ward8',configuration=f'ivfpq_nprobe{nprobe}',metrics=metric(ranks),
        delta_vs_uncompressed_fp32=paired(ranks,base,draws),index_slots=40,storage=storage(metadata,test,len(offsets['clip'])-1,48,ivf=True),
        branch_indices=metadata,positive_candidate_fraction=float(hit.mean()),mean_candidate_videos=float(counts.mean()),
        median_single_thread_query_seconds=float(np.median(elapsed)),nprobe=nprobe,top_vectors_per_branch=1000,
        timing_scope='two branch CPU searches + candidate union + actual fp32 vector dot/max + fusion/sort; excludes query encoding/index build/load',
        timing_threads=1,CPU_priority='nice19/ionice3; magnitude reference only',
        sources=dict(ranks=str(rankpath),candidate_counts=str(folder/(key+'_candidate_counts.npy')),
            candidate_hits=str(folder/(key+'_positive_in_candidates.npy')),query_seconds=str(folder/(key+'_query_seconds.npy'))),seconds=time.perf_counter()-t0)
    write(done,result);print(key,result['metrics'],flush=True);return result
