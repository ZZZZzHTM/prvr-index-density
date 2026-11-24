"""Numerical implementation used by the reported experiment."""
import numpy as np
from torch.nn import functional as F
from pathlib import Path
from dataclasses import asdict
from dataclasses import dataclass
import json
import time
import torch
DTYPES={'fp32':np.float32,'fp16':np.float16,'int8':np.int8}
REPEATS=5
WARMUP=3
BLOCK=1024
K=100


@dataclass(frozen=True)
class Term:
    video: str
    query: str
    weight: float


@dataclass(frozen=True)
class Spec:
    method: str
    positions: dict[str, int]
    terms: tuple[Term, ...]
    mean_fusion: bool = False

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, obj):
        return cls(obj['method'], obj['positions'], tuple(Term(**x) for x in obj['terms']),
                   obj.get('mean_fusion', False))


def pack(value, mode):
    """Input has already been L2 normalized. One fp32 scale per vector for int8."""
    if mode not in DTYPES:
        raise ValueError(mode)
    if mode != 'int8':
        return value.numpy().astype(DTYPES[mode]), None
    maximum = value.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(maximum > 0, maximum / 127., torch.ones_like(maximum))
    quantized = (value / scale).round().clamp(-127, 127).to(torch.int8)
    return quantized.numpy(), scale.squeeze(-1).numpy()


class Index:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.manifest = json.loads((self.folder/'manifest.json').read_text())
        if self.manifest['version'] != 1:
            raise ValueError('Unsupported index version')
        self.spec = Spec.from_dict(self.manifest['spec'])
        self.mode = self.manifest['mode']
        self.video_ids, self.query_ids = self.manifest['video_ids'], self.manifest['query_ids']
        self.count = len(self.video_ids)
        self.values = {name: self._load(name) for name in self.spec.positions}
        self.lengths = {name: self._load(name+'_lengths') for name in self.spec.positions}
        self.scales = {name: self._load(name+'_scales') for name in self.spec.positions} if self.mode == 'int8' else {}
        self.queries = {name: self._load('query_'+name) for name in dict.fromkeys(t.query for t in self.spec.terms)}
        d = self.manifest['dimension']
        for name, positions in self.spec.positions.items():
            if self.values[name].shape != (self.count, positions, d) or self.values[name].dtype != DTYPES[self.mode]:
                raise ValueError('Corrupt video array')
            if self.lengths[name].shape != (self.count,) or self.lengths[name].dtype != np.int32:
                raise ValueError('Corrupt length array')
            if np.any(self.lengths[name] < 1) or np.any(self.lengths[name] > positions):
                raise ValueError('Invalid lengths')
            if self.mode == 'int8' and (self.scales[name].shape != (self.count, positions) or self.scales[name].dtype != np.float32):
                raise ValueError('Corrupt scale array')
        if any(x.shape != (len(self.query_ids), d) or x.dtype != np.float32 for x in self.queries.values()):
            raise ValueError('Corrupt query array')

    def _load(self, name):
        return np.load(self.folder/(name+'.npy'), mmap_mode='r', allow_pickle=False)

    def read(self, start, end, device='cpu'):
        if not 0 <= start < end <= self.count:
            raise ValueError('Invalid video slice')
        values, lengths = {}, {}
        for name in self.spec.positions:
            # Copy only this block. fp16/int8 transfer occurs BEFORE fp32 conversion.
            value = torch.from_numpy(np.array(self.values[name][start:end], copy=True)).to(device)
            value = value.float()
            if self.mode == 'int8':
                scale = torch.from_numpy(np.array(self.scales[name][start:end], copy=True)).to(device)
                value *= scale.unsqueeze(-1)
            values[name] = value
            lengths[name] = torch.from_numpy(np.array(self.lengths[name][start:end], copy=True)).to(device)
        return values, lengths

    def query(self, start, end, device='cpu'):
        return {name: torch.from_numpy(np.array(value[start:end], copy=True)).to(device)
                for name, value in self.queries.items()}

    def size_report(self, per_video=False):
        d = self.manifest['dimension']
        vector_bytes = d*np.dtype(DTYPES[self.mode]).itemsize + (4 if self.mode == 'int8' else 0)
        padded = np.zeros(self.count, dtype=np.int64)
        valid = padded.copy()
        for name, positions in self.spec.positions.items():
            padded += positions*vector_bytes + 4  # one int32 length per physical branch
            valid += np.asarray(self.lengths[name], dtype=np.int64)*vector_bytes + 4
        def stats(value):
            result = dict(total_bytes=int(value.sum()), bytes_per_video_mean=float(value.mean()),
                          bytes_per_video_min=int(value.min()), bytes_per_video_max=int(value.max()))
            if per_video:
                result['per_video_bytes'] = value.tolist()
            return result
        arrays = [self.folder/(name+'.npy') for name in self.spec.positions]
        arrays += [self.folder/(name+'_lengths.npy') for name in self.spec.positions]
        arrays += [self.folder/(name+'_scales.npy') for name in self.scales]
        return dict(padded=stats(padded), valid_only=stats(valid), mode=self.mode,
                    scale_bytes_per_vector=4 if self.mode == 'int8' else 0,
                    length_bytes_per_video=4*len(self.spec.positions),
                    physical_video_array_file_bytes=sum(x.stat().st_size for x in arrays),
                    all_index_files_bytes=sum(p.stat().st_size for p in self.folder.iterdir() if p.is_file()),
                    query_payload_bytes=sum(x.nbytes for x in self.queries.values()),
                    valid_only_is_hypothetical_compaction=True)


def score_block(queries, values, lengths, spec):
    """Inputs are normalized before storage. Storage errors are NOT renormalized away."""
    scores = []
    for term in spec.terms:
        context, query = values[term.video], queries[term.query]
        # Exactly the original matmul orientation; temporary is only [block, P, Q].
        position_scores = torch.matmul(context, query.t())
        valid = torch.arange(context.shape[1], device=context.device)[None, :] < lengths[term.video][:, None]
        position_scores.masked_fill_(~valid.unsqueeze(-1), -torch.inf)
        branch_scores = position_scores.max(dim=1).values
        if spec.method == 'B0' and term.video == 'frame':
            # Original GMMFormerV2 includes zero padding in the 128-position max.
            # Use the original valid length, so compact valid-only storage agrees.
            branch_scores = torch.where(lengths[term.video][:, None] < 128,
                                        branch_scores.clamp_min(0), branch_scores)
        scores.append(branch_scores.t())
        del position_scores
    if spec.mean_fusion:
        # Original Lite implementation: sum then divide, not weighted pre-sums.
        return sum(scores)/len(scores)
    result = scores[0]*spec.terms[0].weight
    for term, value in zip(spec.terms[1:], scores[1:]):
        result = result + value*term.weight
    return result


def select_topk(scores, indices, k):
    """Descending score, ties by original video index ascending, independent of blocks."""
    if indices.ndim == 1:
        indices = indices.expand(scores.shape[0], -1)
    by_index = torch.argsort(indices, dim=1, stable=True)
    indices, scores = indices.gather(1, by_index), scores.gather(1, by_index)
    order = torch.argsort(-scores, dim=1, stable=True)[:, :k]
    return scores.gather(1, order), indices.gather(1, order)


def sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def memory_begin(device):
    if torch.device(device).type != 'cuda':
        return None
    sync(device)
    torch.cuda.reset_peak_memory_stats(device)
    return dict(baseline_allocated_bytes=torch.cuda.memory_allocated(device),
                baseline_reserved_bytes=torch.cuda.memory_reserved(device))


def memory_end(device, baseline):
    if baseline is None:
        return dict(device=str(device), gpu_peak_allocated_bytes=None, gpu_peak_reserved_bytes=None)
    sync(device)
    allocated = torch.cuda.max_memory_allocated(device)
    return dict(device=str(device), **baseline, gpu_peak_allocated_bytes=allocated,
                gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                gpu_peak_increment_bytes=allocated-baseline['baseline_allocated_bytes'])


class RamIndex(Index):
    """Keep Index.read and streaming_topk unchanged; replace mmap by RAM arrays."""
    def __init__(self, spec, values, lengths, queries, mode):
        self.spec, self.mode = spec, mode
        self.count = len(next(iter(values.values())))
        self.manifest = dict(dimension=next(iter(values.values())).shape[-1])
        self.values, self.lengths, self.scales, self.queries = {}, {}, {}, queries
        for name, value in values.items():
            packed, scales = pack(value, mode)
            self.values[name] = packed
            self.lengths[name] = lengths[name].numpy().copy()
            if scales is not None:
                self.scales[name] = scales

    def bytes_report(self):
        d = self.manifest['dimension']
        per_vector = d*np.dtype(DTYPES[self.mode]).itemsize + (4 if self.mode == 'int8' else 0)
        valid = sum(int(np.asarray(lens, dtype=np.int64).sum()) for lens in self.lengths.values())
        padded = sum(self.spec.positions.values())
        return dict(padded_bytes_per_video=padded*per_vector+4*len(self.spec.positions),
                    valid_bytes_per_video=valid*per_vector/self.count+4*len(self.spec.positions),
                    padded_total_bytes=self.count*(padded*per_vector+4*len(self.spec.positions)),
                    valid_total_bytes=valid*per_vector+4*len(self.spec.positions)*self.count,
                    dimension=d, scale_bytes_per_vector=4 if self.mode == 'int8' else 0,
                    length_bytes_per_video=4*len(self.spec.positions),
                    valid_only_is_hypothetical_compaction=True)


class GpuReplica:
    """Physical padded GPU copies; identical decode, scores, stable top-k/merge.

    Only the index transport differs from Index.read. fp16/int8 are decoded per
    block in fp32 without renormalization. Never return virtual/modulo views in
    retrieval: all target videos have allocated, independently copied storage.
    """
    def __init__(self, source, count, device):
        self.spec, self.mode, self.count = source.spec, source.mode, count
        self.values, self.lengths, self.scales = {}, {}, {}
        self.device = device
        for target, origin in ((self.values, source.values), (self.lengths, source.lengths),
                               (self.scales, source.scales)):
            for name, array in origin.items():
                base = torch.from_numpy(array).to(device)
                value = torch.empty((count, *array.shape[1:]), dtype=base.dtype, device=device)
                for start in range(0, count, BLOCK):
                    end = min(start+BLOCK, count)
                    offsets = torch.arange(start, end, device=device) % source.count
                    value[start:end].copy_(base.index_select(0, offsets))
                target[name] = value
                del base, offsets
        sync(device)

    def read(self, start, end, device='cpu'):
        if str(device) != str(self.device):
            raise ValueError('Replica must be read on its allocation device')
        values, lengths = {}, {}
        for name in self.spec.positions:
            value = self.values[name][start:end].float()
            if self.mode == 'int8':
                value *= self.scales[name][start:end].unsqueeze(-1)
            values[name], lengths[name] = value, self.lengths[name][start:end]
        return values, lengths


def stats(seconds, items=1):
    seconds = np.asarray(seconds, dtype=np.float64)
    if seconds.shape != (REPEATS,) or np.any(seconds <= 0):
        raise ValueError('Exactly five positive timing repeats required')
    q1, median, q3 = np.percentile(seconds, [25, 50, 75], method='linear')
    throughput = items/seconds
    t1, tm, t3 = np.percentile(throughput, [25, 50, 75], method='linear')
    return dict(raw_seconds=seconds.tolist(), q1_ms=float(q1*1000), median_ms=float(median*1000),
                q3_ms=float(q3*1000), iqr_ms=float((q3-q1)*1000),
                median_ms_per_item=float(median*1000/items),
                throughput_items_per_second=dict(median=float(tm), q1=float(t1), q3=float(t3), iqr=float(t3-t1)))


@torch.inference_mode()
def measure(fn, device, items=1):
    for _ in range(WARMUP):
        result = fn()
        del result
    sync(device)
    times, peaks = [], []
    for _ in range(REPEATS):
        baseline = memory_begin(device)
        started = time.perf_counter()
        result = fn()
        sync(device)
        times.append(time.perf_counter()-started)
        peaks.append(memory_end(device, baseline))
        del result
    return dict(**stats(times, items), repeats=REPEATS, warmup=WARMUP,
                memory_samples=peaks,
                peak_allocated_bytes=max(p['gpu_peak_allocated_bytes'] for p in peaks),
                peak_reserved_bytes=max(p['gpu_peak_reserved_bytes'] for p in peaks))


@torch.inference_mode()
def encode(models,features,mask,normalize=False):
    outputs=[]
    for model in models:
        q=model.encode_query(features,mask).reshape(len(features),-1)
        outputs.append({'query':F.normalize(q.float(),dim=-1) if normalize else q})
    return outputs


@torch.inference_mode()
def ensemble_search(indexes,queries,block_size=1024,k=100,device='cuda:0'):
    assert indexes and len(indexes)==len(queries) and len({v.count for v in indexes})==1
    best_scores=best_indices=None
    for start in range(0,indexes[0].count,block_size):
        end=min(start+block_size,indexes[0].count);total=None
        for index,q in zip(indexes,queries):
            values,lengths=index.read(start,end,device)
            fused=score_block(q,values,lengths,index.spec)
            total=fused if total is None else total+fused
            del values,lengths,fused
        # Match analysis_math.simple_mean: left-to-right fp32 sum, then fp32 reciprocal.
        if len(indexes)>1:total=total*float(np.float32(1./len(indexes)))
        scores,indices=select_topk(total,torch.arange(start,end,device=device),k)
        if best_scores is not None:
            scores,indices=select_topk(torch.cat((best_scores,scores),dim=1),torch.cat((best_indices,indices),dim=1),k)
        best_scores,best_indices=scores,indices
        del total,scores,indices
    return best_scores,best_indices


def measure_session(model, features, query_mask, clip, full_frame, full_lengths, compact_frame, compact_lengths, device='cuda:0'):
    import gc
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    model=model.eval().requires_grad_(False).to(device)
    features=features.to(device)
    query_mask=query_mask.to(device)
    length=int(query_mask[0].sum())
    inputs={1:(features[:1,:length],query_mask[:1,:length]),128:(features,query_mask)}
    if len(features)!=128:
        raise ValueError('This measurement expects exactly 128 query features.')
    queries={batch:encode([model],*value,normalize=True) for batch,value in inputs.items()}
    measured={}
    for name,context,lengths,mode in (('full_fp32',full_frame,full_lengths,'fp32'),('Ward8_fp32',compact_frame,compact_lengths,'fp32'),('Ward8_int8',compact_frame,compact_lengths,'int8')):
        values={'clip':clip,'frame':context}
        counts={'clip':torch.full((len(clip),),32,dtype=torch.int32),'frame':lengths}
        spec=Spec(name,{k:v.shape[1] for k,v in values.items()},(Term('clip','query',.7),Term('frame','query',.3)))
        source=RamIndex(spec,values,counts,{},mode)
        torch.cuda.empty_cache()
        before=torch.cuda.memory_allocated()
        replica=GpuReplica(source,100000,device)
        allocated=sum(v.numel()*v.element_size() for g in (replica.values,replica.lengths,replica.scales) for v in g.values())
        if allocated!=100000*source.bytes_report()['padded_bytes_per_video']:
            raise AssertionError('Unexpected physical storage')
        row={'resident_bytes':allocated,'resident_increment':torch.cuda.memory_allocated()-before,'loads':{}}
        for batch in (1,128):
            calls={'query_encoding':lambda:encode([model],*inputs[batch],normalize=False),
                   'scoring_retrieval':lambda replica=replica:ensemble_search([replica],queries[batch]),
                   'end_to_end':lambda replica=replica:ensemble_search([replica],encode([model],*inputs[batch],normalize=True))}
            row['loads'][str(batch)]={stage:measure(call,device,batch) for stage,call in calls.items()}
        measured[name]=row
        del calls
        replica = None
        source = None
        gc.collect()
        torch.cuda.empty_cache()
    return measured
