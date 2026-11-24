"""Numerical implementation used by the reported experiment."""
import numpy as np
import hashlib
from scipy.stats import spearmanr
from scipy.linalg import svdvals
import torch
K=np.array([1,5,10,100])
REPS=2000
SEED=20261023


def unit(x):
    x=np.asarray(x,dtype=np.float64)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)


def spectrum(raw,temporal=True):
    x=unit(raw);s=svdvals(x,check_finite=False)
    if s.sum()==0:
        erank=0.;pc90=pc95=0
    else:
        p=s/s.sum();p=p[p>0];erank=float(np.exp(-(p*np.log(p)).sum()))
        energy=np.cumsum(s*s)/np.sum(s*s)
        pc90=int(np.searchsorted(energy,.9)+1);pc95=int(np.searchsorted(energy,.95)+1)
    adjacent=float(np.einsum('ij,ij->i',x[:-1],x[1:]).mean()) if len(x)>1 and temporal else float('nan')
    return dict(effective_rank=erank,pc90=pc90,pc95=pc95,adjacent_cosine=adjacent)


def _rank_plan(values):
    order=np.argsort(values,kind='stable');ordered=values[order]
    starts=np.r_[0,np.flatnonzero(ordered[1:]!=ordered[:-1])+1]
    inverse=np.empty(len(order),dtype=np.int64)
    inverse[order]=np.repeat(np.arange(len(starts)),np.diff(np.r_[starts,len(order)]))
    return order,starts,inverse


def weighted_centered_ranks(counts,plan):
    order,starts,inverse=plan
    weights=np.add.reduceat(counts[:,order],starts,axis=1)
    ranks=np.cumsum(weights,axis=1)-(weights-1.)/2.
    return ranks[:,inverse]-(counts.sum(1,keepdims=True)+1.)/2.


def correlations(counts,rx,ry,rz):
    vz=(counts*rz*rz).sum(1)
    bx=np.divide((counts*rx*rz).sum(1),vz,out=np.zeros(len(counts)),where=vz>0)
    by=np.divide((counts*ry*rz).sum(1),vz,out=np.zeros(len(counts)),where=vz>0)
    xx=rx-bx[:,None]*rz;yy=ry-by[:,None]*rz
    denominator=np.sqrt((counts*xx*xx).sum(1)*(counts*yy*yy).sum(1))
    return np.divide((counts*xx*yy).sum(1),denominator,out=np.full(len(counts),np.nan),where=denominator>0)


def rank_scores(scores,truth):
    scores=np.asarray(scores);truth=np.asarray(truth)
    positive=scores[np.arange(len(truth)),truth]
    rank=(scores>positive[:,None]).sum(1)+1
    tied=np.flatnonzero((scores==positive[:,None]).sum(1)>1)
    if len(tied):
        order=torch.argsort(-torch.from_numpy(np.ascontiguousarray(scores[tied])),dim=1,stable=False).numpy()
        rank[tied]=np.argmax(order==truth[tied,None],axis=1)+1
    return rank.astype(np.int32),len(tied)


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def hits(ranks):
    return np.count_nonzero(np.asarray(ranks)[:, None] <= K, axis=1)


def metrics(ranks):
    a = np.asarray(ranks)
    out = {f'R@{k}':float(100*np.mean(a <= k)) for k in K}
    return dict(**out, SumR=sum(out.values()))


def estimate(point, draws, **meta):
    a = np.asarray(draws); valid = np.isfinite(a); finite = a[valid]
    assert len(finite), 'No finite bootstrap draws'
    lo, hi = np.quantile(finite, [.025, .975], method='linear')
    flag = 'detectable positive' if lo > 0 else 'detectable negative' if hi < 0 else 'no detectable difference'
    return dict(point=float(point), ci_low=float(lo), ci_high=float(hi), conclusion=flag,
                valid_bootstrap=int(valid.sum()), undefined_bootstrap=int((~valid).sum()), **meta)


class Plan:
    def __init__(self, video_ids, truth, folder):
        self.labels, self.inv = np.unique(np.asarray(video_ids)[truth], return_inverse=True)
        self.n = len(self.labels)
        self.counts = np.bincount(self.inv, minlength=self.n)
        rng = np.random.default_rng(SEED)
        weight = np.empty((REPS, self.n), dtype=np.int16)
        first = None
        for s in range(0, REPS, 32):
            b = min(32, REPS-s); indices = rng.integers(0, self.n, size=(b, self.n))
            weight[s:s+b] = np.bincount((indices + np.arange(b)[:, None]*self.n).ravel(),
                                       minlength=b*self.n).reshape(b, self.n)
            if first is None:
                first = indices[0].copy()
        self.weight = weight
        self.first = np.concatenate([np.flatnonzero(self.inv == i) for i in first])
        self.meta = dict(replicates=REPS, seed=SEED, unit='positive video',
                         video_order='lexicographic video ID', n_video=self.n,
                         weights_sha256=digest(weight), same_draws_all_comparisons=True,
                         group_boundaries_fixed=True, group_means='conditional on sampled group queries')
        np.savez_compressed(folder / 'bootstrap_weights.npz', weight=weight, video_ids=self.labels)

    def means(self, values, query_mask):
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        selected = np.asarray(query_mask, dtype=bool)
        counts = np.bincount(self.inv[selected], minlength=self.n)
        totals = np.zeros((self.n, values.shape[1]))
        np.add.at(totals, self.inv[selected], values[selected])
        den = self.weight @ counts
        draws = np.divide(100*(self.weight @ totals), den[:, None],
                          out=np.full((REPS, values.shape[1]), np.nan), where=den[:, None] > 0)
        first = self.first[selected[self.first]]
        assert np.max(np.abs(draws[0] - 100*values[first].mean(0))) < 1e-10
        return 100*values[selected].mean(0), draws


def rank_arrays(clip, frame, ward, truth, label):
    nq, nv = clip.shape
    assert frame.shape == ward.shape == (nq, nv)
    names = ['original', 'F', 'O', 'C', 'original_frame', 'F_frame']
    ranks = {x:np.empty(nq, dtype=np.int32) for x in names}
    ties = {x:0 for x in names}
    for s in range(0, nq, 128):
        if s % 4096 == 0:
            pass
        e = min(s+128, nq); gt = truth[s:e]; ix = np.arange(e-s)
        cs = np.asarray(clip[s:e]); fs = np.asarray(frame[s:e]); ws = np.asarray(ward[s:e])
        base = np.float32(.7)*cs + np.float32(.3)*fs
        comp = np.float32(.7)*cs + np.float32(.3)*ws
        assert np.isfinite(base).all() and np.isfinite(comp).all()
        own = base.copy(); own[ix, gt] = comp[ix, gt]
        other = comp.copy(); other[ix, gt] = base[ix, gt]
        for name, matrix in zip(names, (base, comp, own, other, fs, ws)):
            a, b = rank_scores(matrix, gt)
            ranks[name][s:e] = a; ties[name] += b
    return ranks, ties


def contribution(ranks):
    baseline = hits(ranks['original'])
    a = np.column_stack([hits(ranks[k])-baseline for k in ['F', 'O', 'C']])
    return np.column_stack([a, a[:, 0]-a[:, 1]-a[:, 2]])


def grouped(data, plan, reference_cuts=None, complete=False):
    er = data['er']; truth = data['truth']; v = data['video_ids']; ranks = data['ranks']
    cuts = np.quantile(er, [.25, .5, .75], method='linear') if reference_cuts is None else np.asarray(reference_cuts)
    bins = np.searchsorted(cuts, er, side='left')
    values = contribution(ranks)
    names = ['F', 'O', 'C', 'residual'] if complete else ['F']
    values = values[:, :len(names)]
    result = {}; all_draws = {}
    for q in [-1, 0, 1, 2, 3]:
        name = 'all' if q == -1 else f'Q{q+1}'
        vm = np.ones(len(v), dtype=bool) if q == -1 else bins == q
        qm = vm[truth]
        point, draws = plan.means(values, qm)
        meta = dict(n_query=int(qm.sum()), n_video=int(np.unique(truth[qm]).size),
                    group_total_videos=int(vm.sum()), bootstrap_weights_sha256=plan.meta['weights_sha256'])
        result[name] = dict(**meta, full=metrics(ranks['original'][qm]), compressed=metrics(ranks['F'][qm]),
                            decomposition={key:estimate(point[j], draws[:, j], **meta) for j, key in enumerate(names)})
        result[name]['delta_SumR'] = result[name]['decomposition']['F']
        all_draws[name] = draws
    difference = all_draws['Q1'][:, 0]-all_draws['Q4'][:, 0]
    contrast = estimate(result['Q1']['delta_SumR']['point']-result['Q4']['delta_SumR']['point'], difference,
                        n_query=result['Q1']['n_query']+result['Q4']['n_query'],
                        n_video=result['Q1']['n_video']+result['Q4']['n_video'],
                        bootstrap_weights_sha256=plan.meta['weights_sha256'])
    out = dict(cuts=cuts.tolist(), groups=result, Q1_minus_Q4=contrast, bootstrap=plan.meta)
    return out, all_draws


def losses(data):
    truth = data['truth']; nv = len(data['video_ids'])
    count = np.bincount(truth, minlength=nv)
    out = {}
    for name, before, after in [('fusion', 'original', 'F'), ('frame_only', 'original_frame', 'F_frame')]:
        delta = data['ranks'][after].astype(float)-data['ranks'][before]
        out[name] = np.divide(np.bincount(truth, weights=delta, minlength=nv), count,
                              out=np.full(nv, np.nan), where=count > 0)
    return out


def point_statistics(data):
    loss = losses(data); er = data['er']; length = data['length']
    result = {}
    for branch, y in loss.items():
        valid = np.isfinite(er) & np.isfinite(y) & np.isfinite(length)
        x, yy, z = er[valid], y[valid], length[valid]
        weight = np.ones((1, len(x)))
        rr = [weighted_centered_ranks(weight, _rank_plan(t)) for t in (x, yy, z)]
        result[branch] = dict(rho=float(spearmanr(x, yy).statistic),
                              partial_rho=float(correlations(weight, *rr)[0]), n_video=len(x))
    return result


def correlation_bootstrap(data, plan, partial):
    video_lookup = {x:i for i, x in enumerate(data['video_ids'])}
    order = np.asarray([video_lookup[x] for x in plan.labels])
    er = data['er'][order]; length = data['length'][order]
    loss = {k:v[order] for k, v in losses(data).items()}
    result = {}
    for branch, y in loss.items():
        pass
        valid = np.isfinite(er) & np.isfinite(y) & np.isfinite(length)
        n = int(valid.sum()); x, yy, z = er[valid], y[valid], length[valid]
        if valid.all():
            weights = plan.weight
        else:
            rng = np.random.default_rng(SEED); indices = rng.integers(0, n, size=(REPS, n))
            weights = np.bincount((indices + np.arange(REPS)[:, None]*n).ravel(), minlength=REPS*n).reshape(REPS, n).astype(np.int16)
        px, py, pz = [_rank_plan(a) for a in (x, yy, z)]
        draws = []; pd = []
        for s in range(0, REPS, 32):
            w = weights[s:s+32].astype(float)
            rx, ry = weighted_centered_ranks(w, px), weighted_centered_ranks(w, py)
            den = np.sqrt((w*rx*rx).sum(1)*(w*ry*ry).sum(1))
            draws.extend(np.divide((w*rx*ry).sum(1), den, out=np.full(len(w), np.nan), where=den > 0))
            if partial:
                rz = weighted_centered_ranks(w, pz); pd.extend(correlations(w, rx, ry, rz))
        meta = dict(n_video=n, excluded_no_finite_loss=int((~valid).sum()),
                    weights_sha256=digest(weights), same_cell_query_comparisons_plan=bool(valid.all()),
                    recompute_tied_ranks_every_draw=True)
        point = float(spearmanr(x, yy).statistic)
        row = dict(spearman=estimate(point, draws, **meta))
        if partial:
            w = np.ones((1, n)); rr = [weighted_centered_ranks(w, p) for p in (px, py, pz)]
            point = float(correlations(w, *rr)[0])
            row['partial'] = estimate(point, pd, **meta, control='average tied rank of L; intercept included')
        # Exact first-draw rank check against explicitly repeated videos.
        indices = np.repeat(np.arange(n), weights[0])
        assert abs(draws[0]-spearmanr(x[indices], yy[indices]).statistic) < 1e-10
        if partial:
            expanded = [__import__('scipy').stats.rankdata(a[indices]) for a in (x, yy, z)]
            aa = [a-a.mean() for a in expanded]
            expected = correlations(np.ones((1, len(indices))), *[a[None] for a in aa])[0]
            assert abs(pd[0]-expected) < 1e-10
        row['first_expanded_draw_verified'] = True
        result[branch] = row
    return result
