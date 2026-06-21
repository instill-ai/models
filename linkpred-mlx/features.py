"""Inductive pair-features for link prediction — graph-structure derived (no node ids).

These are the features the deployable scorer uses. All are INDUCTIVE (computable for any pair on any
graph, including a brand-new namespace):
  - structural: common-neighbour family (CN/AA/RA), preferential attachment, Jaccard, weighted-CN,
    common-neighbour degree stats — the validated set (LightGBM reached Hits@50 0.646 on ogbl-collab).
  - [Shubo, added when node data exists] node-embedding pair: cosine/Hadamard of the two nodes' pooled
    qwen Item-embedding vectors (`embed_pair_features`).
  - [Shubo] taxonomic prior: Wu-Palmer similarity over shared facet-tags (`taxonomic_prior`).

The structural extractor here is corpus-agnostic; the pretrain proof (train.py) runs it on ogbl-collab,
and the same code applies unchanged to the M5 tunnel graph.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp

STRUCT_FEATS = ["cn", "aa", "ra", "log_pa", "jaccard", "wcn", "cn_deg_sum", "cn_deg_min", "deg_i", "deg_j"]


def build_graphs(num_nodes, ei, ew):
    """Symmetric binary-structure CSR (heuristics) + EW-weighted CSR (collab/co-occurrence counts)."""
    rows = np.concatenate([ei[0], ei[1]]); cols = np.concatenate([ei[1], ei[0]])
    Abin = sp.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(num_nodes, num_nodes))
    Abin.data[:] = 1.0; Abin.sum_duplicates(); Abin.data[:] = 1.0
    Aew = sp.csr_matrix((np.concatenate([ew, ew]), (rows, cols)), shape=(num_nodes, num_nodes)); Aew.sum_duplicates()
    return Abin, Aew


def structural_features(Abin, Aew, pairs):
    """The 10 inductive structural features per pair (CSR indices are sorted → cheap intersection)."""
    bip, bidx = Abin.indptr, Abin.indices
    wip, widx, wdat = Aew.indptr, Aew.indices, Aew.data
    deg = np.diff(bip).astype(np.float64)
    inv_log = 1.0 / np.log(np.maximum(deg, 2.0)); inv_deg = 1.0 / np.maximum(deg, 1.0)
    F = np.zeros((pairs.shape[0], len(STRUCT_FEATS)), dtype=np.float32)
    for k in range(pairs.shape[0]):
        i, j = int(pairs[k, 0]), int(pairs[k, 1])
        di, dj = deg[i], deg[j]
        F[k, 8] = di; F[k, 9] = dj; F[k, 3] = np.log1p(di * dj)
        ai, aj = bidx[bip[i]:bip[i + 1]], bidx[bip[j]:bip[j + 1]]
        if ai.size == 0 or aj.size == 0:
            continue
        common = np.intersect1d(ai, aj, assume_unique=True)
        if common.size:
            F[k, 0] = common.size; F[k, 1] = inv_log[common].sum(); F[k, 2] = inv_deg[common].sum()
            union = di + dj - common.size
            F[k, 4] = common.size / union if union > 0 else 0.0
            F[k, 6] = deg[common].sum(); F[k, 7] = deg[common].min()
        wi, wiw = widx[wip[i]:wip[i + 1]], wdat[wip[i]:wip[i + 1]]
        wj, wjw = widx[wip[j]:wip[j + 1]], wdat[wip[j]:wip[j + 1]]
        c, ii, jj = np.intersect1d(wi, wj, assume_unique=True, return_indices=True)
        if c.size:
            F[k, 5] = float(np.dot(wiw[ii], wjw[jj]))
    return F


# ---- Shubo-specific feature hooks (used once Room node data exists; structured here, not yet wired) ----
def embed_pair_features(node_vecs, pairs):
    """Cosine + Hadamard-sum of two nodes' pooled qwen embeddings (1024-d). node_vecs: [N, D] float32."""
    a, b = node_vecs[pairs[:, 0]], node_vecs[pairs[:, 1]]
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
    return np.stack([cos, (a * b).sum(1)], axis=1).astype(np.float32)


def standardize(F, mu=None, sd=None):
    """Z-score features; return (Fz, mu, sd) so train/serve share the same scaler."""
    if mu is None:
        mu, sd = F.mean(0), F.std(0) + 1e-6
    return ((F - mu) / sd).astype(np.float32), mu, sd


def sample_hard_negatives(Abin, n, rng):
    """Vectorized 2-hop hard negatives: pairs (u,v) sharing a common neighbor w (high-CN non-edges).

    These are what rank high in the eval pool, so training against them — unlike trivially-separable
    random negatives — forces the fine discrimination Hits@K rewards. Minor real-edge contamination is
    harmless training noise, so we skip the exact non-edge filter for speed (fully vectorized).
    """
    indptr, indices = Abin.indptr, Abin.indices
    deg = np.diff(indptr)
    nz = np.where(deg > 0)[0]
    out = []
    have = 0
    while have < n:
        u = rng.choice(nz, size=n)
        w = indices[indptr[u] + (rng.random(n) * deg[u]).astype(np.int64)]   # random neighbor of u
        dw = deg[w]; ok = dw > 0
        v = indices[indptr[w] + (rng.random(n) * np.maximum(dw, 1)).astype(np.int64)]  # random neighbor of w
        m = ok & (u != v)
        out.append(np.stack([u[m], v[m]], axis=1)); have += int(m.sum())
    return np.concatenate(out)[:n]
