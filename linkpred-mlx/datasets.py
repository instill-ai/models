"""Corpus loaders for the inductive link-scorer pretrain.

Both return the SAME (g, split) shape the OGB link-pred API uses, so train.py stays corpus-agnostic:
  g     = {num_nodes, edge_index (2×E train message edges), edge_weight, node_feat}
  split = {train:{edge}, valid:{edge,edge_neg}, test:{edge,edge_neg}}

- `ogbl-collab` — collaboration graph (domain MISMATCH for M5's significance prior; the available proxy).
- `ogbn-arxiv` — arXiv paper CITATIONS (169K nodes, ~1.16M edges, 128-d title/abstract embeddings). The
  intent-matched SCHOLARLY domain (papers citing papers ≈ Rooms associating), tractable on a laptop — the
  feasible stand-in for the SciDocs/S2ORC co-citation production target. Built here as a random
  85/5/10 link-prediction split (no official one), scored against a sampled negative pool.
"""
from __future__ import annotations
import numpy as np

# OGB pickles its processed datasets; torch>=2.6 defaults torch.load(weights_only=True) and rejects them.
# OGB data is trusted, so restore the permissive load.
try:
    import torch
    _ORIG_TORCH_LOAD = torch.load
    torch.load = lambda *a, **k: _ORIG_TORCH_LOAD(*a, **{**k, "weights_only": False})
except ImportError:
    pass


def _sample_negatives(N, known, n, rng):
    out = np.empty((n, 2), np.int64); got = 0
    while got < n:
        for x, y in rng.integers(0, N, size=(2 * (n - got), 2)):
            if x == y:
                continue
            k = (x, y) if x < y else (y, x)
            if k in known:
                continue
            out[got] = (x, y); got += 1
            if got >= n:
                break
    return out


def load_corpus(name, root, rng, neg_pool=50000):
    if name == "ogbl-collab":
        from ogb.linkproppred import LinkPropPredDataset
        ds = LinkPropPredDataset(name=name, root=root)
        return ds[0], ds.get_edge_split()

    if name == "ogbn-arxiv":
        from ogb.nodeproppred import NodePropPredDataset
        g0 = NodePropPredDataset(name=name, root=root)[0][0]
        N = int(g0["num_nodes"]); ei = g0["edge_index"]; nf = g0["node_feat"]
        lo = np.minimum(ei[0], ei[1]); hi = np.maximum(ei[0], ei[1])
        m = lo != hi
        und = np.unique(np.stack([lo[m], hi[m]], axis=1), axis=0)        # unique undirected edges
        und = und[rng.permutation(und.shape[0])]
        n = und.shape[0]; n_test = n // 10; n_val = n // 20
        test_e, valid_e, train_e = und[:n_test], und[n_test:n_test + n_val], und[n_test + n_val:]
        known = set(map(tuple, und))                                     # all real edges (min,max)
        g = {"num_nodes": N, "edge_index": train_e.T,                    # build_graphs symmetrizes
             "edge_weight": np.ones(train_e.shape[0]), "node_feat": nf}
        split = {"train": {"edge": train_e},
                 "valid": {"edge": valid_e, "edge_neg": _sample_negatives(N, known, neg_pool, rng)},
                 "test":  {"edge": test_e,  "edge_neg": _sample_negatives(N, known, neg_pool, rng)}}
        return g, split

    raise SystemExit(f"unknown corpus: {name}")
