"""Step-2 prep: build a real SEMANTIC citation graph for proactive M5 validation.

ogbn-arxiv gives the citation STRUCTURE (papers citing papers) but only word2vec-128 node features. To test
the M5 bet faithfully we need the REAL embedder, so this re-embeds the papers' actual title+abstract with
qwen3-embedding:4b (via ollama) and caches everything (qwen embeddings + induced citation edges + category
labels) so the scorer harness can reuse it without paying the ~1s/doc embedding cost again.

A dense connected subgraph is taken by BFS from the highest-degree node (random subsets of 169K nodes have
almost no internal citations).

    python prep_arxiv_text.py --n 2000 --out arxiv_sub.npz
"""
from __future__ import annotations
import argparse, os, time, urllib.request
from collections import deque
import numpy as np
import pandas as pd
import requests
import scipy.sparse as sp

try:
    import torch
    _OL = torch.load
    torch.load = lambda *a, **k: _OL(*a, **{**k, "weights_only": False})
except ImportError:
    pass
from ogb.nodeproppred import NodePropPredDataset

ROOT = "/Users/Pinglin/.cache/ogb_datasets"
AR = f"{ROOT}/ogbn_arxiv"
TITLEABS = f"{AR}/titleabs.tsv.gz"
TITLEABS_URL = "https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz"
OLLAMA, MODEL = "http://localhost:11434", "qwen3-embedding:4b"


def qwen_embed(texts, batch=32):
    out = []
    for i in range(0, len(texts), batch):
        r = requests.post(f"{OLLAMA}/api/embed", json={"model": MODEL, "input": texts[i:i + batch]}, timeout=600)
        r.raise_for_status(); out.extend(r.json()["embeddings"])
        print(f"  embedded {min(i+batch, len(texts))}/{len(texts)}", end="\r", flush=True)
    print()
    E = np.asarray(out, dtype=np.float32)
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--out", default="arxiv_sub.npz")
    a = p.parse_args(argv)
    t0 = time.time()
    if os.path.exists(a.out):
        print(f"{a.out} already exists — skipping"); return 0

    g, labels = NodePropPredDataset(name="ogbn-arxiv", root=ROOT)[0]
    N = g["num_nodes"]; ei = g["edge_index"]; y = labels.ravel()
    A = sp.csr_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(N, N)); A = (A + A.T).tocsr()
    indptr, indices = A.indptr, A.indices
    seed = int(np.argmax(np.diff(indptr)))
    seen, q = {seed}, deque([seed])                          # BFS for a dense connected subgraph
    while q and len(seen) < a.n:
        u = q.popleft()
        for v in indices[indptr[u]:indptr[u + 1]]:
            v = int(v)
            if v not in seen:
                seen.add(v); q.append(v)
                if len(seen) >= a.n:
                    break
    nodes = np.array(sorted(seen)); remap = {int(o): i for i, o in enumerate(nodes)}
    nodeset = set(nodes.tolist())
    m = np.array([int(x) in nodeset and int(z) in nodeset for x, z in zip(ei[0], ei[1])])
    edges = np.array([[remap[int(x)], remap[int(z)]] for x, z in zip(ei[0][m], ei[1][m])])
    print(f"subgraph: {len(nodes)} nodes, {edges.shape[0]} internal citations ({time.time()-t0:.1f}s)", flush=True)

    if not os.path.exists(TITLEABS):
        print("downloading titleabs.tsv.gz ...", flush=True)
        urllib.request.urlretrieve(TITLEABS_URL, TITLEABS)
    n2p = pd.read_csv(f"{AR}/mapping/nodeidx2paperid.csv.gz")
    node2pid = dict(zip(n2p.iloc[:, 0].astype(int), n2p.iloc[:, 1].astype(int)))
    ta = pd.read_csv(TITLEABS, sep="\t", header=None, names=["pid", "title", "abs"])
    ta = ta[pd.to_numeric(ta.pid, errors="coerce").notna()]
    ta["pid"] = ta.pid.astype(int)
    pid2text = dict(zip(ta.pid, (ta.title.fillna("") + ". " + ta["abs"].fillna(""))))
    texts = [pid2text.get(node2pid.get(int(o), -1), "")[:2000] for o in nodes]
    miss = sum(1 for t in texts if not t.strip())
    print(f"text mapped ({len(texts)-miss}/{len(texts)} have abstracts) — embedding with {MODEL} ({time.time()-t0:.1f}s)", flush=True)

    E = qwen_embed(texts)
    np.savez(a.out, emb=E, edges=edges, y=y[nodes], n_nodes=len(nodes))
    print(f"saved {a.out}: emb {E.shape}, {edges.shape[0]} edges ({time.time()-t0:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
