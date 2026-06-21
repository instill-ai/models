"""Proactive M5 validation on REPRESENTATIVE data — step 1: does the real qwen embedder recover association?

No production namespace exists yet (greenfield), so we validate the cold-start association algorithm in
advance on a real labeled corpus, embedded with the REAL Shubo embedder (qwen3-embedding:4b via ollama).

The core M5 bet is that *semantic* embeddings make associations good (on the structural ogbl-collab proxy
the scorer was only heuristic-tier — but that proxy had no semantics). This is the cheapest, sharpest test
of that bet: embed real documents, and for each one ask whether its nearest neighbours by qwen cosine share
its topic. If precision@K >> base rate, the semantic signal M5 relies on is real. If not, the approach needs
rework before any production wiring.

    python validate_semantic.py --n 2000 --k 10
"""
from __future__ import annotations
import argparse, sys, time
import numpy as np
import requests

OLLAMA = "http://localhost:11434"
MODEL = "qwen3-embedding:4b"


def qwen_embed(texts, batch=64):
    """Embed texts with the real Shubo embedder via ollama. Returns (N, D) L2-normalized float32."""
    out = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        r = requests.post(f"{OLLAMA}/api/embed", json={"model": MODEL, "input": chunk}, timeout=600)
        r.raise_for_status()
        out.extend(r.json()["embeddings"])
        print(f"  embedded {min(i+batch, len(texts))}/{len(texts)}", end="\r", flush=True)
    print()
    E = np.asarray(out, dtype=np.float32)
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)


def precision_at_k(E, labels, ks):
    """For each item, rank others by cosine; fraction of top-K sharing its label. Mean over items."""
    S = E @ E.T
    np.fill_diagonal(S, -np.inf)            # exclude self
    order = np.argsort(-S, axis=1)          # descending similarity
    lab = np.asarray(labels)
    res = {}
    for k in ks:
        topk = order[:, :k]
        same = (lab[topk] == lab[:, None])
        res[k] = float(same.mean())
    return res


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2000, help="documents to sample")
    p.add_argument("--k", type=int, nargs="+", default=[1, 5, 10, 20])
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    rng = np.random.default_rng(a.seed); t0 = time.time()

    from sklearn.datasets import fetch_20newsgroups
    data = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
    docs, y, names = data.data, np.array(data.target), data.target_names
    keep = np.array([len(d.strip()) > 80 for d in docs])      # drop near-empty posts
    docs = [d for d, k in zip(docs, keep) if k]; y = y[keep]
    idx = rng.choice(len(docs), min(a.n, len(docs)), replace=False)
    docs = [docs[i][:2000] for i in idx]; y = y[idx]          # truncate long posts
    nclass = len(np.unique(y))
    print(f"20-Newsgroups: {len(docs)} docs, {nclass} topics — embedding with {MODEL} ({time.time()-t0:.1f}s)", flush=True)

    E = qwen_embed(docs)
    print(f"embedded ({E.shape[1]}-d) ({time.time()-t0:.1f}s) — measuring association recovery", flush=True)

    res = precision_at_k(E, y, a.k)
    # base rate = expected same-label fraction of a random other item
    _, counts = np.unique(y, return_counts=True)
    base = float(((counts * (counts - 1)).sum()) / (len(y) * (len(y) - 1)))

    print(f"\n== qwen semantic association on real text (20NG, {len(docs)} docs, {nclass} topics) ==")
    print(f"  random base rate (same-topic): {base:.3f}")
    for k in a.k:
        print(f"  precision@{k:<3d}: {res[k]:.3f}   ({res[k]/base:.1f}x base)")
    verdict = "STRONG — semantic signal is real, the M5 bet holds" if res[a.k[-1]] > 3 * base else \
              "WEAK — qwen alone barely beats base rate; the approach needs rework"
    print(f"  verdict: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
