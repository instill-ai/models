"""Step-2: does the full M5 scorer (qwen embeddings + structure) beat raw cosine on a REAL semantic graph?

Uses the arxiv citation subgraph from prep_arxiv_text.py (real qwen embeddings + real citation edges). On the
ogbl-collab proxy the scorer was only heuristic-tier because that graph had NO semantics; here we test on a
graph with BOTH a structural signal (citations) and a semantic one (qwen abstracts). Three scorers, one shared
split, OGB Hits@K:

  1. raw qwen cosine          — embedding similarity alone (the step-1 signal, applied to link prediction)
  2. structure-only MLP       — structural + ELPH-sketch features (the proxy approach, no semantics)
  3. full scorer (qwen+struct)— structural + sketch + Hadamard(qwen_u, qwen_v) → the deployable M5 model

If (3) > (1) and (3) > (2), the M5 algorithm earns its complexity on Shubo-like data.

    python eval_arxiv_semantic.py --in arxiv_sub.npz
"""
from __future__ import annotations
import argparse, time
import numpy as np
import mlx.core as mx
import mlx.optimizers as optim
from ogb.linkproppred import Evaluator
from model import InductiveLinkScorer, infonce_loss
from features import build_graphs, structural_features, standardize
from sketches import build_minhash, sketch_features


def sample_neg(N, known, n, rng):
    out = np.empty((n, 2), np.int64); got = 0
    while got < n:
        for x, y in rng.integers(0, N, (2 * (n - got), 2)):
            k = (x, y) if x < y else (y, x)
            if x != y and k not in known:
                out[got] = (x, y); got += 1
                if got >= n:
                    break
    return out


def hits(ev, pos, neg, ks):
    r = {}
    for k in ks:
        ev.K = k
        r[k] = float(ev.eval({"y_pred_pos": pos, "y_pred_neg": neg})[f"hits@{k}"])
    return r


def train_mlp(tr_pos, tr_neg, va_pos, va_neg, te_pos, te_neg, ev, ks, rng, epochs=60, k_neg=16):
    """InfoNCE-trained inductive scorer. Early-stop on VALID Hits@50 (no test leakage), report on TEST."""
    import os
    model = InductiveLinkScorer(in_dim=tr_pos.shape[1], dropout=0.5)
    opt = optim.AdamW(learning_rate=1e-3, weight_decay=1e-3)
    lvg = mx.value_and_grad(lambda m, xp, xn: infonce_loss(m, xp, xn, k_neg))
    Xp, Xn = mx.array(tr_pos), mx.array(tr_neg)
    np_pos, nneg = Xp.shape[0], Xn.shape[0]
    vp, vn = mx.array(va_pos), mx.array(va_neg)
    tp, tn = mx.array(te_pos), mx.array(te_neg)
    best, patience, ckpt = -1.0, 0, "/tmp/_arxiv_mlp_best.npz"
    for ep in range(1, epochs + 1):
        model.train(); perm = np.random.permutation(np_pos)
        for s in range(0, np_pos, 4096):
            ip = perm[s:s + 4096]
            inn = np.random.randint(0, nneg, size=len(ip) * k_neg)
            loss, grads = lvg(model, Xp[mx.array(ip)], Xn[mx.array(inn)])
            opt.update(model, grads); mx.eval(model.parameters(), opt.state)
        if ep % 3 == 0 or ep == epochs:
            model.eval(); ev.K = 50
            h = ev.eval({"y_pred_pos": np.array(model(vp)), "y_pred_neg": np.array(model(vn))})["hits@50"]
            if h > best:
                best, patience = h, 0; model.save_weights(ckpt)
            else:
                patience += 1
            if patience >= 5:
                break
    model.load_weights(ckpt); model.eval()
    return hits(ev, np.array(model(tp)), np.array(model(tn)), ks)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", default="arxiv_sub.npz")
    p.add_argument("--neg-pool", type=int, default=5000)
    p.add_argument("--emb-dim", type=int, default=64, help="PCA-reduce qwen embeddings before Hadamard (0=full 2560-d); reduces overfit on small graphs")
    p.add_argument("--ks", type=int, nargs="+", default=[10, 50, 100])
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    rng = np.random.default_rng(a.seed); mx.random.seed(a.seed); t0 = time.time()

    d = np.load(a.inp); emb = d["emb"].astype(np.float32); edges = d["edges"]; N = int(d["n_nodes"])
    lo = np.minimum(edges[:, 0], edges[:, 1]); hi = np.maximum(edges[:, 0], edges[:, 1])
    und = np.unique(np.stack([lo, hi], 1)[lo != hi], axis=0)
    und = und[rng.permutation(und.shape[0])]
    n_test = und.shape[0] // 5; n_val = und.shape[0] // 10
    test_e, valid_e, train_e = und[:n_test], und[n_test:n_test + n_val], und[n_test + n_val:]
    known = set(map(tuple, und))
    te_neg = sample_neg(N, known, a.neg_pool, rng)
    va_neg = sample_neg(N, known, a.neg_pool, rng)
    tr_neg = sample_neg(N, known, train_e.shape[0], rng)
    print(f"arxiv-sem: {N} nodes, {train_e.shape[0]} train / {valid_e.shape[0]} valid / {test_e.shape[0]} test citations, qwen {emb.shape[1]}-d, {a.neg_pool} negs ({time.time()-t0:.1f}s)", flush=True)

    Abin, Aew = build_graphs(N, train_e.T, np.ones(train_e.shape[0]))   # message graph = train edges
    mh = build_minhash(N, Abin, 2, 128, rng)
    ev = Evaluator(name="ogbl-collab")

    def struct(prs):
        return np.hstack([structural_features(Abin, Aew, prs), sketch_features(prs, mh, 2)])

    if a.emb_dim and a.emb_dim < emb.shape[1]:                          # PCA-reduce to curb overfit
        C = emb - emb.mean(0)
        _, _, Vt = np.linalg.svd(C, full_matrices=False)
        emb_h = (C @ Vt[:a.emb_dim].T).astype(np.float32)
    else:
        emb_h = emb

    def embfeat(prs):                                                   # cosine (robust 1-d) + reduced Hadamard
        cos = (emb[prs[:, 0]] * emb[prs[:, 1]]).sum(1, keepdims=True)
        had = emb_h[prs[:, 0]] * emb_h[prs[:, 1]]
        return np.hstack([cos, had]).astype(np.float32)

    # ---- 1. raw qwen cosine (normalized emb → dot = cosine) ----
    cos_pos = (emb[test_e[:, 0]] * emb[test_e[:, 1]]).sum(1)
    cos_neg = (emb[te_neg[:, 0]] * emb[te_neg[:, 1]]).sum(1)
    r_cos = hits(ev, cos_pos, cos_neg, a.ks)
    print(f"  [1] raw qwen cosine done ({time.time()-t0:.1f}s)", flush=True)

    def run(fn):   # featurize all splits with one scaler (from train), train+eval the MLP
        mu, sd = standardize(np.vstack([fn(train_e), fn(tr_neg)]))[1:]
        z = lambda prs: standardize(fn(prs), mu, sd)[0]
        return train_mlp(z(train_e), z(tr_neg), z(valid_e), z(va_neg), z(test_e), z(te_neg), ev, a.ks, rng)

    # ---- 2. structure-only MLP ----
    r_struct = run(struct)
    print(f"  [2] structure-only MLP done ({time.time()-t0:.1f}s)", flush=True)

    # ---- 3. full scorer: structure + cosine + PCA-reduced Hadamard(qwen) ----
    r_full = run(lambda prs: np.hstack([struct(prs), embfeat(prs)]))
    print(f"  [3] full scorer (qwen+struct, emb-dim={a.emb_dim}) done ({time.time()-t0:.1f}s)", flush=True)

    print(f"\n== M5 scorer on a REAL semantic citation graph (arxiv, {N} nodes, {a.neg_pool} negs) ==")
    hdr = "  " + " ".join(f"Hits@{k:<4d}" for k in a.ks)
    print(hdr)
    for name, r in [("raw qwen cosine ", r_cos), ("structure-only  ", r_struct), ("full (qwen+strct)", r_full)]:
        print(f"  {name} " + " ".join(f"{r[k]:.3f}    " for k in a.ks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
