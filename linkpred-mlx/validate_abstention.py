"""Step-3: the production-SAFE operating point — calibration + abstention on the validated scorer.

Steps 1-2 showed the qwen+structure scorer works on representative data. "Production-ready" then hinges on
operating it SAFELY: surface only confident associations, abstain otherwise (the M1-W4 pattern). This proves
that concretely on the arxiv semantic graph — does Platt calibration give honest probabilities, and is there
a high-precision operating point at usable coverage?

    python validate_abstention.py --in arxiv_sub.npz
"""
from __future__ import annotations
import argparse, time
import numpy as np
import mlx.core as mx
import mlx.optimizers as optim
from sklearn.linear_model import LogisticRegression
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


def ece(probs, labels, bins=10):
    """Expected calibration error: |confidence − accuracy| averaged over probability bins."""
    e = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        m = (probs >= lo) & (probs < hi)
        if m.sum():
            e += abs(probs[m].mean() - labels[m].mean()) * m.mean()
    return float(e)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", default="arxiv_sub.npz")
    p.add_argument("--neg-pool", type=int, default=50000)
    p.add_argument("--emb-dim", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    rng = np.random.default_rng(a.seed); mx.random.seed(a.seed); t0 = time.time()

    d = np.load(a.inp); emb = d["emb"].astype(np.float32); edges = d["edges"]; N = int(d["n_nodes"])
    lo = np.minimum(edges[:, 0], edges[:, 1]); hi = np.maximum(edges[:, 0], edges[:, 1])
    und = np.unique(np.stack([lo, hi], 1)[lo != hi], axis=0); und = und[rng.permutation(und.shape[0])]
    nt, nv = und.shape[0] // 5, und.shape[0] // 10
    test_e, valid_e, train_e = und[:nt], und[nt:nt + nv], und[nt + nv:]
    known = set(map(tuple, und))
    te_neg = sample_neg(N, known, a.neg_pool, rng)
    va_neg = sample_neg(N, known, max(valid_e.shape[0] * 20, 5000), rng)
    tr_neg = sample_neg(N, known, train_e.shape[0], rng)

    Abin, Aew = build_graphs(N, train_e.T, np.ones(train_e.shape[0]))
    mh = build_minhash(N, Abin, 2, 128, rng)
    C = emb - emb.mean(0); _, _, Vt = np.linalg.svd(C, full_matrices=False)
    emb_h = (C @ Vt[:a.emb_dim].T).astype(np.float32)

    def feat(prs):
        cos = (emb[prs[:, 0]] * emb[prs[:, 1]]).sum(1, keepdims=True)
        return np.hstack([structural_features(Abin, Aew, prs), sketch_features(prs, mh, 2),
                          cos, emb_h[prs[:, 0]] * emb_h[prs[:, 1]]]).astype(np.float32)

    mu, sd = standardize(np.vstack([feat(train_e), feat(tr_neg)]))[1:]
    z = lambda prs: mx.array(standardize(feat(prs), mu, sd)[0])
    print(f"features built ({N} nodes, {train_e.shape[0]} train edges) ({time.time()-t0:.1f}s) — training scorer", flush=True)

    model = InductiveLinkScorer(in_dim=z(train_e[:1]).shape[1], dropout=0.5)
    opt = optim.AdamW(learning_rate=1e-3, weight_decay=1e-3)
    lvg = mx.value_and_grad(lambda m, xp, xn: infonce_loss(m, xp, xn, 16))
    Xp, Xn = z(train_e), z(tr_neg); npos, nneg = Xp.shape[0], Xn.shape[0]
    for ep in range(40):
        model.train(); perm = np.random.permutation(npos)
        for s in range(0, npos, 4096):
            ip = perm[s:s + 4096]; inn = np.random.randint(0, nneg, len(ip) * 16)
            _, g = lvg(model, Xp[mx.array(ip)], Xn[mx.array(inn)])
            opt.update(model, g); mx.eval(model.parameters(), opt.state)
    model.eval()

    def logit(prs): return np.array(model(z(prs)))
    # ---- calibration: Platt-scale raw logits on VALID, apply to TEST ----
    v_log = np.concatenate([logit(valid_e), logit(va_neg)]); v_lab = np.r_[np.ones(len(valid_e)), np.zeros(len(va_neg))]
    t_log = np.concatenate([logit(test_e), logit(te_neg)]); t_lab = np.r_[np.ones(len(test_e)), np.zeros(len(te_neg))]
    raw_p = 1 / (1 + np.exp(-t_log))
    platt = LogisticRegression(C=1e6).fit(v_log.reshape(-1, 1), v_lab)   # no rebalancing (preserve 1.3% prior)
    cal_p = platt.predict_proba(t_log.reshape(-1, 1))[:, 1]
    print(f"\n== production-safe operating point (arxiv, {len(test_e)} true edges vs {a.neg_pool} negs, base {t_lab.mean():.4f}) ==")
    print(f"  calibration ECE: raw {ece(raw_p, t_lab):.3f} -> Platt {ece(cal_p, t_lab):.3f}")

    # ---- precision @ coverage: rank by score; abstain below threshold ----
    order = np.argsort(-cal_p); lab_s = t_lab[order]; tp = np.cumsum(lab_s); total_pos = t_lab.sum()
    prec = tp / np.arange(1, len(lab_s) + 1); cov = tp / total_pos
    print("  abstain below threshold -> what precision at what recall/coverage:")
    for target in (0.95, 0.90, 0.80, 0.50):
        ok = np.where(prec >= target)[0]
        if len(ok):
            i = ok[-1]
            print(f"    precision >= {target:.2f}:  recall {cov[i]:.3f}  (accept top {i+1} of {len(lab_s)} candidates, p>={cal_p[order][i]:.3f})")
        else:
            print(f"    precision >= {target:.2f}:  not reachable")
    for k in (10, 50, 100):
        print(f"    precision@{k:<4d} (top scored): {prec[k-1]:.3f}")
    print(f"  ({time.time()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
