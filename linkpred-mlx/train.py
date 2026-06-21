"""Transfer-pretrain the inductive MLX link-scorer on a public corpus (ogbl-collab proxy).

This is the day-one model: train ONCE on a public graph using only inductive structural features, save
weights + scaler, then serve zero-shot on any graph (a greenfield Shubo namespace) via server.py.
ogbl-collab is the available proxy; SciDocs/S2ORC co-citation (the intent-matched scholarly corpus) is
the production pretrain target — swap the loader, the model/features are unchanged.

    python train.py --epochs 60 --out weights
"""
from __future__ import annotations
import argparse, sys, time
import numpy as np
try:
    import mlx.core as mx
    import mlx.optimizers as optim
    from ogb.linkproppred import LinkPropPredDataset, Evaluator
except ImportError as e:
    sys.exit(f"needs mlx + ogb ({e})")
from model import InductiveLinkScorer, bce_loss, bpr_loss, infonce_loss
from features import build_graphs, structural_features, embed_pair_features, standardize, sample_hard_negatives, STRUCT_FEATS
from sketches import build_minhash, sketch_features, SKETCH_FEATS
from datasets import load_corpus


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/Users/Pinglin/.cache/ogb_datasets")
    p.add_argument("--dataset", choices=["ogbl-collab", "ogbn-arxiv"], default="ogbl-collab",
                   help="pretrain corpus: ogbl-collab (collaboration proxy) or ogbn-arxiv (scholarly citations)")
    p.add_argument("--train-pos", type=int, default=300000)
    p.add_argument("--neg-pool", type=int, default=100000, help="negative pool size for built splits (ogbn-arxiv); matches ogbl-collab's 100k for fair Hits@K")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=8192)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-3, help="AdamW weight decay (regularize vs overfit)")
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--loss", choices=["bce", "bpr", "infonce"], default="infonce", help="bce (pointwise), bpr (pairwise), or infonce (K-negative contrastive — strongest for Hits@K)")
    p.add_argument("--neg-k", type=int, default=16, help="negatives per positive for infonce")
    p.add_argument("--hard-neg-frac", type=float, default=0.0, help="fraction of training negatives that are 2-hop hard negatives (high common-neighbor non-edges)")
    p.add_argument("--eval-steps", type=int, default=3, help="eval valid every N epochs for early stopping")
    p.add_argument("--patience", type=int, default=5, help="early-stop after N evals w/o valid improvement")
    p.add_argument("--eval", choices=["valid", "test"], default="test")
    p.add_argument("--node-feat", action="store_true", help="add node-embedding pair features (cos+hadamard) — the Shubo pooled-qwen feature shape, here using ogbl-collab's 128-d node_feat as stand-in")
    p.add_argument("--sketches", action=argparse.BooleanOptionalAction, default=True, help="ELPH/BUDDY multi-hop subgraph sketches (the BUDDY route; on by default)")
    p.add_argument("--K", type=int, default=128, help="MinHash sketch width")
    p.add_argument("--hops", type=int, default=2, help="sketch propagation hops")
    p.add_argument("--out", default="weights")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    rng = np.random.default_rng(a.seed); mx.random.seed(a.seed); t0 = time.time()

    g, split = load_corpus(a.dataset, a.root, rng, neg_pool=a.neg_pool)
    N = int(g["num_nodes"]); ei = g["edge_index"]; ew = g["edge_weight"].astype(np.float64).ravel()
    tr, va = split["train"]["edge"], split["valid"]["edge"]
    Abin, Aew = build_graphs(N, ei, ew)  # train-only message graph
    if a.eval == "test":
        ev_pos, ev_neg = split["test"]["edge"], split["test"]["edge_neg"]
        tei = np.concatenate([ei, va.T], axis=1); tew = np.concatenate([ew, np.ones(va.shape[0])])
        Abin_e, Aew_e = build_graphs(N, tei, tew)
    else:
        ev_pos, ev_neg = va, split["valid"]["edge_neg"]; Abin_e, Aew_e = Abin, Aew
    print(f"{a.dataset}: {N} nodes, {tr.shape[0]} train / {va.shape[0]} valid / {ev_pos.shape[0]} eval edges ({time.time()-t0:.1f}s)", flush=True)

    n_pos = min(a.train_pos, tr.shape[0])
    pos = tr[rng.choice(tr.shape[0], n_pos, replace=False)]
    known = set((min(x, y), max(x, y)) for E in (tr, va, ev_pos) for x, y in E)
    neg = np.empty((n_pos, 2), np.int64); got = 0
    while got < n_pos:
        for x, y in rng.integers(0, N, (2 * (n_pos - got), 2)):
            if x != y and (min(x, y), max(x, y)) not in known:
                neg[got] = (x, y); got += 1
                if got >= n_pos:
                    break
    if a.hard_neg_frac > 0:   # replace a fraction of random negs with 2-hop hard negatives
        n_hard = int(n_pos * a.hard_neg_frac)
        neg[:n_hard] = sample_hard_negatives(Abin, n_hard, rng)
        print(f"hard negatives: {n_hard}/{n_pos} are 2-hop ({time.time()-t0:.1f}s)", flush=True)
    nf = g["node_feat"].astype(np.float32) if (a.node_feat and g.get("node_feat") is not None) else None
    mh_tr = build_minhash(N, Abin, a.hops, a.K, rng) if a.sketches else None
    mh_ev = (mh_tr if a.eval == "valid" else build_minhash(N, Abin_e, a.hops, a.K, rng)) if a.sketches else None
    if a.sketches:
        print(f"ELPH sketches built (K={a.K}, hops={a.hops}) ({time.time()-t0:.1f}s)", flush=True)

    def feats(Ab, Ae, mh, prs):
        parts = [structural_features(Ab, Ae, prs)]
        if mh is not None:
            parts.append(sketch_features(prs, mh, a.hops))     # BUDDY/ELPH multi-hop bucket counts
        if nf is not None:
            parts.append(embed_pair_features(nf, prs))         # node-embedding pair (Shubo pooled-qwen shape)
        return np.hstack(parts) if len(parts) > 1 else parts[0]

    Xp = feats(Abin, Aew, mh_tr, pos); Xn = feats(Abin, Aew, mh_tr, neg)
    X = np.vstack([Xp, Xn]); y = np.concatenate([np.ones(n_pos), np.zeros(n_pos)]).astype(np.float32)
    Xz, mu, sd = standardize(X)
    print(f"features done ({X.shape[1]} feats) ({time.time()-t0:.1f}s) — training MLX on metal", flush=True)

    import os, json
    os.makedirs(a.out, exist_ok=True)
    best_path = os.path.join(a.out, "scorer.npz")
    model = InductiveLinkScorer(in_dim=X.shape[1], dropout=a.dropout)
    opt = optim.AdamW(learning_rate=a.lr, weight_decay=a.wd)
    ev = Evaluator(name="ogbl-collab")

    if a.loss == "infonce":   # K-negative contrastive (sampled softmax) — strongest ranking objective
        Xzp, Xzn = mx.array(Xz[:n_pos]), mx.array(Xz[n_pos:])
        lvg = mx.value_and_grad(lambda m, xp, xn: infonce_loss(m, xp, xn, a.neg_k))
        def train_epoch():
            model.train(); pp = np.random.permutation(n_pos); tot = nb = 0
            for s in range(0, n_pos, a.batch):
                ip = pp[s:s + a.batch]
                inn = np.random.randint(0, n_pos, size=len(ip) * a.neg_k)   # K random negs / positive
                loss, grads = lvg(model, Xzp[mx.array(ip)], Xzn[mx.array(inn)])
                opt.update(model, grads); mx.eval(model.parameters(), opt.state)
                tot += loss.item(); nb += 1
            return tot / nb
    elif a.loss == "bpr":   # pairwise ranking — pos rows vs (independently shuffled) neg rows
        Xzp, Xzn = mx.array(Xz[:n_pos]), mx.array(Xz[n_pos:])
        lvg = mx.value_and_grad(lambda m, xp, xn: bpr_loss(m, xp, xn))
        def train_epoch():
            model.train(); pp, pn = np.random.permutation(n_pos), np.random.permutation(n_pos)
            tot = nb = 0
            for s in range(0, n_pos, a.batch):
                ip, inn = mx.array(pp[s:s + a.batch]), mx.array(pn[s:s + a.batch])
                loss, grads = lvg(model, Xzp[ip], Xzn[inn])
                opt.update(model, grads); mx.eval(model.parameters(), opt.state)
                tot += loss.item(); nb += 1
            return tot / nb
    else:                 # pointwise BCE
        Xt, yt = mx.array(Xz), mx.array(y); nrows = Xt.shape[0]
        lvg = mx.value_and_grad(lambda m, xb, yb: bce_loss(m, xb, yb))
        def train_epoch():
            model.train(); perm = np.random.permutation(nrows); tot = nb = 0
            for s in range(0, nrows, a.batch):
                idx = mx.array(perm[s:s + a.batch])
                loss, grads = lvg(model, Xt[idx], yt[idx])
                opt.update(model, grads); mx.eval(model.parameters(), opt.state)
                tot += loss.item(); nb += 1
            return tot / nb

    # early-stopping signal: VALID split scored on the TRAIN message graph (no leakage into test)
    va_neg = split["valid"]["edge_neg"]
    Xvp = mx.array(standardize(feats(Abin, Aew, mh_tr, va), mu, sd)[0])
    Xvn = mx.array(standardize(feats(Abin, Aew, mh_tr, va_neg), mu, sd)[0])
    def valid_h50():
        model.eval(); ev.K = 50
        r = ev.eval({"y_pred_pos": np.array(model(Xvp)), "y_pred_neg": np.array(model(Xvn))})["hits@50"]
        model.train(); return float(r)

    best_va, patience = -1.0, 0
    for ep in range(1, a.epochs + 1):
        avg = train_epoch()
        if ep % a.eval_steps == 0 or ep == a.epochs:
            vh = valid_h50()
            if vh > best_va:
                best_va, patience = vh, 0; model.save_weights(best_path)   # checkpoint the best
            else:
                patience += 1
            print(f"  ep {ep:3d}  loss {avg:.4f}  valid_h50 {vh:.4f}  (best {best_va:.4f})  ({time.time()-t0:.1f}s)", flush=True)
            if patience >= a.patience:
                print(f"  early stop @ep{ep} (no valid gain in {a.patience} evals)", flush=True); break

    model.load_weights(best_path); model.eval()   # restore best-valid checkpoint
    def score(prs):
        return np.array(model(mx.array(standardize(feats(Abin_e, Aew_e, mh_ev, prs), mu, sd)[0])))
    res = {}
    for k in (10, 50, 100):
        ev.K = k
        res[k] = float(ev.eval({"y_pred_pos": score(ev_pos), "y_pred_neg": score(ev_neg)})[f"hits@{k}"])
    print(f"\n== inductive MLX scorer — {a.dataset} {a.eval}: Hits@10 {res[10]:.4f}  Hits@50 {res[50]:.4f}  Hits@100 {res[100]:.4f}  (best valid {best_va:.4f}) ==")
    print("  ref: our LightGBM 0.646 | BUDDY 0.659 | PLNLP 0.706 (transductive, NOT deployable)")

    np.savez(os.path.join(a.out, "scaler.npz"), mu=mu, sd=sd)
    feat_names = list(STRUCT_FEATS) + (SKETCH_FEATS if a.sketches else []) + (["emb_cos", "emb_dot"] if nf is not None else [])
    json.dump({"in_dim": int(X.shape[1]), "feats": feat_names, "sketches": a.sketches, "node_feat": nf is not None,
               "K": a.K, "hops": a.hops, "loss": a.loss, "hits@50": res[50], "pretrain": a.dataset,
               "model": "BUDDY-style inductive (ELPH sketches + structural + node-emb) → MLX MLP"},
              open(os.path.join(a.out, "meta.json"), "w"), indent=2)
    print(f"  saved weights+scaler+meta to {a.out}/ ({time.time()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
