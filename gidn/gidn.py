#!/usr/bin/env python3
"""GIDN (Graph Inception Diffusion Network, arXiv:2210.01301) — from-paper reimplementation.

No official GIDN code exists. GIDN = AGDN (Adaptive Graph Diffusion Networks, hop-wise attention,
arXiv:2012.15024) + an Inception module + multi-feature-space diffusion; it reaches Hits@50 0.7096 on
ogbl-collab (AGDN 0.6635). This reimplements it in PURE PyTorch with native scatter (index_add_) so the
graph diffusion runs on Apple-Silicon MPS — sidestepping torch_scatter/torch_sparse (CPU-only, which
made PLNLP ~40h on CPU). AGDN collab config grounds the hyperparameters (K=2, 256 hidden, learned
embeddings / no node feat, hop-wise attention, MLP predictor, CE loss, val-edges-as-input, year 2010).

    python gidn_repro.py --epochs 800
"""
from __future__ import annotations
import argparse, time, sys, os, json
import numpy as np
try:
    import torch, torch.nn as nn, torch.nn.functional as F
    from ogb.linkproppred import LinkPropPredDataset, Evaluator
except ImportError as e:
    sys.exit(f"needs torch + ogb ({e})")


def sym_norm_adj(num_nodes, ei, device):
    """Symmetric-normalised adjacency with self-loops as (src,dst,val) tensors on `device`."""
    src = np.concatenate([ei[0], ei[1], np.arange(num_nodes)])
    dst = np.concatenate([ei[1], ei[0], np.arange(num_nodes)])
    deg = np.bincount(dst, minlength=num_nodes).astype(np.float64)
    dinv = 1.0 / np.sqrt(np.maximum(deg, 1.0))
    val = dinv[src] * dinv[dst]
    return (torch.tensor(src, device=device, dtype=torch.long),
            torch.tensor(dst, device=device, dtype=torch.long),
            torch.tensor(val, device=device, dtype=torch.float32))


def diffuse(H, adj, K):
    """Return [H, ÂH, Â²H, ..., Â^K H] — multi-hop diffusion via native scatter (MPS-friendly)."""
    src, dst, val = adj
    outs = [H]
    cur = H
    for _ in range(K):
        msg = cur[src] * val.unsqueeze(1)
        agg = torch.zeros_like(cur)
        agg.index_add_(0, dst, msg)
        cur = agg
        outs.append(cur)
    return outs  # K+1 tensors


class GIDN(nn.Module):
    """Learned embeddings → ONE diffusion → {hop-wise attention (AGDN) ‖ inception branches} → repr.

    Speed-critical: the multi-hop diffusion (MPS index_add scatter, the bottleneck) is computed ONCE
    per forward and its hop-stack [H0..HK] is shared by BOTH the AGDN hop-attention combination and the
    inception branches — vs the naive per-layer + inception re-diffusion (~3× the scatters)."""
    def __init__(self, num_nodes, dim, K, n_layers, dropout, n_branches=3, node_feat=None):
        super().__init__()
        self.emb = nn.Embedding(num_nodes, dim)
        self.K = K; self.dropout = dropout
        self.input_lin = nn.Linear(dim, dim)                       # project learned embeddings
        # GIDN "different feature spaces": real OGB node features give immediate signal (no warmup) and
        # combine with the learned embedding — without this, full-batch training (1 step/epoch) can't
        # move a from-scratch 235k×256 embedding table fast enough.
        self.feat_lin = nn.Linear(node_feat.shape[1], dim) if node_feat is not None else None
        if node_feat is not None:
            self.register_buffer("nf", node_feat)
        self.hop_att = nn.Parameter(torch.ones(K + 1) / (K + 1))   # AGDN hop-wise attention
        self.branch = nn.ModuleList(nn.Linear(dim * (K + 1), dim) for _ in range(n_branches))
        self.out = nn.Linear(dim * (n_branches + 1), dim)          # inception branches + AGDN-combined
        self.bn = nn.BatchNorm1d(dim)
        nn.init.xavier_uniform_(self.emb.weight)

    def forward(self, adj):
        h = self.input_lin(self.emb.weight)
        if self.feat_lin is not None:
            h = h + self.feat_lin(self.nf)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, self.training)
        hops = diffuse(h, adj, self.K)                             # the ONLY diffusion: K+1 × [N,dim]
        a = torch.softmax(self.hop_att, 0)
        h_agdn = sum(a[k] * hops[k] for k in range(self.K + 1))    # AGDN hop-wise attention
        cat = torch.cat(hops, dim=1)                               # inception input [N, dim*(K+1)]
        branches = [F.relu(b(F.dropout(cat, self.dropout, self.training))) for b in self.branch]
        z = self.out(torch.cat(branches + [h_agdn], dim=1))        # fuse inception + AGDN
        return self.bn(z)


class Predictor(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.l1 = nn.Linear(dim, dim); self.l2 = nn.Linear(dim, 1); self.dropout = dropout
    def forward(self, zi, zj):
        x = zi * zj                                        # Hadamard
        x = F.dropout(F.relu(self.l1(x)), self.dropout, self.training)
        return self.l2(x).squeeze(-1)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/tmp/ogb_datasets")
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--K", type=int, default=2)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--wd", type=float, default=5e-4, help="weight decay (regularise against overfit)")
    p.add_argument("--no-node-feat", action="store_true", help="ablate OGB node features (learned emb only)")
    p.add_argument("--batch-edges", type=int, default=0, help="0=full-batch (1 step/epoch); >0=minibatch edges/step (more gradient steps → learned embeddings actually train)")
    p.add_argument("--batch", type=int, default=65536)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--year", type=int, default=2010)
    p.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    p.add_argument("--out-dir", default="", help="dir to write best.pt + rolling result.json")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = a.device if a.device != "auto" else ("mps" if torch.backends.mps.is_available() else "cpu")
    if dev == "cpu":
        torch.set_num_threads(10)
    t0 = time.time()

    ds = LinkPropPredDataset(name="ogbl-collab", root=a.root); g = ds[0]; split = ds.get_edge_split()
    N = int(g["num_nodes"]); ei = g["edge_index"]; eyear = g["edge_year"].ravel()
    tr = split["train"]["edge"]; va = split["valid"]["edge"]
    # message graph = train edges (year>=filter) + val edges (use_valedges_as_input)
    yr_mask = eyear >= a.year
    msg = np.concatenate([ei[:, yr_mask], va.T], axis=1)
    adj = sym_norm_adj(N, msg, dev)
    # supervision positives = train edges with year >= filter (collab convention; tr aligns with ei/eyear)
    pos = torch.tensor(tr[yr_mask] if tr.shape[0] == yr_mask.shape[0] else tr, dtype=torch.long, device=dev)
    nf = None
    if g.get("node_feat") is not None and not a.no_node_feat:
        nf = torch.tensor(g["node_feat"], dtype=torch.float32, device=dev)
    print(f"ogbl-collab: {N} nodes / msg edges {msg.shape[1]} / train pos {pos.shape[0]} / "
          f"node_feat {None if nf is None else tuple(nf.shape)} / dev {dev} ({time.time()-t0:.1f}s)", flush=True)

    model = GIDN(N, a.dim, a.K, a.layers, a.dropout, node_feat=nf).to(dev)
    pred = Predictor(a.dim, a.dropout).to(dev)
    opt = torch.optim.Adam(list(model.parameters()) + list(pred.parameters()), lr=a.lr, weight_decay=a.wd)
    evaluator = Evaluator(name="ogbl-collab")
    va_pos = torch.tensor(va, dtype=torch.long, device=dev)
    va_neg = torch.tensor(split["valid"]["edge_neg"], dtype=torch.long, device=dev)
    te_pos = torch.tensor(split["test"]["edge"], dtype=torch.long, device=dev)
    te_neg = torch.tensor(split["test"]["edge_neg"], dtype=torch.long, device=dev)

    def evaluate(z, pe, ne):
        model.eval(); pred.eval()
        with torch.no_grad():
            sp_ = torch.cat([pred(z[pe[i:i+a.batch, 0]], z[pe[i:i+a.batch, 1]]) for i in range(0, pe.shape[0], a.batch)])
            sn_ = torch.cat([pred(z[ne[i:i+a.batch, 0]], z[ne[i:i+a.batch, 1]]) for i in range(0, ne.shape[0], a.batch)])
        evaluator.K = 50
        return evaluator.eval({"y_pred_pos": sp_.cpu().numpy(), "y_pred_neg": sn_.cpu().numpy()})["hits@50"]

    def step(pe):
        ne = torch.randint(0, N, (pe.shape[0], 2), device=dev)       # strict global negatives
        opt.zero_grad()
        z = model(adj)                                               # full-graph forward (recompute → embeddings learn)
        po = pred(z[pe[:, 0]], z[pe[:, 1]]); no = pred(z[ne[:, 0]], z[ne[:, 1]])
        loss = F.binary_cross_entropy_with_logits(po, torch.ones_like(po)) + \
               F.binary_cross_entropy_with_logits(no, torch.zeros_like(no))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(pred.parameters()), 1.0)
        opt.step(); return loss.item()

    best_va, best_te, best_ep = 0, 0, 0
    for ep in range(1, a.epochs + 1):
        model.train(); pred.train()
        if a.batch_edges > 0:    # MINIBATCH: many gradient steps/epoch (embeddings need them) — the fix
            perm = torch.randperm(pos.shape[0], device=dev)          # for full-batch's undertraining
            tot = sum(step(pos[perm[s:s + a.batch_edges]]) for s in range(0, pos.shape[0], a.batch_edges))
        else:                    # FULL-BATCH: 1 step/epoch (fast, but undertrains the learned embeddings)
            tot = step(pos)
        if ep % a.eval_every == 0 or ep == a.epochs:
            z = model(adj)
            va_h = evaluate(z, va_pos, va_neg); te_h = evaluate(z, te_pos, te_neg)
            if va_h > best_va:
                best_va, best_te, best_ep = va_h, te_h, ep
                if a.out_dir:
                    torch.save({"model": model.state_dict(), "pred": pred.state_dict()},
                               os.path.join(a.out_dir, "best.pt"))
            print(f"ep {ep:4d}  loss {tot:.3f}  valid {va_h:.4f}  test {te_h:.4f}  (best_va {best_va:.4f} @ep{best_ep} -> test {best_te:.4f})  {time.time()-t0:.0f}s", flush=True)
            if a.out_dir:    # rolling result file so a monitor / the user can read progress any time
                with open(os.path.join(a.out_dir, "result.json"), "w") as f:
                    json.dump({"epoch": ep, "epochs_total": a.epochs, "valid_now": va_h, "test_now": te_h,
                               "best_valid": best_va, "best_test": best_te, "best_epoch": best_ep,
                               "elapsed_s": round(time.time() - t0, 1), "device": dev,
                               "config": {k: getattr(a, k) for k in ("K", "dim", "layers", "dropout", "lr", "year")}}, f, indent=2)
    print(f"\n== GIDN reimpl — ogbl-collab Hits@50: test {best_te:.4f} @ best valid {best_va:.4f} (ep {best_ep}) ==")
    print(f"  ref: AGDN 0.6635 | BUDDY 0.656 | PLNLP 0.706 | GIDN(paper) 0.7096")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
