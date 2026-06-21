"""Inductive link-prediction scorer in MLX (Apple-Silicon GPU).

The DEPLOYABLE association scorer for Shubo's dynamic-organization graph (M5 tunnels). It is
**inductive**: it scores a candidate pair purely from PAIR FEATURES (structural sketches + pooled
node embeddings + taxonomic prior), never from a learned per-node id — so it generalizes zero-shot to
a brand-new (greenfield) namespace, unlike transductive PLNLP/GIDN. Design: backend
`docs/agent/dynamic-organization-m5.md` §"Cold-start deployment".

Tiny by construction (an MLP over ~10-150 features) → trains in seconds, hosts trivially via MLX.
"""
from __future__ import annotations
import mlx.core as mx
import mlx.nn as nn


class InductiveLinkScorer(nn.Module):
    """MLP: pair-feature vector -> link logit. Inductive (no node embedding table)."""

    def __init__(self, in_dim: int, hidden: int = 128, depth: int = 2, dropout: float = 0.3):
        super().__init__()
        dims = [in_dim] + [hidden] * depth
        self.layers = [nn.Linear(dims[i], dims[i + 1]) for i in range(depth)]
        self.norms = [nn.LayerNorm(hidden) for _ in range(depth)]
        self.head = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x):
        for lin, norm in zip(self.layers, self.norms):
            x = self.dropout(nn.relu(norm(lin(x))))
        return self.head(x).squeeze(-1)  # logit


def bce_loss(model, x, y):
    logits = model(x)
    # numerically-stable BCE-with-logits
    return mx.mean(mx.maximum(logits, 0) - logits * y + mx.log1p(mx.exp(-mx.abs(logits))))


def bpr_loss(model, x_pos, x_neg):
    """Pairwise BPR ranking loss = mean −log σ(s_pos − s_neg).

    Optimizes RANKING (Hits@K) directly rather than pointwise calibration — the same lever that took
    the GBDT from a pointwise fit to LambdaMART 0.646. softplus(−d) is the stable −log σ(d).
    """
    d = model(x_pos) - model(x_neg)
    return mx.mean(mx.maximum(-d, 0) + mx.log1p(mx.exp(-mx.abs(d))))


def infonce_loss(model, x_pos, x_neg, k):
    """Sampled-softmax / InfoNCE: rank each positive above K negatives at once.

    Listwise-flavoured (vs BPR's single pair) → closer to LambdaMART; larger K = harder, better Hits@K.
    x_pos:(B,F), x_neg:(B*K,F). loss = mean(logsumexp[s_pos, s_neg_1..K] − s_pos).
    """
    sp = model(x_pos)                                  # (B,)
    sn = model(x_neg).reshape(sp.shape[0], k)          # (B, K)
    z = mx.concatenate([sp[:, None], sn], axis=1)      # (B, 1+K)
    return mx.mean(mx.logsumexp(z, axis=1) - sp)
