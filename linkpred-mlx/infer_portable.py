"""Portable CPU inference — the PRODUCTION path (no MLX, no Metal, no GPU).

Why this exists: production is a Mac k3s fleet, but k3s pods are LINUX containers (k3s runs in a Linux VM on
macOS) and cannot access Metal — so MLX only runs on the macOS *host*, not in pods. The deployable scorer is a
tiny MLP, so the clean prod path is a normal containerized CPU service doing pure-numpy inference. (Local dev
keeps the host-managed MLX server via buckle; prod uses this.)

`PortableScorer` reproduces `InductiveLinkScorer.__call__` exactly in numpy — Linear -> LayerNorm -> ReLU per
layer, then the head, then sigmoid. Validated bit-close to the MLX model below.
"""
from __future__ import annotations
import json, os
import numpy as np


class PortableScorer:
    """Pure-numpy MLP inference from the saved npz weights. Runs anywhere (Linux pod, CPU)."""

    def __init__(self, weights_dir, eps=1e-5):
        self.meta = json.load(open(os.path.join(weights_dir, "meta.json")))
        self.in_dim = self.meta["in_dim"]; self.eps = eps
        w = np.load(os.path.join(weights_dir, "scorer.npz"))
        self.W = {k: w[k].astype(np.float64) for k in w.files}
        self.depth = sum(1 for k in self.W if k.startswith("layers.") and k.endswith(".weight"))
        s = np.load(os.path.join(weights_dir, "scaler.npz"))
        self.mu, self.sd = s["mu"].astype(np.float64), s["sd"].astype(np.float64)

    def _forward(self, x):
        for i in range(self.depth):                                  # Linear -> LayerNorm -> ReLU
            x = x @ self.W[f"layers.{i}.weight"].T + self.W[f"layers.{i}.bias"]
            mu = x.mean(-1, keepdims=True); var = x.var(-1, keepdims=True)
            x = (x - mu) / np.sqrt(var + self.eps) * self.W[f"norms.{i}.weight"] + self.W[f"norms.{i}.bias"]
            x = np.maximum(x, 0.0)
        return (x @ self.W["head.weight"].T + self.W["head.bias"]).squeeze(-1)  # logit

    def score(self, features, calibrated=True):
        """features: (N, in_dim) RAW pair features. Returns link probabilities (sigmoid of logit)."""
        x = (np.asarray(features, dtype=np.float64) - self.mu) / self.sd
        logit = self._forward(x)
        return 1.0 / (1.0 + np.exp(-logit)) if calibrated else logit


if __name__ == "__main__":
    # validate numpy inference matches the MLX model bit-close
    import sys
    wd = sys.argv[1] if len(sys.argv) > 1 else "weights"
    ps = PortableScorer(wd)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((256, ps.in_dim)).astype(np.float32) * ps.sd + ps.mu  # raw-feature scale
    p_np = ps.score(X)
    try:
        import mlx.core as mx
        from model import InductiveLinkScorer
        m = InductiveLinkScorer(in_dim=ps.in_dim); m.load_weights(os.path.join(wd, "scorer.npz")); m.eval()
        Xz = ((X - ps.mu) / ps.sd).astype(np.float32)
        p_mlx = 1.0 / (1.0 + np.exp(-np.array(m(mx.array(Xz)))))
        diff = float(np.max(np.abs(p_np - p_mlx)))
        print(f"portable-numpy vs MLX: max|Δprob| = {diff:.2e} over {X.shape[0]} rows  ->  {'MATCH' if diff < 1e-4 else 'MISMATCH'}")
        print(f"  sample probs: numpy {p_np[:3].round(5)}  mlx {p_mlx[:3].round(5)}")
    except ImportError:
        print(f"MLX not available; portable scorer ran standalone on {X.shape[0]} rows, mean prob {p_np.mean():.4f}")
