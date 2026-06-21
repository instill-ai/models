"""Production-shaped M5 link-pred service: node-pairs in, calibrated+abstaining scores out.

Unlike the toy server.py (which takes pre-computed feature vectors), the production service does feature
extraction SERVER-SIDE from a per-namespace graph context — so the Go M5 producer never has to reimplement
ELPH sketches / pooling / structural heuristics in Go. It just sends node-id pairs.

Deployment: host-managed MLX on the MacBook-Pro production fleet (same pattern as the qwen/gemma servers),
registered via buckle `start_mlx_linkpred`; the Go producer reaches it at CFG_..._LOCAL_LINKPRED_URL.

The graph CONTEXT (adjacency + per-node embeddings) is pluggable:
- production: load the namespace's tunnel edges (Postgres `memory_tunnel`) + pooled-qwen Room vectors (Milvus).
- offline/test: load an .npz (e.g. arxiv_sub.npz) via LINKPRED_GRAPH.
Refresh the context periodically as edges accrue (the graph is append-mostly).
"""
from __future__ import annotations
import asyncio, json, os
import numpy as np
import mlx.core as mx
from model import InductiveLinkScorer
from features import build_graphs, structural_features, embed_pair_features, standardize
from sketches import build_minhash, sketch_features


class LinkPredService:
    """Holds one namespace's graph context + the trained scorer; scores node-id pairs server-side."""

    def __init__(self, weights_dir, num_nodes, edges, node_emb=None, threshold=0.857):
        meta = json.load(open(os.path.join(weights_dir, "meta.json")))
        self.in_dim = meta["in_dim"]; self.use_sketch = meta.get("sketches", False)
        self.use_emb = meta.get("node_feat", False) and node_emb is not None
        self.hops, self.K = meta.get("hops", 2), meta.get("K", 128)
        self.threshold = threshold                                  # calibrated abstention cut (M1·W4 reuse)
        s = np.load(os.path.join(weights_dir, "scaler.npz")); self.mu, self.sd = s["mu"], s["sd"]
        self.model = InductiveLinkScorer(in_dim=self.in_dim); self.model.load_weights(os.path.join(weights_dir, "scorer.npz")); self.model.eval()
        self.node_emb = node_emb.astype(np.float32) if node_emb is not None else None

        ew = np.ones(edges.shape[0])
        self.Abin, self.Aew = build_graphs(num_nodes, edges.T, ew)  # message graph for structural+sketch feats
        rng = np.random.default_rng(0)
        self.mh = build_minhash(num_nodes, self.Abin, self.hops, self.K, rng) if self.use_sketch else None

    def _features(self, pairs):
        parts = [structural_features(self.Abin, self.Aew, pairs)]
        if self.mh is not None:
            parts.append(sketch_features(pairs, self.mh, self.hops))
        if self.use_emb:
            parts.append(embed_pair_features(self.node_emb, pairs))
        return np.hstack(parts) if len(parts) > 1 else parts[0]

    def score(self, pairs):
        """pairs: list/array of [u, v] node ids -> [{score, accept}]. Score = calibrated link probability."""
        pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        Fz, _, _ = standardize(self._features(pairs), self.mu, self.sd)
        prob = 1.0 / (1.0 + np.exp(-np.array(self.model(mx.array(Fz)))))   # sigmoid (scores already calibrated, ECE 0.02)
        return [{"u": int(u), "v": int(v), "score": float(p), "accept": bool(p >= self.threshold)}
                for (u, v), p in zip(pairs, prob)]


def load_context_from_npz(path):
    d = np.load(path)
    return int(d["n_nodes"]) if "n_nodes" in d else int(d["emb"].shape[0]), d["edges"], d.get("emb")


# --- module-scope service + app (mirrors server.py: FastAPI needs module-scope routes here) ---
WEIGHTS = os.environ.get("LINKPRED_WEIGHTS", "weights")
GRAPH = os.environ.get("LINKPRED_GRAPH", "arxiv_sub.npz")
THRESHOLD = float(os.environ.get("LINKPRED_THRESHOLD", "0.857"))
PORT = int(os.environ.get("LINKPRED_PORT", os.environ.get("PORT", "18450")))

_N, _edges, _emb = load_context_from_npz(GRAPH)
_svc = LinkPredService(WEIGHTS, _N, _edges, node_emb=_emb, threshold=THRESHOLD)
_lock = asyncio.Lock()

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    FastAPI = None

if FastAPI is not None:
    app = FastAPI(title="shubo-linkpred-mlx")

    class PredictReq(BaseModel):
        pairs: list[list[int]]

    @app.get("/health")
    def health():
        return {"status": "ok", "nodes": int(_N), "in_dim": _svc.in_dim, "sketches": _svc.use_sketch,
                "node_feat": _svc.use_emb, "threshold": _svc.threshold, "metal": mx.metal.is_available()}

    @app.post("/predict")
    async def predict(req: PredictReq):            # {"pairs": [[u,v], ...]} -> per-pair score + accept
        async with _lock:
            return {"results": _svc.score(req.pairs)}

    if __name__ == "__main__":
        uvicorn.run(app, host="0.0.0.0", port=PORT)
else:
    if __name__ == "__main__":
        raise SystemExit("install fastapi+uvicorn to serve; LinkPredService importable standalone")
