"""MLX link-predictor serving — the hostable /predict service.

Mirrors Shubo's local-model server pattern (buckle/scripts/sandbox/flux-image-server.py): the MLX model
is loaded once at import, inference is guarded by an asyncio lock (MLX runs one job at a time), and it
exposes GET /health + POST /predict. Launch the same way (a `start_mlx_linkpred` in start-local-models.sh
→ deterministic 184xx port, launchd-supervised); the Go significance/recall producer calls it via
CFG_..._LOCAL_LINKPRED_URL, exactly like the ASR/image servers.

Interface (inductive → the caller passes pre-extracted pair features; Go computes them from the graph it
already owns, using the same recipe as features.py):
    POST /predict  {"features": [[...], ...]}  ->  {"scores": [float, ...]}   # link logits/probabilities

    PORT=18450 python server.py
"""
from __future__ import annotations
import asyncio, json, os
import numpy as np
import mlx.core as mx
from model import InductiveLinkScorer

WEIGHTS = os.environ.get("LINKPRED_WEIGHTS", "weights")
PORT = int(os.environ.get("LINKPRED_PORT", os.environ.get("PORT", "18450")))

_meta = json.load(open(os.path.join(WEIGHTS, "meta.json")))
_sc = np.load(os.path.join(WEIGHTS, "scaler.npz"))
_mu, _sd = _sc["mu"], _sc["sd"]
_model = InductiveLinkScorer(in_dim=_meta["in_dim"])
_model.load_weights(os.path.join(WEIGHTS, "scorer.npz"))
_model.eval()
_lock = asyncio.Lock()


def _score(features: np.ndarray) -> np.ndarray:
    fz = ((features - _mu) / _sd).astype(np.float32)
    logits = np.array(_model(mx.array(fz)))
    return 1.0 / (1.0 + np.exp(-logits))  # probability


try:
    from fastapi import FastAPI
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    FastAPI = None

if FastAPI is not None:
    app = FastAPI(title="shubo-linkpred-mlx")

    class PredictReq(BaseModel):
        features: list[list[float]]

    @app.get("/health")
    def health():
        return {"status": "ok", "in_dim": _meta["in_dim"], "pretrain": _meta.get("pretrain"),
                "hits@50": _meta.get("hits@50"), "metal": mx.metal.is_available()}

    @app.post("/predict")
    async def predict(req: PredictReq):
        async with _lock:
            arr = np.asarray(req.features, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != _meta["in_dim"]:
                return {"error": f"expected [N,{_meta['in_dim']}] features, got {arr.shape}"}
            return {"scores": _score(arr).tolist()}

    if __name__ == "__main__":
        uvicorn.run(app, host="0.0.0.0", port=PORT)
else:
    if __name__ == "__main__":
        raise SystemExit("install fastapi+uvicorn to serve; model/_score importable standalone")
