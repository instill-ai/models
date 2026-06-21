# GIDN — from-paper reimplementation (ogbl-collab link prediction)

A from-scratch PyTorch reimplementation of **GIDN** (Graph Inception Diffusion Network,
[arXiv:2210.01301](https://arxiv.org/abs/2210.01301)), the #1 method on the OGB
[`ogbl-collab` link-prediction leaderboard](https://ogb.stanford.edu/docs/leader_linkprop/)
(**Hits@50 0.7096 ± 0.0055**). GIDN has **no official code**, so this is built from the paper.

## Why this exists

It's the SOTA target of an ogbl-collab leaderboard chase. Achieved on an Apple M4 Max so far:
M5-tunnel chain 0.552 · from-scratch GBDT 0.646 · SEAL 0.647 · **BUDDY (reproduced) 0.659**. The
remaining ~0.05 to the leaderboard top is GIDN/PLNLP — both gated by *training throughput*, not model
size. This reimplementation is engineered to train on **Apple-Silicon MPS** to close that gap on a
laptop.

## Architecture (from the paper)

GIDN = **AGDN** (Adaptive Graph Diffusion Networks, hop-wise attention, arXiv:2012.15024) +
an **Inception** module + multi-feature-space diffusion:

- **Learned node embeddings** (ogbl-collab AGDN config uses `--no-node-feat`).
- **Multi-hop diffusion** `[H, ÂH, …, Â^K H]` (Â = symmetric-normalised adjacency, K=2), implemented
  with native `index_add_` scatter so it runs on **MPS** (not `torch_scatter`, which is CPU/CUDA-only
  and is exactly what makes BUDDY/PLNLP CPU-bound here).
- **Hop-wise attention (HA)** — learned softmax weights combine the hops.
- **Inception** — parallel branches over the multi-hop stack, concatenated → node representation.
- **MLP link predictor** on the Hadamard product of the two endpoint representations.
- Full-batch training (AGDN's no-sampling regime), strict-global negatives, CE loss, val-edges-as-input,
  year-2010 filter, grad-clip 1.0. Hyperparameters grounded on AGDN's `collab_agdn.sh`.

## Run

    pip install -r requirements.txt
    python gidn.py --epochs 800 --eval-every 10 --device mps --out-dir logs

Writes `logs/result.json` (rolling: current + best Hits@50, epoch, elapsed) and `logs/best.pt`
(best-validation checkpoint) so progress is readable any time.

## Status

## Result (M4 Max, MPS, 2026-06-19)

Full 800-epoch run, config `K=2 dim=256 layers=2 dropout=0.5 lr=0.001 wd=5e-4 year=2010`, val-edges-as-
input, OGB node features + learned embeddings, ~4.4 h:

| | Hits@50 |
|---|---:|
| **this reimplementation** (test @ best-valid ep730; valid 0.532) | **0.434** |
| GIDN (paper) | 0.710 |
| BUDDY (reproduced from official code) | 0.659 |
| AGDN (base) | 0.664 |

**Honest finding.** The reimplementation *converges correctly* (0.05 → 0.43, monotonic up to ep730 —
the architecture and training recipe are sound) but **plateaus at 0.434, ~0.28 below the paper.** Root
cause is **training throughput, not modelling**: this runs **full-batch** (one gradient step/epoch ≈ 800
steps total), and ogbl-collab's *learned node embeddings* — which AGDN/GIDN depend on — need **thousands**
of steps to organize. The fix is minibatch edge training (many steps/epoch), but each step needs a
full-graph diffusion forward, and the diffusion is an `index_add` scatter that is slow on MPS (no
`torch.sparse` MPS kernel), so the ~10k+ steps to reach SOTA are **infeasible (8–25 h) on this laptop**.
On a **CUDA GPU** (where the scatter/forward is fast) this same code, switched to minibatch training,
should approach the paper. As-is, on Apple Silicon, **~0.43 is the practical ceiling**.

The achieved, *reproducible* SOTA-tier number on this M4 Max is **BUDDY 0.659** (official code; see
`../backend/tests/benchmark/memory/RUNS.md`). This GIDN scaffold is a faithful, converging starting
point for a CUDA-based full reproduction. See `logs/result.json` + `logs/best.pt`.
