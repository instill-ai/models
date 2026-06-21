# PLNLP reproduction (ogbl-collab link prediction)

Reproduction harness for **PLNLP** (Pairwise Learning for Neural Link Prediction,
[arXiv:2112.02936](https://arxiv.org/abs/2112.02936), official code
[zhitao-wang/PLNLP](https://github.com/zhitao-wang/PLNLP)) on OGB `ogbl-collab`. PLNLP is **co-#1 on the
leaderboard** at **Hits@50 0.7059 ± 0.0029** — statistically tied with GIDN's 0.7096 (overlapping error
bars). Unlike GIDN it has official, runnable code, so this is a *reproduction* (run their code), not a
from-scratch rebuild.

This directory is a **wrapper**, not a vendored copy: `reproduce.sh` clones the upstream repo, builds an
isolated env, and runs the exact command; results land in `logs/`. (We don't vendor upstream's source.)

## Hardware reality (Apple M4 Max)

PLNLP uses `torch_sparse` (SparseTensor message passing) and `torch_cluster` (`random_walk`), which have
**no MPS kernels** → it runs on **CPU** on this machine.

- **Headline 0.7059 config** (`--random_walk_augment --loss_func=WeightedHingeAUC --train_on_subgraph`):
  the per-epoch random walks are ~**180 s/epoch** on CPU → **~40 h for 800 epochs. Infeasible here.**
- **Simple config** (DOT predictor, no augmentation): ~10 s/epoch → **~2.4 h feasible**, but a lower
  ceiling (~0.64–0.68). This is the config we run on this laptop.

The headline 0.7059 needs a **CUDA GPU** (fast `torch_sparse`/`torch_cluster`), where the full config
runs in minutes/epoch.

## Run

    ./reproduce.sh         # clones upstream, builds env, runs the feasible simple config (~2.4 h CPU)

Env: Python 3.11, torch 2.0 (CPU), `torch_scatter`/`torch_sparse`/`torch_cluster` (universal2 arm64
wheels), `torch_geometric==2.3.1`, `numpy<2` — same isolated `ss` conda env used for the BUDDY
reproduction.

## Result

See `logs/result.md` (written after the run completes).
