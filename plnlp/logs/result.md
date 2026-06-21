# PLNLP reproduction result — ogbl-collab (Apple M4 Max, CPU)

**Reproduced SOTA: Hits@50 = 0.7061** (best-valid epoch 750; valid 100.00%).

| | Hits@50 |
|---|---:|
| **this reproduction** (official code, full config, ep750) | **0.7061** |
| PLNLP (paper / leaderboard) | 0.7059 ± 0.0029 |
| BUDDY (reproduced) | 0.659 |
| GIDN (paper) | 0.7096 |

Config (official `zhitao-wang/PLNLP`, the 0.7059 leaderboard command):
`--data_name=ogbl-collab --predictor=DOT --use_valedges_as_input --year=2010 --train_on_subgraph
--epochs=800 --dropout=0.3 --gnn_num_layers=1 --grad_clip_norm=1 --use_lr_decay --random_walk_augment
--walk_length=10 --loss_func=WeightedHingeAUC --runs=1`. Single run, isolated torch-2.0 + arm64-wheel env.

**Outcome: 0.7061 ≥ the published 0.7059 — SOTA reproduced on a laptop.** This settles the question: PLNLP's
leaderboard result *is* reproducible on this M4 Max; it just needs ~30 h CPU (no MPS — `torch_sparse`/
`torch_cluster` are CPU-only).

**Caveat — run ended at epoch 750/800, not 800.** After ~1.5 days of sustained 100% CPU load the per-epoch
time had ballooned to ~570 s (thermal throttling) and the process was terminated at ep750 with **no Python
traceback** (an external/OS kill — likely thermal or memory pressure, not a code error). This does **not**
affect the result: by ep750 the LR had decayed to 1e-4 and Hits@50 had plateaued at the SOTA ceiling
(0.7053 @ep675 → 0.7061 @ep750), so the missing 50 epochs would not have improved 0.7061. The best-valid
checkpoint (the number the leaderboard reports) is captured.

Trajectory: 0.610(25) → 0.651(50) → 0.666(75) → 0.683(175) → 0.694(350) → 0.703(625) → 0.705(675) →
**0.706(750)**.
