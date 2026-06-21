# shubo-linkpred-mlx — inductive, MLX-hosted link-predictor (dynamic-org cold-start)

The **deployable** learned association scorer for Shubo's M5 tunnel graph, built for a **greenfield**
production DB (no representative graph data — auto-tagging + memory association must work day one).
Design home: `backend/docs/agent/dynamic-organization-m5.md` §"Cold-start deployment"; board:
`roadmap/FEAT-dynamic-organization.md` (cold-start scope item).

## Why inductive (the load-bearing decision)

The PLNLP/GIDN models benchmarked on ogbl-collab (0.706/0.710) are **transductive** — they learn a
per-node embedding table tied to one fixed graph, so they **cannot score a node they never trained on**
→ useless on a brand-new namespace. This scorer is **inductive**: it scores a pair purely from PAIR
FEATURES, so it generalizes zero-shot to any graph (mirrors M1's zero-shot auto-tagger). Tiny by
construction (MLP over features) → trains in seconds, hosts trivially on MLX.

## Pieces

- `model.py` — `InductiveLinkScorer` (MLX MLP: pair-features → link logit).
- `sketches.py` — **ELPH/BUDDY MinHash subgraph sketches** (the BUDDY route): per-node MinHash sigs
  propagated over hops → the `(d_u,d_v)` multi-hop neighborhood-overlap bucket counts that 2-hop
  heuristics can't capture. Fully inductive (c11 ≈ exact common-neighbours at corr 0.992).
- `features.py` — inductive pair features: structural (CN/AA/RA/PA/Jaccard/wCN/cn-deg-stats) +
  node-embedding pair (cos+Hadamard — Shubo's pooled-qwen shape; validated to help) + hook for the
  Wu-Palmer taxonomic prior. Combined with the sketch features = the BUDDY feature set (19-d here).
- `datasets.py` — corpus loaders behind one `(g, split)` shape: `ogbl-collab` (collaboration proxy) and
  `ogbn-arxiv` (arXiv paper **citations** — the intent-matched scholarly domain, tractable on a laptop;
  the feasible stand-in for the SciDocs/S2ORC co-citation target). `--dataset` selects.
- `train.py` — transfer-pretrain on the chosen corpus. **AdamW + early-stopping on validation** (prevents
  the easy-random-negative overfit). `--loss {bce,bpr,infonce}` (ranking objectives — see Status) and
  `--hard-neg-frac` (2-hop hard negatives). OGB-evaluated, saves `weights*/{scorer.npz, scaler.npz, meta.json}`.
- `server.py` — FastAPI `/health` + `/predict` (load-once + asyncio lock), the hostable MLX service
  matching `buckle/scripts/sandbox/flux-image-server.py`. **Validated serving on Metal** (in_dim 19,
  metal=true; strong pair → 0.99999, weak → 0.00007).

## Run

    pip install -r requirements.txt
    python train.py --epochs 60                 # transfer-pretrain (saves weights/)
    PORT=18450 python server.py                 # host the /predict service
    curl localhost:18450/health

## Hosting (production shape)

Add `start_mlx_linkpred()` to `buckle/scripts/sandbox/start-local-models.sh` (deterministic 184xx port +
launchd supervision, via `shared_server_start`), and wire `CFG_..._LOCAL_LINKPRED_URL` in the agent
backend so the M5 significance/recall producer calls `/predict` (Go computes pair features from the graph
it owns, using the `features.py` recipe). The interface is `{"features": [[...]]} → {"scores": [...]}`.

## Status / gates

- **BUDDY-route inductive scorer + MLX serving = built & validated.** Full feature set (ELPH multi-hop
  sketches + structural + node-embedding) → MLX MLP, trained on Metal in ~60 s (early-stopped), and
  **served on Apple Silicon via `/predict`** (the explicit goal). The architecture is the deployable one:
  inductive (generalizes zero-shot), tiny, MLX-hosted.
- **Objective tuning (ogbl-collab test Hits@50).** The training objective is the real lever — ranking
  beats pointwise, confirming the LambdaMART lesson:

  | objective | Hits@50 | Hits@10 |
  | --- | ---: | ---: |
  | BCE (pointwise) | 0.615 | 0.420 |
  | BPR (pairwise ranking) | 0.623 | 0.450 |
  | **InfoNCE (K=16 contrastive) — default, canonical `weights/`** | **0.628** | **0.455** |
  | InfoNCE + hard negatives (0.3 / 0.5) | 0.604 / 0.590 | — |

  **InfoNCE is canonical.** Hard 2-hop negatives *backfire* — ogbl-collab's eval negatives are random, so
  training on hard ones mis-calibrates the easy ranking the metric rewards (the same effect seen earlier).
  Residual gap to our LightGBM 0.646 is the MLP-vs-GBDT-on-tabular difference; matching BUDDY 0.659 would
  need node-feature *propagation* (bigger build). Note: InfoNCE optimizes *ranking*, not calibration — the
  server's sigmoid keeps ordering correct (Hits@K unaffected); if absolute-probability thresholds are
  needed, Platt-calibrate post-hoc. On the proxy these gains are small and the metric doesn't predict
  Shubo transfer — the real remaining lever is per-namespace fine-tune on accrued M5 edges.
- **Shubo node features** (pooled qwen Item vectors per Room + taxonomic prior) are hooked in
  `features.py` but **not yet wired** — no per-Room vector table exists (M5 stores only edges); derive by
  pooling Item vectors when integrating.
- **Scholarly-citation pretrain = done & feasible on-laptop** via `ogbn-arxiv` (169K nodes, ~984K train
  edges) — the tractable stand-in for the SciDocs/S2ORC co-citation target. **Honest finding:** at a fair
  100K-negative-pool eval, arxiv test Hits@50 ≈ **0.617** — *the same tier* as ogbl-collab's 0.62, i.e.
  matching the corpus to scholarly-citations did **not** change the in-corpus difficulty for the inductive
  scorer (an earlier 50K-pool run's 0.70 was a smaller-pool artifact). The proxy Hits@50 measures fit to
  *that* corpus, not transfer to Shubo; domain-match's real payoff is on the (still-unmeasured) Shubo
  target, where citations remain the more defensible transfer source. S2ORC at scale needs a GPU box.
- **Per-namespace fine-tune** (on accrued M5 edges as weak labels) is the later usage-gated improvement,
  not the launch bar — day-one value is the transfer-pretrained inductive scorer.
