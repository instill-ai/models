#!/usr/bin/env bash
# Reproduce PLNLP on ogbl-collab (feasible simple config) in an isolated conda env.
# Headline 0.7059 needs --random_walk_augment (~180s/epoch CPU, ~40h) → CUDA only; this runs the
# ~10s/epoch simple config (~2.4h CPU, ceiling ~0.64-0.68). See README.md.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="${HERE}/upstream"

# 1. clone upstream (not vendored)
[ -d "$REPO" ] || git clone --depth 1 https://github.com/zhitao-wang/PLNLP "$REPO"

# 2. isolated env (reuse the 'ss' env from the BUDDY repro if present; else create it)
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | grep -q '^ss '; then
  conda create -y -n ss python=3.11
  conda run -n ss pip install "torch==2.0.0"
  conda run -n ss pip install torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
  conda run -n ss pip install "torch_geometric==2.3.1" "numpy==1.26.4" ogb
fi

# 3. run the FULL leaderboard config that reaches 0.7059 (single run, 800 epochs).
#    ~180 s/epoch on CPU (random-walk augmentation) → ~40 h. eval_steps=25 for trajectory visibility.
cd "$REPO"
OMP_NUM_THREADS=10 PYTHONUNBUFFERED=1 conda run -n ss python -u main.py \
  --data_name=ogbl-collab --predictor=DOT --use_valedges_as_input=True --year=2010 \
  --train_on_subgraph=True --epochs=800 --eval_steps=25 --dropout=0.3 --gnn_num_layers=1 \
  --grad_clip_norm=1 --use_lr_decay=True --random_walk_augment=True --walk_length=10 \
  --loss_func=WeightedHingeAUC --runs=1 \
  | tee "${HERE}/logs/run.log"
