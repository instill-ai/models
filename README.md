# Shubo Custom Models

This is **Shubo's custom model repository** — the single home for the models Shubo
serves and the model research it owns. It holds two kinds of thing:

1. **Served models** — model definitions + serving code that **`model-backend`** deploys and that
   the rest of the stack triggers (chat / VLM / embedding / detection models, and the
   **`docling`** document parser that powers structure-aware RAG ingestion).
2. **Model research projects** — Shubo's own MLX/graph work that produces or evaluates models
   (`linkpred-mlx`, `gidn`, `plnlp` — Apple-Silicon link-prediction experiments).

It is part of the **Shubo workspace** and is **managed by `buckle`**: `buckle init` clones it as a
sibling of `backend`/`frontend`/`deploy`, and provisions the shared agent context
(`AGENTS.md`/`CLAUDE.md`/`.claude/skills`) into it. It is **not** a sandbox service tier — it does not
boot in `buckle sandbox` and is not worktreed per task (it sits alongside `deploy`/`cloud`: present and
agent-aware, not part of the running stack). See `../buckle` and the workspace `../AGENTS.md`.

## Layout

```
models/
├── docling/               # the docling document parser served by model-backend (v0.1.x);
│                          #   emits DoclingDocument structure for structure-aware RAG (see below)
├── custom/                # other custom served-model scaffolding
├── <served-model>/        # one dir per served model (LLM / VLM / embedding / detection),
│                          #   each with its own README + versioned vX.Y.Z/ folders
│                          #   e.g. qwen-2-5-vl-7b-instruct, gte-Qwen2-1.5B-instruct, yolov7, …
├── linkpred-mlx/          # MLX link-prediction (ogbl-collab / arxiv-semantic) — research
├── gidn/                  # Graph Inception Diffusion Networks link-prediction — research
└── plnlp/                 # Pairwise Learning for Neural Link Prediction — research
```

Each **served model** folder carries its own `README.md` (config, weights, build/push steps) and one
or more `vX.Y.Z/` version folders. Open the folder README for that model's specifics.

## Serving model

Served models run on **`model-backend`** (the Ray-Serve plane historically; **Ray is disabled in
production today** — see `backend/services/model`). Because the production fleet is **Apple-Silicon
MacBook Pro** k3s nodes, GPU-accelerated inference does **not** run inside the Linux containers (no
Metal passthrough). Instead the established pattern is **host-managed model servers**: an MLX/Metal
FastAPI process runs on the macOS host (supervised by `buckle` via `launchd`), and `model-backend`
routes to it through the `staticruntime` / `runtime_ref` seam (the same way `gemma`/`mlx-vlm`/ASR are
served today). See `buckle/scripts/sandbox/qwen3-asr-server.py` for the host-server template and
`backend/services/model/pkg/llm/runtime/` for the routing seam.

### Apple-Silicon (MLX) docling hosting

`docling` is the document parser behind structure-aware RAG (it must emit the
`DoclingDocument` `export_to_dict()` tree the backend consumes — see
`backend/docs/artifact/m7-w1b-producer-wiring.md`). To get Metal acceleration on the Apple-Silicon
fleet, docling is hosted as an **MLX host server** (mirroring the ASR/VLM host servers) rather than a
Ray container. The design — host server, `buckle` role registration, and the two routing options
(redirect the parsing-router `model_url`, vs. a model-backend external-utility runtime) — lives in
[`docling/docs/mlx-host-serving.md`](./docling/docs/mlx-host-serving.md).

## Supported serving runtimes (LLM/VLM)

|                           Runtime                           | AMD64 CPU | ARM64 CPU | AMD64 GPU (CUDA) | Apple GPU (Metal/MLX) |
| :---------------------------------------------------------: | :-------: | :-------: | :--------------: | :-------------------: |
|        [vLLM](https://github.com/vllm-project/vllm)         |     ✅     |     ✅     |        ✅         |          —            |
|     [mlx-vlm](https://github.com/Blaizzy/mlx-vlm)           |     —     |     —     |        —         |          ✅           |
| [Transformers](https://github.com/huggingface/transformers) |     ✅     |     ✅     |        ✅         |        ✅ (MPS)        |
| [llama.cpp](https://github.com/ggml-org/llama.cpp)          |     ✅     |     ✅     |        ✅         |          ✅           |

On the Apple-Silicon fleet, **MLX/Metal runtimes are the accelerated path** (host-managed, as above).

## Research projects

- **`linkpred-mlx`** — Apple-Silicon (MLX) link prediction on `ogbl-collab` / arxiv-semantic graphs.
- **`gidn`** — Graph Inception Diffusion Networks for link prediction.
- **`plnlp`** — Pairwise Learning for Neural Link Prediction.

These are reproducible experiments (data + scripts + logs), not served models; they feed model design.

## License

MIT — see the workspace `LICENSE`.
