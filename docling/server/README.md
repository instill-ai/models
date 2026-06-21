# granite-docling MLX host server

Metal-accelerated document parsing for the Apple-Silicon fleet. Runs
[`ibm-granite/granite-docling-258M-mlx`](https://huggingface.co/ibm-granite/granite-docling-258M-mlx)
(a 258M VLM, ~631 MB) via `mlx_vlm`, and returns the canonical DoclingDocument contract the backend
consumes:

```json
{ "markdown_pages": ["…"], "structured_document": { "schema_name": "DoclingDocument", … }, "num_pages": 2 }
```

`structured_document` is exactly the `export_to_dict()` tree `docdoc.FromDoclingExport` decodes
(round-trip verified) — so this **is** the M7 W1b producer, on Metal. Design + rationale:
[`../docs/mlx-host-serving.md`](../docs/mlx-host-serving.md).

## Why this runs on the host (not in a container)

The fleet's k8s pods are Linux VMs (OrbStack) with **no Metal passthrough**. MLX needs the Apple GPU,
so this is a **host process** (launchd-supervised, like the existing `mlx-vlm`/ASR servers), and k8s
routes to it — the same `staticruntime`/`model_url` seam the other host models use.

## Setup

Native arm64 Python (Metal). The pins matter — see `requirements.txt` (granite-docling needs the slow
Idefics3 image processor, which only `transformers <5` auto-maps; that also keeps the stack **torch-free**).

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

Two entry points, same core (`granite_docling.py`):

```bash
# Plain FastAPI MVP — one model, single process (mirrors qwen3-asr-server.py)
.venv/bin/uvicorn server:app --host 0.0.0.0 --port 8088

# Ray Serve front — multi-replica, PAGE-PARALLEL (the throughput lever; a doc's pages fan out
# across replicas). SHUBO_DOCLING_MIN_REPLICAS / _MAX_REPLICAS (default 1..6).
.venv/bin/serve run serve_app:app
```

Endpoints (both): `GET /health`, `POST /convert` with `{"pdf_b64": "…"}` or `{"image_b64": "…"}`.

Env: `SHUBO_DOCLING_MODEL`, `SHUBO_DOCLING_MAX_TOKENS` (4096), `SHUBO_DOCLING_PDF_DPI` (150).

## Performance (measured, M-series)

- **~2.3 s/page warm** (~300 tok/s; ~700 DocTags tokens). The work is generation-bound, so per-page
  latency has a floor; **throughput scales with replicas** (Ray Serve front): a 2-page doc ran in
  **1.7 s on 2 replicas** (≈ one page's time — pages parallelized).
- The latest `mlx-vlm` (0.6.x) was tested and is **not faster** (comparable, and it forces
  `torch`+`torchvision`), so we pin `0.3.3` — lighter, torch-free, same speed.

## Verify (the proven contract)

```bash
.venv/bin/python probe_granite_docling.py test_page.png      # page -> export_to_dict, schema check
.venv/bin/python perf_probe.py test_page.png                 # latency / tok/s + resolution sweep
.venv/bin/python smoke_serve.py                              # Ray Serve: 2 replicas, /convert, contract
```

The Go consumer round-trip (`docdoc.FromDoclingExport` parses this server's `structured_document`) is
verified in the backend (`services/artifact/pkg/extraction/docdoc`, the #349 test approach).

## Layout

| file | role |
|------|------|
| `granite_docling.py` | core: load MLX model once, `convert(images) -> contract`, PDF render |
| `server.py` | plain FastAPI MVP (single process) |
| `serve_app.py` | Ray Serve front (multi-replica, page-parallel, autoscale) |
| `probe_*.py`, `perf_probe.py`, `smoke_serve.py` | verification / benchmarks |
| `requirements.txt` | pinned stack (mlx-vlm 0.3.3 + transformers <5, torch-free) |
