# Docling MLX host server

Metal-accelerated document parsing for the Apple-Silicon fleet. Defaults to
[`sahilchachra/unlimited-ocr-mxfp8-mlx`](https://huggingface.co/sahilchachra/unlimited-ocr-mxfp8-mlx)
via `mlx_vlm`, and returns the canonical DoclingDocument contract the backend consumes:

```json
{ "markdown_pages": ["…"], "structured_document": { "schema_name": "DoclingDocument", … }, "num_pages": 2 }
```

For DocTags models such as `ibm-granite/granite-docling-258M-mlx`, `structured_document` is the
direct `export_to_dict()` tree. For Unlimited-OCR, the model emits Markdown, so the server wraps one
page-level text leaf per page into a minimal `DoclingDocument` tree while preserving the OCR output
in `markdown_pages`. This keeps the `docdoc.FromDoclingExport` seam stable. Design + rationale:
[`../docs/mlx-host-serving.md`](../docs/mlx-host-serving.md).

## Why this runs on the host (not in a container)

The fleet's k8s pods are Linux VMs (OrbStack) with **no Metal passthrough**. MLX needs the Apple GPU,
so this is a **host process** (launchd-supervised, like the existing `mlx-vlm`/ASR servers), and k8s
routes to it — the same `staticruntime`/`model_url` seam the other host models use.

## Setup

Native arm64 Python 3.10+ (Metal). The pins matter — see `requirements.txt` (`mlx-vlm` 0.6.x
carries the DeepSeekOCR loader and MXFP8 quantization support required by Unlimited-OCR). Do not
use Apple's system Python 3.9; it cannot resolve the current `mlx-vlm` / Transformers pins.

```bash
python3.12 -m venv .venv
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

Env: `SHUBO_DOCLING_MODEL`, `SHUBO_DOCLING_PROMPT`, `SHUBO_DOCLING_MAX_TOKENS` (4096),
`SHUBO_DOCLING_PDF_DPI` (150).

Set `SHUBO_DOCLING_MODEL=ibm-granite/granite-docling-258M-mlx` to use the older, smaller DocTags
producer instead of the Unlimited-OCR default.

## Performance (measured, M-series)

- The Unlimited-OCR MXFP8 card reports **4.98 GB peak memory** and **3.66 GB disk** on an M4 Pro.
  Keep one worker per low-memory MacBook unless measured headroom says otherwise.
- The older Granite DocTags model is still available by override when memory is more important than
  OCR recall/quality.
- The work is generation-bound, so per-page latency has a floor; **throughput scales with replicas**
  (Ray Serve front): pages fan out across replicas.

## Verify (the proven contract)

```bash
.venv/bin/python probe_granite_docling.py test_page.png      # page -> contract, schema check
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
| `requirements.txt` | pinned stack (`mlx-vlm` 0.6.x + Transformers 5.x for DeepSeekOCR/MXFP8) |
