# Serving docling on the Apple-Silicon fleet — plain-MLX host serving

> **Status: Built.** Targets the production fleet (Apple-Silicon MacBook Pro k3s nodes). Closes the
> M7 W1b producer gap (`backend/docs/artifact/m7-w1b-producer-wiring.md` Part 1) *and* gives
> Metal-accelerated document parsing. Served by a **plain-MLX FastAPI process** (`server/server.py`) —
> one dedicated MLX process per model, **no Ray** (Ray is dropped, see "Why not …"). Page-level
> throughput in prod comes from running **N independent processes** behind a Service, not from
> in-process replicas. Consistent with the deploy repo's ADR 009
> (`deploy/docs/adr/fleet/009-host-managed-mlx-model-serving.md`). Owner: @pinglin.

## The problem

Structure-aware RAG needs a **docling producer** that emits the canonical `DoclingDocument`
`export_to_dict()` tree (`schema_name: "DoclingDocument"` — the exact bytes
`docdoc.FromDoclingExport` consumes). Two hard constraints shape *how* we serve it:

1. **The fleet is Apple-Silicon, and containers can't see Metal.** k3s nodes are MacBook Pros; pods
   run in OrbStack Linux VMs with **no Metal passthrough**. Torch/EasyOCR docling inside a container is
   **CPU-bound** (multi-GB weights, slow). Acceleration is only available to a **native macOS host
   process** (MLX/Metal, unified memory).
2. **The single Metal GPU is the throughput bottleneck.** One Mac has one GPU, and every host model
   server contends for it. So docling is its own **dedicated** process on its own port (the ADR 009
   one-dedicated-server-per-model rule), and page throughput is bought by **process count**, not by
   stacking replicas onto one GPU.

## The model — `unlimited-ocr-mxfp8-mlx` (the default)

[`sahilchachra/unlimited-ocr-mxfp8-mlx`](https://huggingface.co/sahilchachra/unlimited-ocr-mxfp8-mlx)
is the default document OCR/parser checkpoint. The model card reports **4.98 GB peak memory** and
**3.66 GB disk** on an M4 Pro, with stronger OCR quality than the lower-memory 4-bit variants.
It replaces the heavy torch+EasyOCR docling stack with a Metal-accelerated VLM:

- **Input:** a rendered **page image** (PIL). **Output:** Markdown from Unlimited-OCR.
- **It preserves our backend contract.** The server keeps the model output in `markdown_pages` and
  wraps each page into a minimal `DoclingDocument` text leaf so `docdoc.FromDoclingExport` continues
  to consume `structured_document`:

```python
from granite_docling import GraniteDocling
from PIL import Image

engine = GraniteDocling("sahilchachra/unlimited-ocr-mxfp8-mlx").load()
result = engine.convert([Image.open(page_png).convert("RGB")])

structured_document = result["structured_document"]  # schema_name="DoclingDocument"
markdown_pages      = result["markdown_pages"]
```

So a single host server returns **both** `markdown_pages` (unchanged contract) **and**
`structured_document` (the W1b payload) — it *is* the producer the M7 handoff
(`m7-w1b-producer-wiring.md` Part 1) asks for, and it runs on Metal.

> Note: `ibm-granite/granite-docling-258M-mlx` remains supported via
> `SHUBO_DOCLING_MODEL=ibm-granite/granite-docling-258M-mlx` when a smaller DocTags producer is more
> important than OCR quality.

## Serving — a plain-MLX FastAPI process on the macOS host

A single load-bearing rule shapes everything:

> **The server must be a *native arm64 host process* to use MLX/Metal.** *Inside* a Linux container
> there is no Metal (same wall as everything else). So the docling server runs on the **macOS host**
> (launchd-supervised, exactly like the VLM/embeddings/ASR host servers in `buckle/scripts/sandbox/`),
> and k8s routes to its **HTTP endpoint** on a host port.

This is the same host-process pattern proven for the VLM/embeddings/ASR servers: **one dedicated MLX
process per model**, a plain `mlx_vlm`-backed FastAPI app — **no Ray** (see "Why not …").

### The server (`server/server.py`)

`server.py` loads `sahilchachra/unlimited-ocr-mxfp8-mlx` once by default and exposes three routes:

- **`GET /health`** — `{status, model, loaded}`.
- **`POST /convert`** — `{"pdf_b64": "…"}` or `{"image_b64": "…"}` → the DoclingDocument contract.
- **`POST /v1alpha/namespaces/{ns}/models/{model}/versions/{version}/trigger`** — the
  **model-backend-trigger-compatible** drop-in. It reads `taskInputs[0].data.doc_content` (a data-uri /
  base64 doc) and returns `{"taskOutputs": [{"data": …}]}`, where `data` carries `markdown_pages` +
  `structured_document` — exactly what the parsing-router's `routedConvertResultParser` consumes.

The `/trigger` path mirrors the served `models/docling` model's shape, so pointing the parsing-router
step's endpoint at this server is a drop-in — callers don't change.

> The handlers are `async` on purpose: MLX's Metal stream is **thread-local**, so generation must run
> on the same event-loop thread that loaded the model (a sync `def` would dispatch to a threadpool
> worker and raise `no Stream(gpu, 0)`; the model is warmed on the event-loop thread at startup). One
> process therefore serves one page at a time (~2.3 s/page); for throughput, run **N processes** behind
> a Service — not replicas in one process.

### Throughput — N plain processes behind a round-robin Service

Generation is the floor, so per-page latency can't be driven below it on one GPU. Throughput comes
from **page-level parallelism**: a multi-page doc fans out page-by-page, and the pages are spread
across **N independent `server.py` processes** sitting behind a round-robin k8s **Service**. Each
process is a dedicated MLX model on its own host port; the Service picks the boundary, the processes
do the work. Size **N** by benchmarking per-page latency on the target Mac (and remember those
processes share the one GPU with the host's other model servers).

> One Metal GPU per Mac is the constraint that decides the shape. Running N processes is the way to
> overlap page generation across pages; in-process replicas/autoscaling buy nothing on a single GPU,
> which is exactly why Ray was dropped (see "Why not …").

### Performance levers (priority)

| Lever | Why it helps | Notes |
|---|---|---|
| **MXFP8 quantization** | Keeps Unlimited-OCR under ~5 GB peak memory | one worker per constrained MacBook until measured otherwise |
| **MLX unified memory** | Zero CPU↔GPU copy on Apple Silicon | MLX is designed for this; the whole reason to host on the Mac |
| **N processes behind a Service** | Page-level parallelism across a doc's pages | small model → high process density per Mac |
| **`max_tokens` cap + `temp=0.0`** | Bounded, deterministic generation | cap per page |
| **Page fan-out** | A 30-page PDF = 30 independent requests | the round-robin Service spreads them across processes |

Primary throughput = **processes × per-page latency**. Benchmark per-page latency on the target Mac
before sizing **N**.

## Routing — how k8s reaches the host server

The backend already has the seam; pick the boundary:

- **Option A — redirect the recipe `model_url` (smallest).** The parsing-router preset templates the
  docling endpoint off a `model_url` variable
  (`backend/services/artifact/pkg/pipeline/preset/pipelines/parsing-router/v1.2.0/recipe.yaml`). Point
  it at the host server's `/trigger` endpoint (a stable node IP / `ExternalName` Service /
  `host.docker.internal` in sandbox; the round-robin Service in prod). model-backend is bypassed for
  docling. Least code; loses model-backend's registry/health/accounting.
- **Option B — model-backend external-utility runtime (cleaner).** Register `docling` with a
  `runtime_ref` host endpoint via the existing `staticruntime` seam
  (`backend/services/model/pkg/llm/runtime/staticruntime/`), and add a `/trigger`-shaped adapter
  (docling is not OpenAI-compatible, so it needs a small non-LLM provider alongside `openaicompat`).
  Keeps one front door, health probing (`GET /health`), and the registry. More work.

**Recommendation:** ship **A** to light up M7 end-to-end fast (the recipe already supports it), then
migrate to **B** for a single managed front door once the host server is stable.

### Host-process supervision

Reuse buckle's existing host-model supervisor (`buckle/scripts/sandbox/shared-model-servers.sh` +
`start-local-models.sh`, launchd, refcounted, deterministic ports — the shared host model-server port
band is **12400-12499**, below the per-sandbox container port range so it can never collide; docling's
derived port is **12463**) — register a `docling` role pointing at this server. For **prod**, run **N
`server.py` processes** as launchd daemons on a **labeled** fleet Mac (mirror the derper
`ai.shubo.derper` / the `node.shubo/ozone-datanode` pinning pattern), fronted by a round-robin k8s
Service. Mind the OrbStack double-NAT + lossy-WiFi fabric (see infra memory) when choosing the node
and the route.

## Build plan

1. **`docling/server/`** — the plain-MLX FastAPI server (`mlx_vlm` + `docling-core`); `/trigger`
   returns `{markdown_pages, structured_document}`. Pin `mlx-vlm`, `docling-core`, model rev. Add
   `/health`.
2. **Verify the contract** — assert `structured_document.schema_name == "DoclingDocument"` and that
   `docdoc.FromDoclingExport` parses it (reuse the #349 round-trip test).
3. **buckle role** — register `docling` in `shared-model-servers.sh` / `start-local-models.sh`
   (launchd, host port in the 12400-12499 band), gateable via `SHUBO_LOCAL_DOCLING_ENABLE`. Sandbox
   first.
4. **Route (Option A)** — point the parsing-router `model_url` at the host server; ingest a PDF with
   `CFG_DYNORG_DOCLING_CONVERSION_ENABLED=true` → `converted_file.evidence_tree` non-NULL → grounded
   chunking mints real `anchor_id`s.
5. **Prod** — N launchd daemons on a labeled Mac + a round-robin k8s Service; benchmark per-page
   latency; size **N**. Optionally migrate to Option B.
6. **(Later) Option B** — model-backend external-utility runtime + non-LLM `/trigger` provider.

## Why not …

- **Torch/EasyOCR docling in a container** — CPU-only on the fleet (no Metal), multi-GB, slow. This is
  the status quo we're replacing.
- **Ray (Serve or in-pod)** — **dropped.** Ray inside k8s containers has no Metal passthrough, so it
  defeats the purpose; and on the host, Ray Serve's replicas/autoscaling buy nothing on a **single**
  GPU — the one GPU is the bottleneck, so concurrency must come from N independent host processes, not
  from in-process replicas. The plain-MLX FastAPI server is the whole design, not an MVP that Ray later
  wraps.
- **Co-serving docling with another model in one process** — rejected per ADR 009: one dedicated MLX
  server per model. Co-serving just serializes two models onto the same GPU stream.

## References

- Default model: https://huggingface.co/sahilchachra/unlimited-ocr-mxfp8-mlx
- Granite fallback: https://huggingface.co/ibm-granite/granite-docling-258M-mlx ·
  https://www.ibm.com/new/announcements/granite-docling-end-to-end-document-conversion
- Canonical design: `deploy/docs/adr/fleet/009-host-managed-mlx-model-serving.md` (ADR 009 —
  host-managed MLX, one-dedicated-server-per-model, the pod↔host-process pattern)
- Backend seams: `services/model/pkg/llm/runtime/staticruntime/`,
  `services/artifact/pkg/pipeline/preset/pipelines/parsing-router/v1.2.0/recipe.yaml`,
  `services/artifact/pkg/pipeline/client.go` (the "Docling Model" parser)
- Producer contract: `backend/docs/artifact/m7-w1b-producer-wiring.md` (Part 1) · ADR-0020
  (`backend/docs/adr/dynamic-organization/0020-extraction-producer-architecture.md`)
- Host-server pattern: `buckle/scripts/sandbox/qwen3-asr-server.py`,
  `buckle/scripts/sandbox/mlx-embeddings-server.py`, `shared-model-servers.sh`
