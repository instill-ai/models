# Serving docling on the Apple-Silicon fleet — MLX host serving (with Ray Serve)

> **Status: Design.** Targets the production fleet (Apple-Silicon MacBook Pro k3s nodes). Closes the
> M7 W1b producer gap (`backend/docs/artifact/m7-w1b-producer-wiring.md` Part 1) *and* gives
> Metal-accelerated document parsing. Owner: @pinglin.

## The problem

Structure-aware RAG needs a **docling producer** that emits the canonical `DoclingDocument`
`export_to_dict()` tree (`schema_name: "DoclingDocument"` — the exact bytes
`docdoc.FromDoclingExport` consumes). Two hard constraints shape *how* we serve it:

1. **The fleet is Apple-Silicon, and containers can't see Metal.** k3s nodes are MacBook Pros; pods
   run in OrbStack Linux VMs with **no Metal passthrough**. Torch/EasyOCR docling inside a container is
   **CPU-bound** (multi-GB weights, slow). Acceleration is only available to a **native macOS host
   process** (MLX/Metal, unified memory).
2. **Ray is off in prod today** (`deploy/k8s/apps/model/configmap.yaml` `CFG_RAY_ENABLED:false`), and
   the legacy docling path is either the GPU-Ray container (inert) or the in-pipeline torch lib (CPU).
   Neither is accelerated. Performance is a first-class requirement (ingestion throughput).

## The model — `granite-docling-258M-mlx` (the convergence)

[`ibm-granite/granite-docling-258M-mlx`](https://huggingface.co/ibm-granite/granite-docling-258M-mlx)
is a **258M-param vision-language model** (631 MB, 4-bit MLX, "optimized to run efficiently on Apple
Silicon"). It replaces the heavy torch+EasyOCR docling stack with **one tiny Metal-accelerated VLM**:

- **Input:** a rendered **page image** (PIL). **Output:** **DocTags** — a layout-preserving markup
  (tables, code, math, reading order, bboxes).
- **It produces our exact contract.** DocTags convert straight to a `DoclingDocument`, which exports
  the tree we already consume:

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from docling_core.types.doc.document import DocTagsDocument, DoclingDocument
from PIL import Image

model, processor, config = load("ibm-granite/granite-docling-258M-mlx")   # cold load (Metal)
pil = Image.open(page_png)                                                # one page
prompt = apply_chat_template(processor, config, "Convert this page to docling.", num_images=1)
doctags = generate(model, processor, prompt, pil, max_tokens=4096, temperature=0.0, verbose=False)

dtd  = DocTagsDocument.from_doctags_and_image_pairs([doctags], [pil])
doc  = DoclingDocument.load_from_doctags(dtd)
structured_document = doc.export_to_dict()    # ← schema_name="DoclingDocument" — the M7 contract
markdown            = doc.export_to_markdown() # ← the markdown_pages the recipe already reads
```

So a single host server returns **both** `markdown_pages` (unchanged contract) **and**
`structured_document` (the W1b payload) — it *is* the producer the M7 handoff
(`m7-w1b-producer-wiring.md` Part 1) asks for, and it runs on Metal.

> Note: granite-docling is **page-image → DocTags**, so the server is fed page renders. Multi-page
> docs fan out page-by-page (see throughput). This matches the visual-RAG direction (ADR-0021) — the
> page image is already in hand.

## Serving — Ray Serve on the macOS host

Ray **does run natively on Apple Silicon** (single-node supported). The load-bearing rule:

> **A Ray Serve replica must be a *native arm64 host process* to use MLX/Metal.** Ray *inside* a Linux
> container gets no Metal (same wall as everything else). So the Ray cluster + Serve replicas run on
> the **macOS host** (launchd-supervised, exactly like today's `mlx-vlm`/ASR host servers in
> `buckle/scripts/sandbox/`), and k8s routes to the Serve **HTTP ingress** on a host endpoint.

This is the same host-process pattern already proven for `gemma`/`mlx-vlm`/ASR, upgraded from a plain
FastAPI server to **Ray Serve** for the performance features below.

### The deployment (batching + replicas)

```python
from ray import serve
from fastapi import FastAPI, Request

api = FastAPI()

@serve.deployment(
    num_replicas="auto",                       # autoscale on load
    autoscaling_config={"min_replicas": 1, "max_replicas": 4, "target_ongoing_requests": 4},
    ray_actor_options={"num_cpus": 2},         # MLX uses the GPU via Metal, scheduled by count not CUDA
)
@serve.ingress(api)
class DoclingMLX:
    def __init__(self):
        from mlx_vlm import load
        self.model, self.processor, self.config = load("ibm-granite/granite-docling-258M-mlx")

    @serve.batch(max_batch_size=8, batch_wait_timeout_s=0.05)
    async def _generate(self, pages: list) -> list:
        # batch page-images through MLX where supported; else iterate (still 1 replica, warm weights)
        ...

    @api.post("/v1alpha/namespaces/instill-ai/models/docling/versions/v0.1.0/trigger")
    async def trigger(self, req: Request):
        # decode page image(s) -> _generate -> DocTags -> DoclingDocument
        #   -> {"markdown_pages": [...], "structured_document": doc.export_to_dict()}
        ...
```

The ingress path mirrors the existing `models/docling/.../trigger` shape so callers don't change.

### Performance levers (priority)

| Lever | Why it helps | Notes |
|---|---|---|
| **Tiny model (631 MB)** | Many warm replicas fit in unified memory | vs multi-GB torch docling; cold-load once per replica |
| **MLX unified memory** | Zero CPU↔GPU copy on Apple Silicon | MLX is designed for this; the whole reason to host on the Mac |
| **Multiple replicas / autoscale** | Page-level parallelism across a doc's pages | `num_replicas:"auto"`; small model → high replica density |
| **`@serve.batch`** | Vectorized generation when MLX batching applies | VLM batch support is uneven — treat as opportunistic; replicas are the primary throughput lever |
| **`max_tokens` cap + `temp=0.0`** | Bounded, deterministic generation | DocTags are compact; cap per page |
| **Page fan-out** | A 30-page PDF = 30 independent requests | Ray Serve handles routing/queueing across replicas |

Primary throughput = **replicas × per-page latency**; batching is a secondary, opportunistic gain
given current MLX-VLM batch maturity. Benchmark per-page latency on the target Mac before sizing
`max_replicas`.

## Routing — how k8s reaches the host server

The backend already has the seam; pick the boundary:

- **Option A — redirect the recipe `model_url` (smallest).** The parsing-router preset templates the
  docling endpoint off a `model_url` variable
  (`backend/services/artifact/pkg/pipeline/preset/pipelines/parsing-router/v1.2.0/recipe.yaml`). Point
  it at the host Ray Serve ingress (a stable node IP / `ExternalName` Service / `host.docker.internal`
  in sandbox). model-backend is bypassed for docling. Least code; loses model-backend's
  registry/health/accounting.
- **Option B — model-backend external-utility runtime (cleaner).** Register `docling` with a
  `runtime_ref` host endpoint via the existing `staticruntime` seam
  (`backend/services/model/pkg/llm/runtime/staticruntime/`), and add a `/trigger`-shaped adapter
  (docling is not OpenAI-compatible, so it needs a small non-LLM provider alongside `openaicompat`).
  Keeps one front door, health probing (`GET /health`), and the registry. More work.

**Recommendation:** ship **A** to light up M7 end-to-end fast (the recipe already supports it), then
migrate to **B** for a single managed front door once the host server is stable.

### Host-process supervision

Reuse buckle's existing host-model supervisor (`buckle/scripts/sandbox/shared-model-servers.sh` +
`start-local-models.sh`, launchd, refcounted, deterministic ports) — register a `docling` role
pointing at this server. For **prod**, run the Ray Serve app as a launchd daemon on a **labeled** fleet
Mac (mirror the derper `ai.shubo.derper` / the `node.shubo/ozone-datanode` pinning pattern), fronted by
a k8s Service. Mind the OrbStack double-NAT + lossy-WiFi fabric (see infra memory) when choosing the
node and the route.

## Build plan

1. **`docling/server/`** — the Ray Serve app (`mlx_vlm` + `docling-core`), `/trigger` returns
   `{markdown_pages, structured_document}`. Pin `mlx-vlm`, `docling-core`, model rev. Add `/health`.
2. **Verify the contract** — assert `structured_document.schema_name == "DoclingDocument"` and that
   `docdoc.FromDoclingExport` parses it (reuse the #349 round-trip test).
3. **buckle role** — register `docling` in `shared-model-servers.sh` / `start-local-models.sh`
   (launchd, host port). Sandbox first.
4. **Route (Option A)** — point the parsing-router `model_url` at the host server; ingest a PDF with
   `CFG_DYNORG_DOCLING_CONVERSION_ENABLED=true` → `converted_file.evidence_tree` non-NULL → grounded
   chunking mints real `anchor_id`s.
5. **Prod** — launchd daemon on a labeled Mac + k8s Service; benchmark per-page latency; size
   `max_replicas`. Optionally migrate to Option B.
6. **(Later) Option B** — model-backend external-utility runtime + non-LLM `/trigger` provider.

## Why not …

- **Torch/EasyOCR docling in a container** — CPU-only on the fleet (no Metal), multi-GB, slow. This is
  the status quo we're replacing.
- **Ray inside k8s containers** — no Metal passthrough; defeats the purpose. Ray must be a host process.
- **Plain FastAPI host server (no Ray)** — fine as an MVP (it's today's `qwen3-asr-server.py` shape),
  but Ray Serve adds batching/replicas/autoscaling/health that the performance requirement wants. The
  server code (mlx_vlm → DocTags → export_to_dict) is identical; Ray Serve wraps it.

## References

- Model: https://huggingface.co/ibm-granite/granite-docling-258M-mlx ·
  https://www.ibm.com/new/announcements/granite-docling-end-to-end-document-conversion
- Ray Serve: https://docs.ray.io/en/latest/serve/index.html (deployments, `@serve.batch`, autoscaling,
  fractional resources)
- Backend seams: `services/model/pkg/llm/runtime/{staticruntime,rayruntime}/`,
  `services/artifact/pkg/pipeline/preset/pipelines/parsing-router/v1.2.0/recipe.yaml`,
  `services/artifact/pkg/pipeline/client.go` (the "Docling Model" parser)
- Producer contract: `backend/docs/artifact/m7-w1b-producer-wiring.md` (Part 1) · ADR-0020
  (`backend/docs/adr/dynamic-organization/0020-extraction-producer-architecture.md`)
- Host-server pattern: `buckle/scripts/sandbox/qwen3-asr-server.py`, `shared-model-servers.sh`
