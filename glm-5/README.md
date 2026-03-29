# GLM-5

## Introduction

GLM-5 is a 744B-parameter Mixture-of-Experts (MoE) model with 40B active parameters, targeting complex systems engineering and long-horizon agentic tasks. It integrates DeepSeek Sparse Attention (DSA) to reduce deployment cost while preserving long-context capacity (up to 202K tokens).

GLM-5 achieves best-in-class performance among open-source models on reasoning, coding, and agentic tasks (SWE-bench Verified: 77.8%, Terminal-Bench 2.0: 56.2%).

- [Technical Report](https://arxiv.org/abs/2602.15763)
- [HuggingFace Model](https://huggingface.co/zai-org/GLM-5)
- [HuggingFace GGUF (Unsloth)](https://huggingface.co/unsloth/GLM-5-GGUF)
- [Unsloth Deployment Guide](https://unsloth.ai/docs/models/glm-5)
- [GitHub](https://github.com/zai-org/GLM-5)

| Task Type                                                  | Description                                                                                 |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| [Chat](https://www.instill-ai.dev/docs/model/ai-task#chat) | A task to generate conversational style text output base on single or multi-modality input. |

## Compatibility Matrix

| Instill Core Version | Python SDK Version |
| -------------------- | ------------------ |
| >= v0.51.0           | >= v0.18.0         |

> **Note:** Always ensure that you are using compatible versions to avoid unexpected issues.

## Architecture

This model uses a **llama-server subprocess** managed inside the `@instill_deployment` class. The native C++ `llama-server` binary is built from source in `Dockerfile.llamacpp` with A100-specific optimizations (SM_80, Flash Attention). At runtime, `model.py` starts `llama-server` as a subprocess in `__init__` and proxies requests to its OpenAI-compatible API via the `openai` Python client in `__call__`.

```
Ray Serve Replica (4x A100)
┌──────────────────────────────────────────────────┐
│  model.py (@instill_deployment)                  │
│    __init__: subprocess.Popen(llama-server)       │
│    __call__: openai.Client → localhost:8081/v1    │
│                                                  │
│  llama-server (native C++)                       │
│    row split · flash-attn · KV q4_0 cache        │
│    4x A100 80GB                                  │
└──────────────────────────────────────────────────┘
         ↑
  model-backend-ee (gRPC)
```

### Why llama-server subprocess instead of llama-cpp-python?

- **No Python GIL**: Native C++ inference avoids Python binding overhead, yielding ~1.8x faster prompt evaluation.
- **Continuous batching**: `llama-server` supports `--cont-batching` and `--parallel` for concurrent request handling.
- **Source-built for A100**: Compiled with `-DCMAKE_CUDA_ARCHITECTURES=80` and `-DGGML_FLASH_ATTN=ON` for optimal kernel performance.
- **Thinking mode control**: `--reasoning off` disables GLM-5's default chain-of-thought mode, ensuring all generated tokens are useful content.

### Why GGUF instead of vLLM?

GLM-5 is a 744B MoE model. Even in FP8, the weights are ~702 GB — far exceeding 4x A100 80GB capacity. llama.cpp with GGUF quantization (UD-Q2_K_XL at 281 GB) fits entirely in GPU memory with room for KV cache.

## Hardware Requirements

### Production: 4x A100 80GB (a2-ultragpu-4g)

| Setting                   | Value       | Rationale                                          |
| ------------------------- | ----------- | -------------------------------------------------- |
| `--n-gpu-layers`          | -1          | Offload all layers to GPU                          |
| `--split-mode`            | row         | Row split — distributes MoE experts across GPUs    |
| `--ctx-size`              | 16384       | Context window size                                |
| `--flash-attn`            | on          | Flash attention for memory-efficient KV cache      |
| `--cache-type-k/v`        | q4_0        | KV cache quantization — saves ~40 GB VRAM          |
| `--batch-size`            | 512         | Optimized prompt processing batch                  |
| `--ubatch-size`           | 512         | Micro-batch size                                   |
| `--parallel`              | 2           | Handle 2 concurrent requests                       |
| `--cont-batching`         | enabled     | Continuous batching for throughput                  |
| `--reasoning`             | off         | Disable chain-of-thought thinking mode             |
| GGUF quant                | UD-Q2_K_XL  | 281 GB — fits in 320 GB VRAM with headroom for KV  |

### Memory Budget

```
4x A100 80GB = 320 GB VRAM total
GGUF weights (UD-Q2_K_XL): ~281 GB
KV cache (q4_0):            ~20 GB
Remaining headroom:         ~19 GB
```

## Performance (4x A100 80GB, llama-server b8563)

| Metric               | Value         |
| -------------------- | ------------- |
| Prompt eval          | ~30-34 tok/s  |
| Generation           | ~11.1 tok/s   |
| Thinking tokens      | 0 (disabled)  |
| Concurrent requests  | 2             |

All generated tokens are useful content — no chain-of-thought overhead.

## Preparation

Follow [this](../README.md) guide to get your custom model up and running! But before you do that, please read through the following sections to have all the necessary files ready.

### Install Python SDK

```bash
pip install instill-sdk=={version}
```

### Model Weights

Weights are downloaded at runtime from HuggingFace inside `__init__`. No need to pre-download for production. For local testing or POC, download manually:

```bash
pip install huggingface_hub hf_transfer
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download unsloth/GLM-5-GGUF \
  --include "*UD-Q2_K_XL*" --local-dir ./GLM-5-GGUF
```

### Environment Variables

| Variable             | Default                       | Description                              |
| -------------------- | ----------------------------- | ---------------------------------------- |
| `GGUF_QUANT`         | `UD-Q2_K_XL`                  | GGUF quantization variant to use         |
| `WEIGHTS_DIR`        | `/home/ray/GLM-5-GGUF`        | Directory for GGUF weight files          |
| `N_GPU_LAYERS`       | `-1`                          | Number of layers to offload to GPU       |
| `N_CTX`              | `16384`                       | Context window size                      |
| `LLAMA_SERVER_PORT`  | `8081`                        | Port for the llama-server subprocess     |
| `LLAMA_SERVER_BIN`   | `/usr/local/bin/llama-server`  | Path to the llama-server binary          |
| `BATCH_SIZE`         | `512`                         | Prompt processing batch size             |
| `UBATCH_SIZE`        | `512`                         | Micro-batch size                         |
| `KV_CACHE_K`         | `q4_0`                        | KV cache key quantization type           |
| `KV_CACHE_V`         | `q4_0`                        | KV cache value quantization type         |
| `PARALLEL_REQUESTS`  | `2`                           | Number of parallel request slots         |

## Production Deployment

Production uses a lightweight Docker image (~3-5 GB, no weights baked in). The image contains the pre-built `llama-server` binary. Weights are downloaded from HuggingFace at container startup inside `__init__`.

### Build and push

```bash
cd v0.1.0
instill build admin/glm-5 -a amd64
instill push admin/glm-5 -u <registry>
```

### Deploy

Deploy via model-backend-ee admin API. The backend resolves `glm-5` to 4 GPUs (configured in `InstillAIModelGPUAssignmentMap`), and PUTs the Ray Serve config to the Ray cluster.

### Startup Time

First startup takes ~15-30 min (281 GB download within GCP + model load into GPU memory). Subsequent restarts on the same pod use cached weights and start in ~2 min.

## Test model image

After building the model image, test locally before pushing:

```bash
instill run admin/glm-5 -g -i '{"prompt": "how much do you know about python?"}'
```

Input payload format:

```json
{
  "prompt": "..."
}
```

Flags supported by `instill run`:

- `-t, --tag`: tag for the model image, default to `latest`
- `-g, --gpu`: pass through GPU from host into container
- `-i, --input`: input in JSON format
