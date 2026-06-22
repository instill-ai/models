# Shubo model benchmarks

Apple-Silicon (mlx) benchmarks for choosing Shubo's local model defaults.

## `qwen36_vlm_bench.py` — Qwen3.6 VLM quantization sweep

Picks the ship quantization for the **default local VLM** (replacing Gemma 4) and the **W1b
document-parsing classifier**. For each `mlx-community/Qwen3.6-*` quant it reports load time, peak
Metal memory, and warm latency / throughput on the classifier task (Standard / Docling / VLM) and
on DocTags extraction.

```bash
# from the fleet host that will serve the model (numbers must reflect the prod Apple-Silicon target)
~/.shubo-llm/mlx/bin/python qwen36_vlm_bench.py --json qwen36.json
# a single quant:
~/.shubo-llm/mlx/bin/python qwen36_vlm_bench.py --models mlx-community/Qwen3.6-35B-A3B-4bit
```

Models download from the HF hub on first use (cached under `~/.cache/huggingface`). The full quant
matrix (4/5/6/8-bit, DWQ, mxfp4, the 27B and 35B-A3B families) is large — run the sweep on the
fleet, not a laptop, and extend `DEFAULT_MODELS` as the eval deepens. The 35B-A3B (MoE, 3B active)
4-bit is the current ship candidate; this harness is how we confirm the quality/latency/memory
trade-off before flipping the buckle default (`SHUBO_LOCAL_VLM_MODEL`).
