#!/usr/bin/env python3
"""Benchmark Qwen3.6 VLM quantizations for the W1b document-parsing classifier + DocTags.

For each mlx-community Qwen3.6 quantization, measures load time, peak Metal memory, and warm
inference latency / throughput on (a) the parsing-router classification task (Standard / Docling /
VLM) and (b) DocTags extraction — so we can pick the ship quant for Shubo's default local VLM
(replacing Gemma 4) and the W1b document classifier. Apple-Silicon / mlx-vlm only.

Usage:
  python qwen36_vlm_bench.py                       # default quant matrix
  python qwen36_vlm_bench.py --models mlx-community/Qwen3.6-35B-A3B-4bit
  python qwen36_vlm_bench.py --image PAGE.png --max-tokens 2048 --json out.json

Each model is downloaded from the HF hub on first use (cached). Run on the fleet host that will
serve the model so the numbers reflect the production Apple-Silicon target.
"""
import argparse
import gc
import json
import time
from pathlib import Path

from PIL import Image

import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

# The 35B-A3B (MoE, 3B active) ship candidate + the dense 27B, at the quants worth comparing.
# Extend with 5/6bit, DWQ, mxfp4 etc. as the eval deepens.
DEFAULT_MODELS = [
    "mlx-community/Qwen3.6-35B-A3B-4bit",
    "mlx-community/Qwen3.6-35B-A3B-8bit",
    "mlx-community/Qwen3.6-27B-4bit",
    "mlx-community/Qwen3.6-27B-8bit",
]

# Mirrors the parsing-router classifier intent (see the W1b parsing-router recipe).
CLASSIFY_PROMPT = (
    "You are a document-parsing router. Look at this document page and pick the best "
    "conversion strategy, replying with EXACTLY one of:\n"
    "- Standard Document Operator (clean, text-only, simple layout)\n"
    "- Docling Model (complex layout: tables, figures, formulas, multi-column)\n"
    "- Visual Language Model Pipeline (scanned / handwritten / image-only)\n"
    "Reply with only the chosen strategy."
)
DOCTAGS_PROMPT = "Convert this page to docling."


def _peak_mem_gb() -> float:
    for owner in (mx, getattr(mx, "metal", None)):
        fn = getattr(owner, "get_peak_memory", None)
        if fn:
            try:
                return fn() / 1e9
            except Exception:
                pass
    return float("nan")


def _reset_peak_mem() -> None:
    for owner in (mx, getattr(mx, "metal", None)):
        fn = getattr(owner, "reset_peak_memory", None)
        if fn:
            try:
                fn()
                return
            except Exception:
                pass


def _clear_cache() -> None:
    for owner in (mx, getattr(mx, "metal", None)):
        fn = getattr(owner, "clear_cache", None)
        if fn:
            try:
                fn()
                return
            except Exception:
                pass


def _gen(model, processor, prompt, img, max_tokens):
    t0 = time.time()
    out = generate(model, processor, prompt, image=[img], max_tokens=max_tokens,
                   temperature=0.0, verbose=False)
    dt = time.time() - t0
    text = out.text if hasattr(out, "text") else str(out)
    try:
        ntok = len(processor.tokenizer.encode(text))
    except Exception:
        ntok = max(1, len(text) // 4)
    return dt, ntok, text


def bench(model_id: str, img, max_tokens: int) -> dict:
    r = {"model": model_id}
    _reset_peak_mem()

    t0 = time.time()
    loaded = load(model_id)
    if len(loaded) == 3:
        model, processor, config = loaded
    else:
        model, processor = loaded
        config = getattr(model, "config", None)
    r["load_s"] = round(time.time() - t0, 2)

    cls_prompt = apply_chat_template(processor, config, CLASSIFY_PROMPT, num_images=1)
    dtg_prompt = apply_chat_template(processor, config, DOCTAGS_PROMPT, num_images=1)

    # Warm-up: the first call pays the graph-compile cost — not timed.
    _gen(model, processor, cls_prompt, img, 16)

    dt, _, text = _gen(model, processor, cls_prompt, img, 48)
    r["classify_s"] = round(dt, 2)
    r["classify_out"] = " ".join(text.split())[:80]

    dt, ntok, _ = _gen(model, processor, dtg_prompt, img, max_tokens)
    r["doctags_s"] = round(dt, 2)
    r["doctags_tok"] = ntok
    r["doctags_tok_s"] = round(ntok / dt, 1) if dt else 0.0

    r["peak_mem_gb"] = round(_peak_mem_gb(), 2)

    del model, processor
    gc.collect()
    _clear_cache()
    return r


def main() -> None:
    here = Path(__file__).resolve().parent
    default_img = here.parent / "docling" / "server" / "test_page.png"

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--image", default=str(default_img), help="document page image")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--json", default="", help="optional path to write the full result JSON")
    args = ap.parse_args()

    img = Image.open(args.image).convert("RGB")
    print(f"image: {args.image} {img.size}   models: {len(args.models)}", flush=True)

    rows = []
    for m in args.models:
        print(f"\n=== {m} ===", flush=True)
        try:
            r = bench(m, img, args.max_tokens)
        except Exception as e:  # one bad model must not abort the sweep
            r = {"model": m, "error": str(e)[:200]}
        rows.append(r)
        print("  ", {k: v for k, v in r.items() if k != "model"}, flush=True)

    hdr = (f"{'model':44s} {'load_s':>7s} {'mem_gb':>7s} {'cls_s':>6s} "
           f"{'dtg_s':>6s} {'tok/s':>7s}  classification")
    print("\n=== Qwen3.6 VLM quantization benchmark ===")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if "error" in r:
            print(f"{r['model']:44s}  ERROR: {r['error']}")
            continue
        print(f"{r['model']:44s} {r['load_s']:7.2f} {r['peak_mem_gb']:7.2f} "
              f"{r['classify_s']:6.2f} {r['doctags_s']:6.2f} {r['doctags_tok_s']:7.1f}  "
              f"{r.get('classify_out', '')[:44]}")

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
