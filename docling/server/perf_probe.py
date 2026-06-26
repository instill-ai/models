"""Measure where the Docling MLX model's per-page time goes, and the resolution lever."""
import os
import sys
import time

from PIL import Image

from granite_docling import DEFAULT_MODEL, GraniteDocling

MODEL = os.environ.get("SHUBO_DOCLING_MODEL", DEFAULT_MODEL)
img_path = sys.argv[1] if len(sys.argv) > 1 else "test_page.png"

print(f"load {MODEL} ...", flush=True)
t0 = time.time()
engine = GraniteDocling(MODEL).load()
print(f"  loaded {time.time()-t0:.2f}s", flush=True)

base = Image.open(img_path).convert("RGB")


def run(img, label):
    # warm + timed
    t0 = time.time()
    text = engine.page_to_text(img)
    dt = time.time() - t0
    # token count via the tokenizer
    try:
        ntok = len(engine._processor.tokenizer.encode(text))
    except Exception:
        ntok = len(text) // 4
    print(f"[{label:14s}] size={str(img.size):14s} gen={dt:5.2f}s  out~{ntok:4d}tok  "
          f"{ntok/dt:6.1f} tok/s  chars={len(text)}", flush=True)
    return dt, ntok

print("\n=== warm-up (first call pays graph compile) ===", flush=True)
run(base, "warmup")
print("\n=== resolution sweep (warm) ===", flush=True)
for scale in (1.0, 0.75, 0.5):
    w, h = base.size
    img = base if scale == 1.0 else base.resize((int(w*scale), int(h*scale)))
    run(img, f"scale={scale}")
