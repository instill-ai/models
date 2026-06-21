"""Measure where granite-docling's per-page time goes, and the resolution lever."""
import sys, time
from PIL import Image
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

MODEL = "ibm-granite/granite-docling-258M-mlx"
img_path = sys.argv[1] if len(sys.argv) > 1 else "test_page.png"

print(f"load {MODEL} ...", flush=True)
t0 = time.time()
loaded = load(MODEL)
if len(loaded) == 3:
    model, processor, config = loaded
else:
    model, processor = loaded
    config = getattr(model, "config", None)
print(f"  loaded {time.time()-t0:.2f}s", flush=True)

base = Image.open(img_path).convert("RGB")
prompt = apply_chat_template(processor, config, "Convert this page to docling.", num_images=1)

def run(img, label):
    # warm + timed
    t0 = time.time()
    out = generate(model, processor, prompt, image=[img], max_tokens=4096,
                   temperature=0.0, verbose=False)
    dt = time.time() - t0
    text = out.text if hasattr(out, "text") else str(out)
    # token count via the tokenizer
    try:
        ntok = len(processor.tokenizer.encode(text))
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
