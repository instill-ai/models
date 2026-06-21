"""Increment-1 probe: granite-docling-258M-mlx -> DocTags -> DoclingDocument.export_to_dict().

Proves the MLX model emits the exact M7 `structured_document` contract
(schema_name == "DoclingDocument"), which the Go consumer `docdoc.FromDoclingExport` parses.
"""
import json
import sys
import time

from PIL import Image
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

MODEL = "ibm-granite/granite-docling-258M-mlx"
IMAGE = sys.argv[1] if len(sys.argv) > 1 else "test_page.png"
OUT = sys.argv[2] if len(sys.argv) > 2 else "granite_tree.json"

print(f"loading {MODEL} (cold load on Metal)...", flush=True)
t0 = time.time()
loaded = load(MODEL)
# mlx_vlm.load returns (model, processor) or (model, processor, config) across versions
if len(loaded) == 3:
    model, processor, config = loaded
else:
    model, processor = loaded
    config = getattr(model, "config", None)
print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

pil = Image.open(IMAGE).convert("RGB")
print(f"page image: {pil.size}", flush=True)

prompt_text = "Convert this page to docling."
try:
    prompt = apply_chat_template(processor, config, prompt_text, num_images=1)
except Exception as e:
    print(f"  apply_chat_template fallback ({e})", flush=True)
    prompt = prompt_text

print("generating DocTags...", flush=True)
t0 = time.time()
result = generate(
    model, processor, prompt, image=[pil],
    max_tokens=4096, temperature=0.0, verbose=False,
)
doctags = result.text if hasattr(result, "text") else str(result)
dt = time.time() - t0
print(f"  generated {len(doctags)} chars in {dt:.1f}s", flush=True)

# DocTags -> DoclingDocument -> the canonical tree
from docling_core.types.doc.document import DocTagsDocument, DoclingDocument

dtd = DocTagsDocument.from_doctags_and_image_pairs([doctags], [pil])
doc = DoclingDocument.load_from_doctags(dtd)
tree = doc.export_to_dict()
markdown = doc.export_to_markdown()

json.dump(tree, open(OUT, "w"))
print("--- RESULT ---")
print("schema_name :", tree.get("schema_name"))
print("texts       :", len(tree.get("texts", [])))
print("tables      :", len(tree.get("tables", [])))
print("pictures    :", len(tree.get("pictures", [])))
print("markdown    :", len(markdown), "chars")
print("tree written:", OUT)
print("CONTRACT OK" if tree.get("schema_name") == "DoclingDocument" else "CONTRACT FAIL")
