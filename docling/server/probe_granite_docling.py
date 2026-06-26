"""Probe: default Docling MLX model -> DoclingDocument-compatible contract.

Proves the MLX model emits the M7 `structured_document` contract
(schema_name == "DoclingDocument"), which the Go consumer `docdoc.FromDoclingExport` parses.
"""
import json
import os
import sys
import time

from PIL import Image

from granite_docling import DEFAULT_MODEL, GraniteDocling

MODEL = os.environ.get("SHUBO_DOCLING_MODEL", DEFAULT_MODEL)
IMAGE = sys.argv[1] if len(sys.argv) > 1 else "test_page.png"
OUT = sys.argv[2] if len(sys.argv) > 2 else "docling_tree.json"

print(f"loading {MODEL} (cold load on Metal)...", flush=True)
t0 = time.time()
engine = GraniteDocling(MODEL).load()
print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

pil = Image.open(IMAGE).convert("RGB")
print(f"page image: {pil.size}", flush=True)

print("generating Docling contract...", flush=True)
t0 = time.time()
result = engine.convert([pil])
dt = time.time() - t0
tree = result["structured_document"]
markdown = "\n\n".join(result["markdown_pages"])
print(f"  generated {len(markdown)} markdown chars in {dt:.1f}s", flush=True)

with open(OUT, "w") as f:
    json.dump(tree, f)
print("--- RESULT ---")
print("model       :", result.get("model"))
print("family      :", result.get("model_family"))
print("schema_name :", tree.get("schema_name"))
print("texts       :", len(tree.get("texts", [])))
print("tables      :", len(tree.get("tables", [])))
print("pictures    :", len(tree.get("pictures", [])))
print("markdown    :", len(markdown), "chars")
print("tree written:", OUT)
print("CONTRACT OK" if tree.get("schema_name") == "DoclingDocument" else "CONTRACT FAIL")
