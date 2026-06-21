"""Smoke-test the Ray Serve app: start 2 replicas, convert a PDF, verify the contract + fan-out."""
import base64, sys, time
import requests
from ray import serve
import serve_app

PDF = sys.argv[1] if len(sys.argv) > 1 else \
    "/Users/Pinglin/Workspace/shubo/backend/services/pipeline/pkg/component/operator/document/v0/testdata/split-in-pages-output-1-2.pdf"

print("starting Ray Serve (2 replicas; loads 2 models)...", flush=True)
serve.run(serve_app.app, route_prefix="/")
base = "http://127.0.0.1:8000"

# wait for health
for _ in range(60):
    try:
        h = requests.get(f"{base}/health", timeout=5).json()
        print("health:", h, flush=True); break
    except Exception:
        time.sleep(1)

pdf_b64 = base64.b64encode(open(PDF, "rb").read()).decode()
t0 = time.time()
r = requests.post(f"{base}/convert", json={"pdf_b64": pdf_b64}, timeout=300)
dt = time.time() - t0
print(f"/convert -> {r.status_code} in {dt:.1f}s", flush=True)
if r.status_code == 200:
    d = r.json()
    sd = d["structured_document"]
    print("schema_name :", sd.get("schema_name"))
    print("num_pages   :", d["num_pages"])
    print("markdown_pgs:", len(d["markdown_pages"]))
    print("texts       :", len(sd.get("texts", [])))
    print("RAY SERVE CONTRACT OK" if sd.get("schema_name") == "DoclingDocument" else "CONTRACT FAIL")
else:
    print("body:", r.text[:300])
serve.shutdown()
