# pylint: skip-file
import os
import glob
import time
import subprocess
import logging

from openai import OpenAI

from instill.helpers.ray_config import instill_deployment, InstillDeployable
from instill.helpers import (
    parse_task_chat_to_chat_input,
    construct_task_chat_output,
)

logger = logging.getLogger(__name__)

GGUF_REPO = "unsloth/GLM-5-GGUF"
GGUF_QUANT = os.environ.get("GGUF_QUANT", "UD-Q2_K_XL")
WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "/tmp/ray/GLM-5-GGUF")
LLAMA_SERVER_PORT = os.environ.get("LLAMA_SERVER_PORT", "8081")
LLAMA_SERVER_BIN = os.environ.get("LLAMA_SERVER_BIN", "/home/ray/bin/llama-server")


def _ensure_weights() -> str:
    """Download GGUF weights from HuggingFace if not already cached."""
    pattern = os.path.join(WEIGHTS_DIR, GGUF_QUANT, "*.gguf")
    shards = sorted(glob.glob(pattern))
    if shards:
        return shards[0]

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=GGUF_REPO,
        allow_patterns=f"*{GGUF_QUANT}*",
        local_dir=WEIGHTS_DIR,
    )

    shards = sorted(glob.glob(pattern))
    if not shards:
        raise FileNotFoundError(
            f"No .gguf files found in {WEIGHTS_DIR}/{GGUF_QUANT} after download"
        )
    return shards[0]


def _wait_for_server(base_url: str, timeout: int = 600, poll: float = 2.0):
    """Block until the llama-server /health endpoint returns 200."""
    import urllib.request
    import urllib.error

    health_url = base_url.rstrip("/").replace("/v1", "") + "/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = urllib.request.urlopen(health_url, timeout=5)
            if resp.status == 200:
                logger.info("llama-server is healthy at %s", health_url)
                return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(poll)
    raise TimeoutError(
        f"llama-server did not become healthy within {timeout}s"
    )


@instill_deployment
class GLM5:
    def __init__(self):
        model_path = _ensure_weights()

        cmd = [
            LLAMA_SERVER_BIN,
            "--model", model_path,
            "--n-gpu-layers", os.environ.get("N_GPU_LAYERS", "-1"),
            "--split-mode", os.environ.get("SPLIT_MODE", "layer"),
            "--ctx-size", os.environ.get("N_CTX", "16384"),
            "--flash-attn", "on",
            "--batch-size", os.environ.get("BATCH_SIZE", "512"),
            "--ubatch-size", os.environ.get("UBATCH_SIZE", "512"),
            "--cache-type-k", os.environ.get("KV_CACHE_K", "q4_0"),
            "--cache-type-v", os.environ.get("KV_CACHE_V", "q4_0"),
            "--parallel", os.environ.get("PARALLEL_REQUESTS", "2"),
            "--cont-batching",
            "--reasoning", "off",
            "--port", LLAMA_SERVER_PORT,
        ]

        logger.info("Starting llama-server: %s", " ".join(cmd))
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        self._openai_base_url = f"http://127.0.0.1:{LLAMA_SERVER_PORT}/v1"
        _wait_for_server(self._openai_base_url)

        self._client = OpenAI(
            base_url=self._openai_base_url,
            api_key="sk-no-key-required",
        )
        logger.info("llama-server ready, proxying via OpenAI client")

    async def __call__(self, request):
        chat_inputs = await parse_task_chat_to_chat_input(request=request)

        finish_reasons = []
        indexes = []
        created = []
        messages = []
        completion_tokens = []
        prompt_tokens = []
        total_tokens = []

        for inp in chat_inputs:
            extra_kwargs = {}
            if inp.seed and inp.seed != 0:
                extra_kwargs["seed"] = inp.seed

            result = self._client.chat.completions.create(
                model=self._client.models.list().data[0].id,
                messages=inp.messages,
                max_tokens=inp.max_tokens,
                temperature=inp.temperature,
                top_p=inp.top_p,
                **extra_kwargs,
            )

            finish_reasons_per_seq = []
            indexes_per_seq = []
            created_per_seq = []
            messages_per_seq = []

            for choice in result.choices:
                messages_per_seq.append(
                    {
                        "content": choice.message.content or "",
                        "role": "assistant",
                    }
                )
                finish_reasons_per_seq.append(choice.finish_reason or "stop")
                indexes_per_seq.append(choice.index)
                created_per_seq.append(result.created)

            finish_reasons.append(finish_reasons_per_seq)
            indexes.append(indexes_per_seq)
            created.append(created_per_seq)
            messages.append(messages_per_seq)

            if result.usage:
                completion_tokens.append(result.usage.completion_tokens or 0)
                prompt_tokens.append(result.usage.prompt_tokens or 0)
                total_tokens.append(result.usage.total_tokens or 0)
            else:
                completion_tokens.append(0)
                prompt_tokens.append(0)
                total_tokens.append(0)

        return construct_task_chat_output(
            request=request,
            finish_reasons=finish_reasons,
            indexes=indexes,
            created_timestamps=created,
            messages=messages,
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            total_tokens=total_tokens,
        )

    def __del__(self):
        if hasattr(self, "_proc") and self._proc.poll() is None:
            logger.info("Terminating llama-server (pid=%d)", self._proc.pid)
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()


entrypoint = InstillDeployable(GLM5).get_deployment_handle()
