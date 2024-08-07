import time
import torch
from transformers import pipeline

from instill.helpers.ray_config import instill_deployment, InstillDeployable
from instill.helpers import (
    parse_task_chat_to_chat_input,
    construct_task_chat_output,
)


@instill_deployment
class TinyLlama:
    def __init__(self):
        self.pipeline = pipeline(
            "text-generation",
            model="tinyllama",
            torch_dtype=torch.float32,
            device_map="cpu",
        )

    async def __call__(self, request):
        conversation_inputs = await parse_task_chat_to_chat_input(request=request)

        finish_reasons = []
        indexes = []
        created = []
        messages = []
        for i, inp in enumerate(conversation_inputs):
            print(inp.messages)
            prompt = self.pipeline.tokenizer.apply_chat_template(
                inp.messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # inference
            sequences = self.pipeline(
                prompt,
                max_new_tokens=inp.max_tokens,
                do_sample=True,
                temperature=inp.temperature,
                top_p=inp.top_p,
            )

            output = (
                sequences[0]["generated_text"]
                .split("<|assistant|>\n")[-1]
                .strip()
                .encode("utf-8")
            )

            messages.append([{"content": str(output), "role": "assistant"}])
            finish_reasons.append(["length"])
            indexes.append([i])
            created.append([int(time.time())])

        return construct_task_chat_output(
            request=request,
            finish_reasons=finish_reasons,
            indexes=indexes,
            messages=messages,
            created_timestamps=created,
        )


entrypoint = InstillDeployable(TinyLlama).get_deployment_handle()
