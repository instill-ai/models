import torch

from transformers import pipeline

from instill.helpers.ray_config import instill_deployment, InstillDeployable
from instill.helpers import (
    parse_task_text_generation_chat_to_conversation_input,
    construct_task_text_generation_chat_output,
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

    async def Trigger(self, request):
        conversation_inputs = parse_task_text_generation_chat_to_conversation_input(
            request=request
        )

        outputs = []
        for inp in conversation_inputs:
            prompt = self.pipeline.tokenizer.apply_chat_template(
                inp.conversation,
                tokenize=False,
                add_generation_prompt=True,
            )

            # inference
            sequences = self.pipeline(
                prompt,
                max_new_tokens=inp.max_new_tokens,
                do_sample=True,
                temperature=inp.temperature,
                top_k=inp.top_k,
                top_p=0.95,
            )

            output = (
                sequences[0]["generated_text"]
                .split("<|assistant|>\n")[-1]
                .strip()
                .encode("utf-8")
            )

            outputs.append(output)

        return construct_task_text_generation_chat_output(outputs)


entrypoint = InstillDeployable(TinyLlama).get_deployment_handle()
