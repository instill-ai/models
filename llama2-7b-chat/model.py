# pylint: skip-file
import os

TORCH_GPU_DEVICE_ID = 0
os.environ["CUDA_VISIBLE_DEVICES"] = f"{TORCH_GPU_DEVICE_ID}"

import time
import torch
import transformers
from transformers import LlamaTokenizer

from instill.helpers.ray_config import instill_deployment, InstillDeployable
from instill.helpers import (
    parse_task_chat_to_chat_input,
    construct_task_chat_output,
)


@instill_deployment
class Llama2Chat:
    def __init__(self):
        self.pipeline = transformers.pipeline(
            "text-generation",
            model="Llama-2-7b-chat-hf",
            torch_dtype=torch.float16,
            device_map="auto",
        )

    async def __call__(self, request):
        conversation_inputs = await parse_task_chat_to_chat_input(request=request)

        if len(conversation_inputs) > 1:
            raise Exception("this model does not support batch input")

        conv = self.pipeline.tokenizer.apply_chat_template(
            conversation_inputs[0].messages, tokenize=False, add_generation_prompt=True
        )

        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        sequences = self.pipeline(
            conv,
            do_sample=True,
            top_p=conversation_inputs[0].top_p,
            temperature=conversation_inputs[0].temperature,
            num_return_sequences=conversation_inputs[0].n,
            eos_token_id=terminators,
            max_length=conversation_inputs[0].max_tokens,
        )

        text_outputs = []
        for seq in sequences:
            generated_text = (
                seq["generated_text"].split("[/INST]")[-1].strip().encode("utf-8")
            )
            text_outputs.append(generated_text)

        finish_reasons = [[]]
        indexes = [[]]
        created = [[]]
        for i in range(len(text_outputs)):
            finish_reasons[0].append("length")
            indexes[0].append(i)
            created[0].append(int(time.time()))

        return construct_task_chat_output(
            request=request,
            finish_reasons=finish_reasons,
            indexes=indexes,
            created_timestamps=created,
            messages=[text_outputs],
        )


entrypoint = InstillDeployable(Llama2Chat).get_deployment_handle()
