# pylint: skip-file
import time
import torch
import transformers

from instill.helpers.ray_config import instill_deployment, InstillDeployable
from instill.helpers import (
    parse_task_chat_to_chat_input,
    construct_task_chat_output,
)


@instill_deployment
class Zephyr:
    def __init__(self):
        self.pipeline = transformers.pipeline(
            "text-generation",
            model="zephyr-7b-alpha",
            torch_dtype=torch.float16,
            device_map="cuda",
        )

    async def __call__(self, request):
        chat_inputs = await parse_task_chat_to_chat_input(request=request)

        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        finish_reasons = []
        indexes = []
        created = []
        messages = []
        for inp in chat_inputs:
            conv = self.pipeline.tokenizer.apply_chat_template(
                inp.messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            sequences = self.pipeline(
                conv,
                do_sample=True,
                temperature=inp.temperature,
                top_p=inp.top_p,
                num_return_sequences=inp.n,
                eos_token_id=terminators,
                max_new_tokens=inp.max_tokens,
            )

            finish_reasons_per_seq = []
            indexes_per_seq = []
            created_per_seq = []
            messages_per_seq = []
            for i, seq in enumerate(sequences):
                generated_text = seq["generated_text"].strip().encode("utf-8")
                messages_per_seq.append(
                    {"content": str(generated_text), "role": "assistant"}
                )
                finish_reasons_per_seq.append("length")
                indexes_per_seq.append(i)
                created_per_seq.append(int(time.time()))

            finish_reasons.append(finish_reasons_per_seq)
            indexes.append(indexes_per_seq)
            created.append(created_per_seq)
            messages.append(messages_per_seq)

        return construct_task_chat_output(
            request=request,
            finish_reasons=finish_reasons,
            indexes=indexes,
            created_timestamps=created,
            messages=messages,
        )


entrypoint = InstillDeployable(Zephyr).get_deployment_handle()
