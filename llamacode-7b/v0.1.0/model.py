# pylint: skip-file
import os

import time
import torch
import transformers

from instill.helpers.ray_config import instill_deployment, InstillDeployable
from instill.helpers import (
    parse_task_completion_to_completion_input,
    construct_task_completion_output,
)


@instill_deployment
class CodeLlama:
    def __init__(self):
        self.pipeline = transformers.pipeline(
            "text-generation",
            model="CodeLlama-7b-hf",
            torch_dtype=torch.float16,
            device_map="cuda",
        )

    async def __call__(self, request):
        completion_inputs = await parse_task_completion_to_completion_input(
            request=request
        )

        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        finish_reasons = []
        indexes = []
        created = []
        contents = []
        for inp in completion_inputs:

            sequences = self.pipeline(
                inp.prompt,
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
            contents_per_seq = []
            for i, seq in enumerate(sequences):
                generated_text = seq["generated_text"].strip()
                contents_per_seq.append(generated_text)
                finish_reasons_per_seq.append("length")
                indexes_per_seq.append(i)
                created_per_seq.append(int(time.time()))

            finish_reasons.append(finish_reasons_per_seq)
            indexes.append(indexes_per_seq)
            created.append(created_per_seq)
            contents.append(contents_per_seq)

        return construct_task_completion_output(
            request=request,
            finish_reasons=finish_reasons,
            indexes=indexes,
            created_timestamps=created,
            contents=contents,
        )


entrypoint = InstillDeployable(CodeLlama).get_deployment_handle()
