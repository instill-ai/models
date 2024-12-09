# pylint: skip-file
from pathlib import Path
from typing import List
from vllm import LLM, SamplingParams, SamplingParams, RequestOutput
from instill.helpers.ray_config import instill_deployment, InstillDeployable
from instill.helpers import (
    parse_task_chat_to_chat_input,
    construct_task_chat_output,
)


@instill_deployment
class Qwen25Coder:
    def __init__(self):
        files = [
            f.name for f in Path.cwd().iterdir() if f.is_file() and ".gguf" in f.name
        ]
        self.model = LLM(
            model=files[0],
            max_num_seqs=1,
            gpu_memory_utilization=0.99,
            enable_chunked_prefill=True,
            enforce_eager=True,
        )

    async def __call__(self, request):
        chat_inputs = await parse_task_chat_to_chat_input(request=request)

        finish_reasons = []
        indexes = []
        created = []
        messages = []
        for inp in chat_inputs:
            params = SamplingParams(
                max_tokens=inp.max_tokens,
                n=inp.n,
                temperature=inp.temperature,
                top_p=inp.top_p,
            )

            if inp.seed != 0:
                params.seed = inp.seed

            sequences: List[RequestOutput] = self.model.chat(
                messages=inp.messages,
                sampling_params=params,
                use_tqdm=False,
            )

            finish_reasons_per_seq = []
            indexes_per_seq = []
            created_per_seq = []
            messages_per_seq = []
            for i, seq in enumerate(sequences):

                messages_per_seq.append(
                    {"content": seq.outputs[0].text, "role": "assistant"}
                )
                finish_reasons_per_seq.append(seq.outputs[0].finish_reason)
                indexes_per_seq.append(i)
                created_per_seq.append(int(seq.metrics.finished_time))

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


entrypoint = InstillDeployable(Qwen25Coder).get_deployment_handle()
