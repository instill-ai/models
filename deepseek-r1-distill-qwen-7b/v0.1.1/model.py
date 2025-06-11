from typing import List
from vllm import LLM, SamplingParams, RequestOutput
from instill.helpers.ray_config import instill_deployment, InstillDeployable
from instill.helpers import (
    parse_task_chat_to_chat_input,
    construct_task_chat_output,
)


@instill_deployment
class DeepSeekR1DistillQwen7B:
    def __init__(self):
        self.model = LLM(
            model="DeepSeek-R1-Distill-Qwen-7B",
            dtype="float16",
            max_model_len=8192,
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
                temperature=inp.temperature,
                top_p=inp.top_p,
                n=inp.n,
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
                messages_per_seq.append({
                    "content": seq.outputs[0].text,
                    "role": "assistant"
                })
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


entrypoint = InstillDeployable(
    DeepSeekR1DistillQwen7B
).get_deployment_handle()
