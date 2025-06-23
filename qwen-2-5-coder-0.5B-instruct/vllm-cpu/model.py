"""This model is used to serve the Qwen2.5-Coder-1.5B-Instruct model using vllm."""

from typing import List

from vllm import LLM, SamplingParams, RequestOutput

from instill.helpers.ray_config import instill_deployment, InstillDeployable
from instill.helpers import (
    parse_task_chat_to_chat_input,
    construct_task_chat_output,
)

@instill_deployment
class Qwen25Coder:
    """Qwen2.5-Coder-0.5B-Instruct model using vllm."""

    def __init__(self):
        self.model = LLM(
            model="Qwen2.5-Coder-0.5B-Instruct",
            dtype="float16")

    async def __call__(self, request):
        chat_inputs = await parse_task_chat_to_chat_input(request=request)

        results = {"finish_reasons": [], "indexes": [], "created": [], "messages": []}

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

            seq_data = {"finish_reasons": [], "indexes": [], "created": [], "messages": []}
            for i, seq in enumerate(sequences):
                seq_data["messages"].append(
                    {"content": seq.outputs[0].text, "role": "assistant"}
                )
                seq_data["finish_reasons"].append(seq.outputs[0].finish_reason)
                seq_data["indexes"].append(i)
                seq_data["created"].append(int(seq.metrics.finished_time))

            results["finish_reasons"].append([seq_data["finish_reasons"][0]])
            results["indexes"].append([seq_data["indexes"][0]])
            results["created"].append([seq_data["created"][0]])
            results["messages"].append([seq_data["messages"][0]])

        return construct_task_chat_output(
            request=request,
            finish_reasons=results["finish_reasons"],
            indexes=results["indexes"],
            created_timestamps=results["created"],
            messages=results["messages"],
        )


entrypoint = InstillDeployable(Qwen25Coder).get_deployment_handle()
