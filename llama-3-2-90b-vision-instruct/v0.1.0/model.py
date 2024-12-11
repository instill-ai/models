# pylint: skip-file
from typing import List
from vllm import LLM, SamplingParams, SamplingParams, RequestOutput
from instill.helpers.ray_config import instill_deployment, InstillDeployable
from instill.helpers import (
    parse_task_chat_to_multimodal_chat_input,
    construct_task_chat_output,
)


@instill_deployment
class Llama32Vision:
    def __init__(self):
        self.model = LLM(
            model="Llama-3.2-90B-Vision-Instruct-GGUF",
            max_num_seqs=1,
            gpu_memory_utilization=0.99,
            enforce_eager=True,
        )

    async def __call__(self, request):
        chat_inputs = await parse_task_chat_to_multimodal_chat_input(request=request)

        finish_reasons = []
        indexes = []
        created = []
        messages = []
        for inp in chat_inputs:
            prompt = ""
            image = None
            for i, (message, images) in enumerate(zip(inp.messages, inp.prompt_images)):
                if len(images) == 1:
                    prompt += f"<|image|><|begin_of_text|>{message['content']}"
                    image = images[0]
                elif len(images) == 0:
                    prompt += f"<|begin_of_text|>{message['content']}"
                else:
                    raise Exception(
                        "This model accepts 0 or 1 image in each conversation"
                    )

            params = SamplingParams(
                max_tokens=inp.max_tokens,
                n=inp.n,
                temperature=inp.temperature,
                top_p=inp.top_p,
            )

            if inp.seed > 0:
                params.seed = inp.seed

            sequences: List[RequestOutput] = self.model.generate(
                {
                    "prompt": prompt,
                    "multi_modal_data": {
                        image,
                    },
                }
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


entrypoint = InstillDeployable(Llama32Vision).get_deployment_handle()
