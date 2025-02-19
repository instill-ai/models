# pylint: skip-file
import time
from mlc_llm import MLCEngine
from instill.helpers.ray_config import instill_deployment, InstillDeployable
from instill.helpers import (
    parse_task_chat_to_chat_input,
    construct_task_chat_output,
)


@instill_deployment
class Llama32Instruct:
    def __init__(self):
        self.model_id = "Llama-3.2-3B-Instruct-q4f16_0-MLC"
        self.model = MLCEngine(self.model_id)

    async def __call__(self, request):
        chat_inputs = await parse_task_chat_to_chat_input(request=request)

        finish_reasons = []
        indexes = []
        created = []
        messages = []
        for inp in chat_inputs:
            resp = self.model.chat.completions.create(
                messages=inp.messages,
                model=self.model_id,
                max_tokens=inp.max_tokens,
                n=inp.n,
                temperature=inp.temperature,
                top_p=inp.top_p,
                seed=inp.seed,
                stream=inp.stream,
            )

            finish_reasons_per_seq = []
            indexes_per_seq = []
            created_per_seq = []
            messages_per_seq = []
            for choice in resp.choices:
                messages_per_seq.append(choice.message)
                finish_reasons_per_seq.append(choice.finish_reason)
                indexes_per_seq.append(choice.index)
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


entrypoint = InstillDeployable(Llama32Instruct).get_deployment_handle()
