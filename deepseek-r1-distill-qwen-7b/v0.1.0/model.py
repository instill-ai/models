import time
from mlc_llm import AsyncMLCEngine

from instill.helpers.ray_config import instill_deployment, InstillDeployable
from instill.helpers import (
    parse_task_chat_to_chat_input,
    construct_task_chat_output,
)


@instill_deployment
class DeepSeekR1DistillQwen7B:
    def __init__(self):
        self.model = "DeepSeek-R1-Distill-Qwen-7B-q4f16_1-MLC"
        self.engine = AsyncMLCEngine(self.model)

    async def __call__(self, request):
        conversation_inputs = await parse_task_chat_to_chat_input(
            request=request
        )

        finish_reasons = []
        indexes = []
        created = []
        messages = []
        for i, inp in enumerate(conversation_inputs):
            print(inp.messages)

            # Use MLC-LLM chat completions API
            response = await self.engine.chat.completions.create(
                messages=inp.messages,
                model=self.model,
                max_tokens=inp.max_tokens,
                temperature=inp.temperature,
                top_p=inp.top_p,
                stream=False,
            )

            output = response.choices[0].message.content.strip()

            messages.append([{"content": output, "role": "assistant"}])
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


entrypoint = InstillDeployable(DeepSeekR1DistillQwen7B).get_deployment_handle()
