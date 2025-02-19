# pylint: skip-file
import time
from mlc_llm import MLCEngine
from mlc_llm.serve.config import EngineConfig
from instill.helpers.ray_config import instill_deployment, InstillDeployable
from instill.helpers import (
    parse_task_chat_to_chat_input,
    construct_task_chat_output,
)


@instill_deployment
class Qwen25Coder:
    def __init__(self):
        self.model_id = "Qwen2.5-Coder-32B-Instruct-q4f16_1-MLC"
        # ref: https://github.com/mlc-ai/mlc-llm/blob/60597635082f6888fb50847b27e112102167543a/python/mlc_llm/serve/config.py#L9
        engine_config = EngineConfig(
            max_num_sequence=1,
            gpu_memory_utilization=0.99,
            prefill_chunk_size=1024,
        )

        self.model = MLCEngine(
            model=self.model_id,
            mode="server",
            engine_config=engine_config,
        )

    async def __call__(self, request):
        chat_inputs = await parse_task_chat_to_chat_input(request=request)

        finish_reasons = []
        indexes = []
        created = []
        messages = []
        for inp in chat_inputs:

            total_tps = 0
            total_tokens = 0
            total_elapsed_time = 0
            for _ in range(100):
                start = time.time()
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
                stop = time.time()

                finish_reasons_per_seq = []
                indexes_per_seq = []
                created_per_seq = []
                messages_per_seq = []
                for choice in resp.choices:
                    messages_per_seq.append(choice.message)
                    finish_reasons_per_seq.append(choice.finish_reason)
                    indexes_per_seq.append(choice.index)
                    created_per_seq.append(int(time.time()))

                tps = resp.usage.completion_tokens / (stop - start)

                total_tokens += resp.usage.completion_tokens
                total_elapsed_time += stop - start
                total_tps += tps

            print("====================================")
            print(f"total rounds: 100")
            print(f"total output tokens: {total_tokens}")
            print(f"total elapsed time: {total_elapsed_time}")
            print(f"average TPS: {total_tps/100}")
            print("====================================")

            finish_reasons.append([finish_reasons_per_seq[0]])
            indexes.append([indexes_per_seq[0]])
            created.append([created_per_seq[0]])
            messages.append([messages_per_seq[0]])

        return construct_task_chat_output(
            request=request,
            finish_reasons=finish_reasons,
            indexes=indexes,
            created_timestamps=created,
            messages=messages,
        )


entrypoint = InstillDeployable(Qwen25Coder).get_deployment_handle()
