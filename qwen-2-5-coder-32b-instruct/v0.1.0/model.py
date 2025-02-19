# pylint: skip-file
import time
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
        self.model = LLM(
            model="Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf",
            max_model_len=12000,
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

            total_tps = 0
            total_tokens = 0
            total_elapsed_time = 0
            for _ in range(100):
                start = time.time()
                sequences: List[RequestOutput] = self.model.chat(
                    messages=inp.messages,
                    sampling_params=params,
                    use_tqdm=False,
                )
                stop = time.time()

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

                    token_ids = seq.outputs[0].token_ids
                    tps = len(token_ids) / (stop - start)

                    total_tokens += len(token_ids)
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
