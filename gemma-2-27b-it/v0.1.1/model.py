# pylint: skip-file
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from instill.helpers.ray_config import instill_deployment, InstillDeployable
from instill.helpers import (
    parse_task_chat_to_chat_input,
    construct_task_chat_output,
)


@instill_deployment
class Gemma2:
    def __init__(self):
        model_id = "gemma-2-27b-it-bnb-4bit"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    async def __call__(self, request):
        chat_inputs = await parse_task_chat_to_chat_input(request=request)

        finish_reasons = []
        indexes = []
        created = []
        messages = []
        for inp in chat_inputs:

            # gemma does not support system role
            if inp.messages[0]["role"] == "system":
                inp.messages.pop(0)

            input_text = self.tokenizer.apply_chat_template(
                inp.messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            input_ids = self.tokenizer.encode(
                input_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to("cuda")

            output_ids = self.model.generate(
                input_ids=input_ids,
                do_sample=True,
                temperature=inp.temperature,
                top_p=inp.top_p,
                num_return_sequences=inp.n,
                max_new_tokens=inp.max_tokens,
            )

            output_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, output_ids)
            ]

            outputs = self.tokenizer.batch_decode(
                output_ids_trimmed,
                skip_special_tokens=True,
            )

            finish_reasons_per_seq = []
            indexes_per_seq = []
            created_per_seq = []
            messages_per_seq = []
            for i, out in enumerate(outputs):
                messages_per_seq.append({"content": out, "role": "assistant"})
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


entrypoint = InstillDeployable(Gemma2).get_deployment_handle()
