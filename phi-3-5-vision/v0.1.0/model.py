# pylint: skip-file
import time
import random
import numpy as np

from transformers import AutoProcessor, AutoModelForCausalLM
from instill.helpers.ray_config import instill_deployment, InstillDeployable
from instill.helpers import (
    parse_task_chat_to_multimodal_chat_input,
    construct_task_chat_output,
)


@instill_deployment
class PhiVision:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            "Phi-3.5-vision-instruct",
            device_map="cuda",
            trust_remote_code=True,
            torch_dtype="auto",
            _attn_implementation="eager",
        )

        self.processor = AutoProcessor.from_pretrained(
            "Phi-3.5-vision-instruct",
            trust_remote_code=True,
            num_crops=4,
        )

    async def __call__(self, request):
        chat_inputs = await parse_task_chat_to_multimodal_chat_input(request=request)

        finish_reasons = []
        indexes = []
        created = []
        messages = []
        # image_ids must start from 1, and must be continuous int, e.g. [1, 2, 3], cannot be [0].
        img_idx = 1
        for inp in chat_inputs:
            if inp.seed > 0:
                random.seed(inp.seed)
                np.random.seed(inp.seed)

            placeholder = ""
            for i in range(len(inp.messages)):
                if inp.messages[i]["role"] == "user":
                    for _ in range(len(inp.prompt_images[i])):
                        placeholder += f"<|image_{img_idx}|>\n"
                        img_idx += 1

                inp.messages[i]["content"] = placeholder + inp.messages[i]["content"]

            prompt = self.processor.tokenizer.apply_chat_template(
                inp.messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            images = [i for row in inp.prompt_images for i in row]
            inputs = self.processor(prompt, images, return_tensors="pt").to("cuda:0")

            generation_args = {
                "max_new_tokens": inp.max_tokens,
                "temperature": inp.temperature,
                "do_sample": False,
            }

            generate_ids = self.model.generate(
                **inputs,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                **generation_args,
            )

            generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
            response = self.processor.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            finish_reasons.append(["length"])
            indexes.append([0])
            created.append([int(time.time())])
            messages.append([{"content": response, "role": "assistant"}])

        return construct_task_chat_output(
            request=request,
            finish_reasons=finish_reasons,
            indexes=indexes,
            created_timestamps=created,
            messages=messages,
        )


entrypoint = InstillDeployable(PhiVision).get_deployment_handle()
