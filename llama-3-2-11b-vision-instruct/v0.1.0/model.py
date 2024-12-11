# pylint: skip-file
import time
import random
import numpy as np
import torch
from transformers import AutoProcessor, MllamaForConditionalGeneration
from instill.helpers.ray_config import instill_deployment, InstillDeployable
from instill.helpers import (
    parse_task_chat_to_multimodal_chat_input,
    construct_task_chat_output,
)


@instill_deployment
class Llama32Vision:
    def __init__(self):
        self.model = MllamaForConditionalGeneration.from_pretrained(
            "Llama-3.2-11B-Vision-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )

        self.processor = AutoProcessor.from_pretrained(
            "Llama-3.2-11B-Vision-Instruct",
        )

    async def __call__(self, request):
        chat_inputs = await parse_task_chat_to_multimodal_chat_input(request=request)

        finish_reasons = []
        indexes = []
        created = []
        messages = []
        for inp in chat_inputs:
            if inp.seed > 0:
                random.seed(inp.seed)
                np.random.seed(inp.seed)

            prompt_messages = []
            image = None
            for message, images in zip(inp.messages, inp.prompt_images):
                if len(images) != 1:
                    raise Exception("this model accetps exactly 1 image")
                image = images[0]
                prompt_messages.append(
                    {
                        "role": message["role"],
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": message["content"]},
                        ],
                    }
                )

            input_text = self.processor.apply_chat_template(
                prompt_messages,
                add_generation_prompt=True,
            )

            inputs = self.processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to("cuda:0")

            generate_ids = self.model.generate(
                **inputs,
                max_new_tokens=inp.max_tokens,
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


entrypoint = InstillDeployable(Llama32Vision).get_deployment_handle()
