# pylint: skip-file
import time
import random
import torch

import numpy as np

from instill.helpers.ray_config import instill_deployment, InstillDeployable
from instill.helpers import (
    parse_task_chat_to_multimodal_chat_input,
    construct_task_chat_output,
)

import transformers
from transformers import AutoTokenizer
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from llava.conversation import conv_templates, Conversation, SeparatorStyle
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.mm_utils import process_images, tokenizer_image_token


@instill_deployment
class Llava:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "llava-v1.6-vicuna-13b", use_fast=False
        )

        self.model = LlavaLlamaForCausalLM.from_pretrained(
            "llava-v1.6-vicuna-13b",
            low_cpu_mem_usage=True,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        vision_tower = self.model.get_vision_tower()
        vision_tower.load_model()
        self.image_processor = vision_tower.to(
            device="cuda", dtype=torch.float16
        ).image_processor

    async def __call__(self, request):
        chat_inputs = await parse_task_chat_to_multimodal_chat_input(request=request)

        finish_reasons = []
        indexes = []
        created = []
        messages = []
        for inp in chat_inputs:

            random.seed(inp.seed)
            np.random.seed(inp.seed)

            conv = conv_templates["llava_v1"].copy()
            images = []
            for img_list, message in zip(inp.prompt_images, inp.messages):
                content = message["content"]

                conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + content)
                conv.append_message(conv.roles[1], None)

                images.extend(img_list)

            prompt = conv.get_prompt()

            image_tensors = process_images(
                images, self.image_processor, self.model.config
            ).to(self.model.device, dtype=torch.float16)

            input_ids = (
                tokenizer_image_token(
                    prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                .unsqueeze(0)
                .cuda()
            )

            output_ids = self.model.generate(
                input_ids,
                images=image_tensors,
                do_sample=True,
                temperature=inp.temperature,
                top_p=inp.top_p,
                max_new_tokens=inp.max_tokens,
                use_cache=False,
            )

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (
                (input_ids != output_ids[:, :input_token_len]).sum().item()
            )
            if n_diff_input_output > 0:
                print(
                    f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
                )

            outputs = self.tokenizer.batch_decode(
                output_ids[:, input_token_len:], skip_special_tokens=True
            )[0].strip()

            finish_reasons.append(["length"])
            indexes.append([0])
            created.append([int(time.time())])
            messages.append([{"content": outputs, "role": "assistant"}])

        return construct_task_chat_output(
            request=request,
            finish_reasons=finish_reasons,
            indexes=indexes,
            created_timestamps=created,
            messages=messages,
        )


entrypoint = InstillDeployable(Llava).get_deployment_handle()
