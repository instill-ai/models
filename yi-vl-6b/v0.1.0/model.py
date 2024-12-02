# pylint: skip-file
import os
import time
import random
import numpy as np
import torch

from llava.conversation import conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    expand2square,
    get_model_name_from_path,
    load_pretrained_model,
    tokenizer_image_token,
)
from llava.model.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX

from instill.helpers.ray_config import instill_deployment, InstillDeployable
from instill.helpers import (
    parse_task_chat_to_multimodal_chat_input,
    construct_task_chat_output,
)

setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


@instill_deployment
class YIVL:
    def __init__(self):
        model_path = os.path.expanduser("~/Yi-VL-6B")
        get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = (
            load_pretrained_model(model_path)
        )
        self.model = self.model.to(dtype=torch.bfloat16)

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

            conv = conv_templates["mm_default"].copy()
            image = None
            for message, images in zip(inp.messages, inp.prompt_images):
                if message["role"] == "user":
                    conv.append_message(
                        conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{message['content']}"
                    )
                elif message["role"] == "assistant":
                    conv.append_message(conv.roles[1], message["content"])

                if len(images) > 1:
                    raise Exception(
                        "This model only support 0 or 1 image per conversation"
                    )
                elif len(images) == 1:
                    image = images[0]

            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = (
                tokenizer_image_token(
                    prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                .unsqueeze(0)
                .cuda()
            )

            if getattr(self.model.config, "image_aspect_ratio", None) == "pad":
                image = expand2square(
                    image, tuple(int(x * 255) for x in self.image_processor.image_mean)
                )
            image_tensor = self.image_processor.preprocess(image, return_tensors="pt")[
                "pixel_values"
            ][0]

            stop_str = conv.sep
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(
                keywords, self.tokenizer, input_ids
            )

            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).to(dtype=torch.bfloat16).cuda(),
                    do_sample=True,
                    temperature=inp.temperature,
                    top_p=inp.top_p,
                    stopping_criteria=[stopping_criteria],
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
            )[0]
            outputs = outputs.strip()

            if outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)]
            outputs = outputs.strip()

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


entrypoint = InstillDeployable(YIVL).get_deployment_handle()
