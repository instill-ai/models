# pylint: skip-file
import torch
import time
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from instill.helpers.ray_config import instill_deployment, InstillDeployable
from instill.helpers import (
    parse_task_chat_to_multimodal_chat_input,
    construct_task_chat_output,
)


@instill_deployment
class Qwen2VL:
    def __init__(self):
        model_id = "Qwen2-VL-72B-Instruct-AWQ"
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    async def __call__(self, request):
        chat_inputs = await parse_task_chat_to_multimodal_chat_input(request=request)

        finish_reasons = []
        indexes = []
        created = []
        messages = []
        for inp in chat_inputs:

            message_list = []
            image_list = []
            for message, images in zip(inp.messages, inp.prompt_images):
                d = {
                    "role": message["role"],
                    "content": [
                        {
                            "type": "text",
                            "text": message["content"],
                        }
                    ],
                }
                for _ in range(len(images)):
                    d["content"].insert(0, {"type": "image"})
                message_list.append(d)
                image_list.extend(images)

            prompt = self.processor.apply_chat_template(
                message_list,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = self.processor(
                text=[prompt],
                images=image_list,
                padding=True,
                return_tensors="pt",
            ).to("cuda")

            generated_ids = self.model.generate(
                **inputs,
                temperature=inp.temperature,
                top_p=inp.top_p,
                max_new_tokens=inp.max_tokens,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            finish_reasons.append(["length"])
            indexes.append([0])
            created.append([int(time.time())])
            messages.append([{"content": output_text, "role": "assistant"}])

        return construct_task_chat_output(
            request=request,
            finish_reasons=finish_reasons,
            indexes=indexes,
            created_timestamps=created,
            messages=messages,
        )


entrypoint = InstillDeployable(Qwen2VL).get_deployment_handle()
