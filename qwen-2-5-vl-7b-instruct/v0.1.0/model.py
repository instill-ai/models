# pylint: skip-file
import torch
import time
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from instill.helpers.ray_config import instill_deployment, InstillDeployable
from instill.helpers import (
    parse_task_chat_to_multimodal_chat_input,
    construct_task_chat_output,
)


@instill_deployment
class Qwen25VL:
    def __init__(self):
        model_id = "Qwen2.5-VL-7B-Instruct"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
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

            # placeholder image
            # minimum factor is 28
            if len(image_list) == 0:
                message_list[0]["content"].insert(0, {"type": "image"})
                image_list.append(Image.new("RGB", (28, 28)))

            prompt = self.processor.apply_chat_template(
                message_list,
                tokenize=False,
                add_generation_prompt=True,
                add_vision_id=True,
            )

            inputs = self.processor(
                text=[prompt],
                images=[image_list],
                padding=True,
                return_tensors="pt",
            ).to("cuda")

            generated_ids = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=inp.temperature,
                top_p=inp.top_p,
                max_new_tokens=inp.max_tokens,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            outputs = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
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


entrypoint = InstillDeployable(Qwen25VL).get_deployment_handle()
