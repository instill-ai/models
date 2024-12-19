# pylint: skip-file
import base64
import random
import numpy as np
import torch

from io import BytesIO
from diffusers import StableDiffusion3Pipeline

from instill.helpers.ray_config import instill_deployment, InstillDeployable
from instill.helpers import (
    parse_task_text_to_image_input,
    construct_task_text_to_image_output,
)

torch.set_float32_matmul_precision("high")
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True

# for quantization
# from transformers import T5EncoderModel, BitsAndBytesConfig
# quantization_config = BitsAndBytesConfig(load_in_8bit=True)


@instill_deployment
class Sd35Large:
    def __init__(self):
        model_id = "stable-diffusion-3.5-large"
        # quantized T5
        # text_encoder = T5EncoderModel.from_pretrained(
        #     model_id,
        #     subfolder="text_encoder_3",
        #     quantization_config=quantization_config,
        # )

        self.model = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            # text_encoder_3=None,
            # tokenizer_3=None,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        self.model.set_progress_bar_config(disable=True)
        # compiled components
        self.model.transformer.to(memory_format=torch.channels_last)
        self.model.vae.to(memory_format=torch.channels_last)

        self.model.transformer = torch.compile(
            self.model.transformer, mode="max-autotune", fullgraph=True
        )
        self.model.vae.decode = torch.compile(
            self.model.vae.decode, mode="max-autotune", fullgraph=True
        )

        self.ratio_map = {
            "9:21": {
                "width": 672,
                "height": 1600,
            },
            "9:16": {
                "width": 768,
                "height": 1344,
            },
            "2:3": {
                "width": 768,
                "height": 1152,
            },
            "4:5": {
                "width": 912,
                "height": 1136,
            },
            "5:4": {
                "width": 1136,
                "height": 912,
            },
            "3:2": {
                "width": 1152,
                "height": 768,
            },
            "16:9": {
                "width": 1344,
                "height": 768,
            },
            "21:9": {
                "width": 1600,
                "height": 672,
            },
            "1:1": {
                "width": 1024,
                "height": 1024,
            },
        }

        # Warm Up
        prompt = "a photo of a cat holding a sign that says hello world"
        for ratio_dict in self.ratio_map.values():
            _ = self.model(
                prompt=prompt,
                height=ratio_dict["height"],
                width=ratio_dict["width"],
            )

    async def __call__(self, request):
        inputs = await parse_task_text_to_image_input(request=request)

        finish_reasons = []
        images = []
        for inp in inputs:
            if inp.seed > 0:
                random.seed(inp.seed)
                np.random.seed(inp.seed)

            generated_images = self.model(
                prompt=inp.prompt,
                negative_prompt=inp.negative_prompt,
                num_images_per_prompt=inp.n,
                max_sequence_length=512,
                num_inference_steps=45,
                height=self.ratio_map[inp.aspect_ratio]["height"],
                width=self.ratio_map[inp.aspect_ratio]["width"],
                guidance_scale=4.5,
            ).images

            imgs = []
            finishes = []
            for img in generated_images:
                buff = BytesIO()
                img.save(buff, format="JPEG", keep_rgb=True, quality=95)
                imgs.append(base64.b64encode(buff.getvalue()).decode())
                finishes.append("success")

            images.append(imgs)
            finish_reasons.append(finishes)

        return construct_task_text_to_image_output(
            request=request,
            finish_reasons=finish_reasons,
            images=images,
        )


entrypoint = InstillDeployable(Sd35Large).get_deployment_handle()
