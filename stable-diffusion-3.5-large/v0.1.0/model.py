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

        # compiled components
        self.model = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            # text_encoder_3=None,
            # tokenizer_3=None,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        self.model.set_progress_bar_config(disable=True)
        self.model.transformer.to(memory_format=torch.channels_last)
        self.model.vae.to(memory_format=torch.channels_last)

        self.model.transformer = torch.compile(
            self.model.transformer, mode="max-autotune", fullgraph=True
        )
        self.model.vae.decode = torch.compile(
            self.model.vae.decode, mode="max-autotune", fullgraph=True
        )

        # Warm Up
        prompt = "a photo of a cat holding a sign that says hello world"
        for _ in range(3):
            _ = self.model(prompt=prompt, generator=torch.manual_seed(1))

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
                num_inference_steps=28,
                height=1024,
                width=1024,
                guidance_scale=7.0,
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
