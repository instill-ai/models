# pylint: skip-file
import random
import numpy as np
import torch

from diffusers import StableDiffusion3Pipeline
from instill.helpers.ray_config import instill_deployment, InstillDeployable
from instill.helpers import (
    parse_task_text_to_image_input,
    construct_task_text_to_image_output,
)


@instill_deployment
class Sd3:
    def __init__(self):
        self.pipeline = StableDiffusion3Pipeline.from_pretrained(
            "stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16
        ).to("cuda")

    async def __call__(self, request):
        inputs = await parse_task_text_to_image_input(request=request)

        random.seed(inputs.seed)
        np.random.seed(inputs.seed)

        finish_reasons = []
        images = []
        for inp in inputs:

            generated_images = self.pipeline(
                inp.prompt,
                negative_prompt=inp.negative_prompt,
                num_inference_steps=28,
                guidance_scale=7.0,
                num_images_per_prompt=inp.n,
            )

            imgs = []
            finishes = []
            for img in generated_images:
                imgs.append(img)
                finishes.append("success")

            images.append(imgs)
            finish_reasons.append(finishes)

        return construct_task_text_to_image_output(
            request=request,
            finish_reasons=finish_reasons,
            images=images,
        )


entrypoint = InstillDeployable(Sd3).get_deployment_handle()
