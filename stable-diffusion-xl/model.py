# pylint: skip-file
import random
import numpy as np
import torch

from diffusers import DiffusionPipeline
from instill.helpers.ray_config import instill_deployment, InstillDeployable
from instill.helpers import (
    parse_task_text_to_image_input,
    construct_task_text_to_image_output,
)


@instill_deployment
class SdXL:
    def __init__(self):
        self.base = DiffusionPipeline.from_pretrained(
            "stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        ).to("cuda")
        self.base.unet = torch.compile(self.base.unet, mode="reduce-overhead", fullgraph=True)
        self.refiner = DiffusionPipeline.from_pretrained(
            "stable-diffusion-xl-refiner-1.0",
            text_encoder_2=self.base.text_encoder_2,
            vae=self.base.vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        ).to("cuda")
        self.refiner.unet = torch.compile(self.base.unet, mode="reduce-overhead", fullgraph=True)


    async def __call__(self, request):
        inputs = await parse_task_text_to_image_input(request=request)

        n_steps = 40
        high_noise_frac = 0.8 

        finish_reasons = []
        images = []
        for inp in inputs:
            random.seed(inp.seed)
            np.random.seed(inp.seed)

            images = self.base(
                prompt=inp.prompt,
                num_inference_steps=n_steps,
                denoising_end=high_noise_frac,
                output_type="latent",
            ).images

            generated_images = self.refiner(
                prompt=prompt,
                num_inference_steps=n_steps,
                denoising_end=high_noise_frac,
                image=images,
            ).images

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


entrypoint = InstillDeployable(SdXL).get_deployment_handle()
