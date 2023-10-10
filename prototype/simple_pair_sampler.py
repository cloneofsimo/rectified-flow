import random

import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler


def sample_n(
    num_images=8 * 16,
):
    pipe = StableDiffusionPipeline.from_pretrained(
        "nota-ai/bk-sdm-tiny", torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    prompt = [
        "a photo of a {}".format("cat" if random.random() < 0.5 else "dog")
        for _ in range(num_images)
    ]

    bs = 8

    prompt_batched = [prompt[i : i + bs] for i in range(0, len(prompt), bs)]

    initial_positions = []
    end_positions = []

    for promptset in prompt_batched:
        latents = (
            torch.randn(len(promptset), 4, 64, 64)
            .to(pipe.unet.device)
            .to(pipe.unet.dtype)
        )

        latents_out = pipe(
            promptset, guidance_scale=5.0, latents=latents, output_type="latent"
        ).images

        initial_positions.append(latents)
        end_positions.append(latents_out)

    torch.save(torch.cat(initial_positions, dim=0), "initial_positions.pt")
    torch.save(torch.cat(end_positions, dim=0), "end_positions.pt")


if __name__ == "__main__":
    sample_n()
