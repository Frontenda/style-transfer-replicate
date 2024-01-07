# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from pipeline_sd import load_pipeline, preprocess_image
import torch
import random
from typing import List

BASE_SIZE = {
    "sd": 512,
    "1024": 768,
}


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.pipe = load_pipeline()

    def predict(
        self,
        style_image: Path = Input(description="Style image"),
        content_image: Path = Input(description="Content image"),
        prompt: str = Input(description="Prompt", type=str, default=""),
        negative_prompt: str = Input(
            description="Negative prompt", type=str, default=""
        ),
        seed: int = Input(description="Seed", type=int, default=-1),
        steps: int = Input(description="Steps", type=int, default=30),
        num_images_per_prompt: int = Input(
            description="Number of images per prompt", type=int, default=1, max=4, min=1
        ),
    ) -> List[Path]:
        if seed == -1:
            seed = random.randint(0, 1e9)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        images = self.pipe(
            prompt=prompt,
            generator=generator,
            negative_prompt=negative_prompt,
            controlnet_conditioning_scale=0.8,
            image=content_image,
            ip_adapter_image=style_image,
            num_inference_steps=steps,
            num_images_per_prompt=num_images_per_prompt,
        ).images
        paths = []
        for i, image in enumerate(images):
            image.save("/tmp/out_{}.png".format(i))
            paths.append(Path("/tmp/out_{}.png".format(i)))
        return paths
