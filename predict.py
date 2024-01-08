# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from pipeline_sd import load_pipeline, preprocess_image
import torch
import random
from typing import List
from PIL import Image

BASE_SIZE = {
    "sd": 1024,
    "1024": 1024,
}


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.pipe = load_pipeline()

    def predict(
        self,
        style_image: Path = Input(description="Style image"),
        content_image: Path = Input(description="Content image"),
        prompt: str = Input(description="Prompt", default=""),
        negative_prompt: str = Input(description="Negative prompt", default=""),
        seed: int = Input(description="Seed", default=-1),
        steps: int = Input(description="Steps", default=30),
        num_images_per_prompt: int = Input(
            description="Number of images per prompt", default=1, le=4, ge=1
        ),
    ) -> List[Path]:
        if seed == -1:
            seed = random.randint(0, 1e9)
        content_image = Image.open(content_image).convert("RGB")
        o_width, o_height = content_image.size
        style_image = Image.open(style_image).convert("RGB")
        generator = torch.Generator(device="cuda").manual_seed(seed)
        control_images = preprocess_image(content_image)
        if o_width > o_height:
            width = BASE_SIZE["sd"]
            height = int(BASE_SIZE["sd"] * (o_height / o_width))
            height = (height // 8) * 8
        else:
            height = BASE_SIZE["sd"]
            width = int(BASE_SIZE["sd"] * (o_width / o_height))
            width = (width // 8) * 8
        images = self.pipe(
            prompt=prompt,
            generator=generator,
            negative_prompt=negative_prompt,
            controlnet_conditioning_scale=0.8,
            image=control_images,
            ip_adapter_image=style_image,
            num_inference_steps=steps,
            num_images_per_prompt=num_images_per_prompt,
            width=width,
            height=height,
        ).images
        paths = []
        for i, image in enumerate(images):
            image.save("/tmp/out_{}.png".format(i))
            paths.append(Path("/tmp/out_{}.png".format(i)))
        return paths
