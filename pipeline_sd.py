from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch
from controlnet_aux.processor import Processor


def load_pipeline():
    controlnet = [
        ControlNetModel.from_single_file(
            "lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16
        ),
        ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_mlsd", torch_dtype=torch.float16
        ),
    ]
    pipe = StableDiffusionControlNetPipeline.from_single_file(
        "/sd.safetensors",
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )

    pipe.load_ip_adapter(
        "h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin"
    )
    pipe.to("cuda")
    return pipe


def preprocess_image(image):
    processors = [
        Processor("depth_midas"),
        Processor("mlsd"),
    ]
    control_images = [processor(image, to_pil=True) for processor in processors]
    return control_images
