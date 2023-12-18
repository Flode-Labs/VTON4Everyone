
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler
from diffusers.utils import load_image
from controlnet_aux import OpenposeDetector


init_image = load_image("./test/model.png")
mask_image = load_image("./test/mask.png")
ip_image = load_image("./test/garment.png") # The clothing image
prompt = "a woman wearing a dress, best quality, high quality"


# Load pre-trained controlnet models
controlnet = [
    ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16),
    ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float16),
]

# Initialize the pipeline
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", controlnet=controlnet, torch_dtype=torch.float16
)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# Move pipeline to GPU
pipe.to("cuda")
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus_sd15.bin")


# Function to calculate new image sizes
def new_sizes(image):
    original_width, original_height = image.size

    width_ratio = 768 / original_width
    height_ratio = 768 / original_height

    resize_ratio = min(width_ratio, height_ratio)

    new_width = int(original_width * resize_ratio)
    new_height = int(original_height * resize_ratio)

    closest_width = ((new_width + 7) // 8) * 8
    closest_height = ((new_height + 7) // 8) * 8

    return closest_width, closest_height


# Function to create inpaint condition
def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image


# Load and resize initial image and mask
resize_width, resize_height = new_sizes(init_image)
init_image = init_image.resize((resize_width, resize_height), Image.LANCZOS)
mask_image = mask_image.convert("L").resize((resize_width, resize_height))
inpainting_control_image = make_inpaint_condition(init_image, mask_image)

# Load and resize openpose image
openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
openpose_image = openpose(init_image)
openpose_image = openpose_image.resize((resize_width, resize_height))


# Specify images and prompts for the controlnet
images = [inpainting_control_image, openpose_image]
negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, disfigured"

# Set IP adapter scale
pipe.set_ip_adapter_scale(0.85)

# Invoke the pipeline with specified images and prompts
images = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=init_image,
    mask_image=mask_image,
    ip_adapter_image=ip_image,
    control_image=images,
    controlnet_conditioning_scale=[0.5, 1.0],
    num_inference_steps=30,
    strength=0.7,
    guidance_scale=7,
    eta=1,
).images