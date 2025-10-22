import torch
import sys
import os
sys.path.append('./diffusers/src')
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

from PIL import Image
import matplotlib.pyplot as plt

model_id = "stabilityai/stable-diffusion-2-1-base"

scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
model = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)

model = model.to("cuda")

image = model(
    prompt="a photo of a cat holding a sign that says hello world",
    negative_prompt="",
    # num_inference_steps=28,
    height=512,
    width=512,
    guidance_scale=7.0,
    # generator=g,
).images[0]

# image.save("sd3_hello_world.png")
plt.imshow(image)
plt.axis("off")  # Hide axes for a cleaner display
plt.show()