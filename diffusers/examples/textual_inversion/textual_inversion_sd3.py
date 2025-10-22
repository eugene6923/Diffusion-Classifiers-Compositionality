#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Adapted for Stable Diffusion 3 by modifying the SDXL script.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import logging
import math
import os
import random
import shutil
import copy # Added import
from pathlib import Path
from typing import List, Union, Optional
def login_wandb():
    # if ARNAS_USES:
    # api_key = "c743143a34e5f47f22c8d98469cfb33387573055"
    api_key = "local-addff5ac065ec24ca8d463a7188760e623c531d7"
    api_key = "local-9e68aa2925ebe8ee4e638fc0e33a0d47922c20f6"
    api_key = "local-1d3a330785de0daa6c5e0dd683723eabe9d80d35"

    # api_key = "local-9e68aa2925ebe8ee4e638fc0e33a0d47922c20f6"
    # add timeout period longer os
    import os

    os.environ["WANDB__SERVICE_WAIT"] = "30000"
    api_key = "9876773f72d210923a9694ae701f8d71c9d30381"

    os.environ["WANDB_API_KEY"] = api_key
    # os.environ["WANDB_BASE_URL"] = "http://185.80.128.108:8080"
    # os.environ["WANDB_BASE_URL"] = "http://176.118.198.12:8080"
login_wandb()

import numpy as np
import PIL
import safetensors
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    T5EncoderModel,
)

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    SD3Transformer2DModel, # Changed from UNet2DConditionModel
    FlowMatchEulerDiscreteScheduler, # Added import
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available

from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3, # Added import
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available


if is_wandb_available():
    import wandb

# PIL >= 9.1.0 uses Resampling
if hasattr(PIL.Image, "Resampling"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }
# ------------------------------------------------------------------------------

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# SD3 requires a recent diffusers version. Adjust if needed.
check_min_version("0.28.0") # Example version, adjust as necessary for SD3 support

logger = get_logger(__name__)


def save_model_card(repo_id: str, images=None, base_model=str, repo_folder=None):
    img_str = ""
    if images is not None:
        for i, image in enumerate(images):
            # Ensure the image is in RGB format before saving
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            img_str += f"![img_{i}](./image_{i}.png)\n"

    model_description = f'''
# Textual inversion text2image fine-tuning - {repo_id}
These are textual inversion adaption weights for {base_model}. You can find some example images in the following. \n
{img_str}

## Training procedure

Trained using the `textual_inversion_sd3.py` script from the diffusers examples.

## Usage

```python
from diffusers import StableDiffusion3Pipeline
import torch

model_id = "{base_model}" # Or your base SD3 model
pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

# Load the TI weights for all 3 encoders
# Assumes weights are saved as learned_embeds_t1.safetensors, learned_embeds_t2.safetensors, learned_embeds_t3.safetensors
# You might need to load them individually if load_textual_inversion doesn't handle multiple files per token
# or if you combined them into a single file.
pipe.load_textual_inversion("learned_embeds_t1.safetensors", token="{args.placeholder_token}", text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)
pipe.load_textual_inversion("learned_embeds_t2.safetensors", token="{args.placeholder_token}", text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2)
pipe.load_textual_inversion("learned_embeds_t3.safetensors", token="{args.placeholder_token}", text_encoder=pipe.text_encoder_3, tokenizer=pipe.tokenizer_3)


prompt = "A photo of {args.placeholder_token}" # Use your placeholder token
image = pipe(prompt, num_inference_steps=28, guidance_scale=7.0).images[0]
image.save("my_image.png")
```
'''
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="openrail++", # SD3 uses openrail++
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    tags = [
        "stable-diffusion-3",
        "stable-diffusion-3-diffusers",
        "text-to-image",
        "diffusers",
        "diffusers-training",
        "textual_inversion",
    ]

    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))


def log_validation(
    text_encoder_1,
    text_encoder_2,
    text_encoder_3, # Added text_encoder_3
    tokenizer_1,
    tokenizer_2,
    tokenizer_3, # Added tokenizer_3
    transformer, # Changed from unet
    vae,
    args,
    accelerator,
    weight_dtype,
    epoch,
    is_final_validation=False,
):
    # --- ACCELERATOR PRINT 2 ---
    accelerator.print(f"ACCELERATOR DEBUG: Entered log_validation for step (epoch var): {epoch}")

    print(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )

    # Ensure models are unwrapped and on the correct device/dtype
    text_encoder_1 = accelerator.unwrap_model(text_encoder_1)
    text_encoder_2 = accelerator.unwrap_model(text_encoder_2)
    text_encoder_3 = accelerator.unwrap_model(text_encoder_3)
    transformer_val = accelerator.unwrap_model(transformer)
    vae_val = accelerator.unwrap_model(vae)

    # Use the correct pipeline for SD3
    try:
        from diffusers import StableDiffusion3Pipeline
    except ImportError:
        logger.error("StableDiffusion3Pipeline not found. Please update diffusers.")
        return []

    pipeline = StableDiffusion3Pipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder=text_encoder_1,
        text_encoder_2=text_encoder_2,
        text_encoder_3=text_encoder_3, # Added
        tokenizer=tokenizer_1,
        tokenizer_2=tokenizer_2,
        tokenizer_3=tokenizer_3, # Added
        transformer=transformer_val, # Changed from unet
        vae=vae_val,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype, # Use the training dtype for validation pipeline
    )
    # SD3 might use a different default scheduler or DPMSolver might need specific config
    # Using the default from the loaded pipeline for now.
    # pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    try:
        pipeline = pipeline.to(accelerator.device)
        pipeline.set_progress_bar_config(disable=True)
    except Exception as e:
        logger.error(f"Failed to move pipeline to device or set progress bar: {e}")
        del pipeline
        torch.cuda.empty_cache()
        return []


    # run inference
    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    images = []

    # Use autocast for potentially lower precision validation if needed and supported
    is_mixed_precision = accelerator.mixed_precision in ["fp16", "bf16"]
    with torch.autocast(device_type=accelerator.device.type, dtype=weight_dtype, enabled=is_mixed_precision):
        for i in range(args.num_validation_images):
            try:
                # SD3 pipeline might have different default inference steps or parameters
                image = pipeline(
                    args.validation_prompt,
                    num_inference_steps=28, # Example SD3 defaults
                    generator=generator,
                    guidance_scale=7.0 # Example SD3 defaults
                    ).images[0]
                images.append(image)
            except Exception as e:
                print(f"Failed to generate validation image {i+1}: {e}")
                # Optionally add a placeholder or break
                # images.append(Image.new('RGB', (args.resolution, args.resolution), color = 'red'))
                continue # Skip failed image

    tracker_key = "test" if is_final_validation else "validation"
    for tracker in accelerator.trackers:
        if not images: # Skip logging if no images were generated
            break
        try:
            if tracker.name == "tensorboard":
                np_images = np.stack([np.asarray(img.convert("RGB")) for img in images]) # Ensure RGB
                tracker.writer.add_images(tracker_key, np_images, epoch, dataformats="NHWC")
            if tracker.name == "wandb":
                # --- ACCELERATOR PRINT 3 ---
                accelerator.print(f"ACCELERATOR DEBUG: Attempting to log {len(images)} images to WandB.")
                tracker.log(
                    {
                        tracker_key: [
                            wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)
                        ]
                    }
                )
        except Exception as e:
            logger.error(f"Failed to log validation images to {tracker.name}: {e}")


    del pipeline, transformer_val, vae_val # Clean up models used only for validation
    torch.cuda.empty_cache()
    return images


def save_progress(text_encoder, placeholder_token_ids, accelerator, args, save_path, safe_serialization=True, embedding_layer=None, text_encoder_name=""):
    """ Saves the learned embeddings for a single text encoder. """
    logger.info(f"Saving embeddings for {text_encoder_name}")
    unwrapped_encoder = accelerator.unwrap_model(text_encoder)

    # Get the embedding layer directly if passed, otherwise try to find it
    if embedding_layer is None:
        if hasattr(unwrapped_encoder, "get_input_embeddings"):
            embedding_layer = unwrapped_encoder.get_input_embeddings()
        elif hasattr(unwrapped_encoder, "shared"): # T5 specific
            embedding_layer = unwrapped_encoder.shared
        else:
            logger.error(f"Could not find embedding layer for {text_encoder_name}. Skipping save.")
            return

    if not hasattr(embedding_layer, "weight"):
        logger.error(f"Embedding layer for {text_encoder_name} does not have a 'weight' attribute. Skipping save.")
        return

    # Ensure placeholder_token_ids is not empty and contains valid indices
    if not placeholder_token_ids:
        logger.error(f"Placeholder token IDs list is empty for {text_encoder_name}. Skipping save.")
        return
    min_id = min(placeholder_token_ids)
    max_id = max(placeholder_token_ids)
    vocab_size = embedding_layer.weight.shape[0]
    if max_id >= vocab_size or min_id < 0:
         logger.error(f"Placeholder token IDs ({min_id}-{max_id}) are out of bounds for {text_encoder_name} vocab size ({vocab_size}). Skipping save.")
         return


    learned_embeds_weights = embedding_layer.weight[min_id : max_id + 1]

    # Use the main placeholder token for the dictionary key
    learned_embeds_dict = {args.placeholder_token: learned_embeds_weights.detach().cpu()}

    try:
        if safe_serialization:
            safetensors.torch.save_file(learned_embeds_dict, save_path, metadata={"format": "pt"})
        else:
            torch.save(learned_embeds_dict, save_path)
        logger.info(f"Saved embeddings for {text_encoder_name} to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save embeddings for {text_encoder_name} to {save_path}: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Textual Inversion script for Stable Diffusion 3.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save learned embeddings every X updates steps.",
    )
    # Save as full pipeline might be complex due to 3 encoders. Saving embeddings is standard.
    # parser.add_argument(
    #     "--save_as_full_pipeline",
    #     action="store_true",
    #     help="Save the complete stable diffusion pipeline (experimental for SD3 TI).",
    # )
    parser.add_argument(
        "--num_vectors",
        type=int,
        default=1,
        help="How many textual inversion vectors shall be used per placeholder token (per text encoder).",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-3-medium-diffusers", # Default to SD3 medium
        required=False,
        help="Path to pretrained SD3 model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None, # SD3 might use variants like bf16, fp16 etc. Check model card.
        help="Variant of the model files (e.g., 'fp16', 'bf16'). Check SD3 model page.",
    )
    parser.add_argument(
        "--train_data_dir", type=str, default=None, required=True, help="A folder containing the training data (images)."
    )
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default=None,
        required=True,
        help="A token string to use as a placeholder for the concept (e.g., '<my-cat>'). Should be unique and not in vocabulary.",
    )
    parser.add_argument(
        "--initializer_token", type=str, default=None, required=True, help="A single token string to use as initializer word (e.g., 'cat'). Must exist in all tokenizers."
    )
    parser.add_argument("--learnable_property", type=str, default="object", choices=["object", "style"], help="Choose between 'object' and 'style'")
    parser.add_argument("--repeats", type=int, default=100, help="How many times to repeat the training data per epoch.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd3-textual-inversion-output", # Changed default name
        help="The output directory where the model checkpoints and embeddings will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024, # SD3 common resolution
        help=(
            "The resolution for input images, all images in the train/validation dataset will be resized to this resolution."
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution instead of random crop."
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader. SD3 is memory intensive, start low."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=3000, # Typical TI steps
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4, # Accumulate gradients to simulate larger batch sizes
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass. Highly recommended for SD3.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4, # Common TI LR, may need tuning
        help="Initial learning rate (after the potential warmup period) to use for the embeddings.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0, # Start with 0 for debugging, increase if I/O is a bottleneck
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes to save memory."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model and embeddings to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository on the Hub to push the results to.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None, # Default to None, accelerator will figure it out. bf16 is recommended if available.
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). "
            "Bf16 requires PyTorch >= 1.10 and an Nvidia Ampere GPU or newer. "
            "SD3 training often uses bf16."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        default=True, # Default to True for Ampere+ GPUs
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning. E.g., 'A photo of {}'",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` formatted with the placeholder token, generating"
            " `args.num_validation_images` images, and logging them."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state (optimizer, scalar states etc.) every X updates."
            " These checkpoints are suitable for resuming training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None, # Keep all checkpoints by default
        help=("Max number of checkpoints to store. Old ones will be deleted."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint in the output dir.'
        ),
    )
    # Xformers might not be relevant for SD3's Transformer block
    # parser.add_argument(
    #     "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers (if applicable)."
    # )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help="Save memory by setting grads to None instead of zero when calling optimizer.zero_grad().",
    )
    # Add argument for tokenizer max length if needed
    # parser.add_argument("--tokenizer_max_length", type=int, default=None, help="Maximum sequence length for tokenizers.")

    # Arguments from train_dreambooth_lora_sd3.py for timestep/sigma generation
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="logit_normal",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"],
        help="Timestep weighting scheme for sampling, from SD3 Dreambooth script."
    )
    parser.add_argument(
        "--logit_mean",
        type=float,
        default=0.0,
        help="Mean for logit_normal weighting scheme."
    )
    parser.add_argument(
        "--logit_std",
        type=float,
        default=1.0,
        help="Std for logit_normal weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale for mode weighting scheme."
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.train_data_dir is None:
        raise ValueError("You must specify a train data directory (--train_data_dir).")

    # Sanity checks for placeholder and initializer tokens
    if args.placeholder_token is None or args.initializer_token is None:
            raise ValueError("Must provide --placeholder_token and --initializer_token.")

    # Basic check on placeholder token format (recommend using < >)
    if not (args.placeholder_token.startswith("<") and args.placeholder_token.endswith(">")):
        logging.warning(
            f"Placeholder token '{args.placeholder_token}' does not follow the '<token>' convention. This is recommended for clarity."
        )

    # Format validation prompt with placeholder token if provided
    if args.validation_prompt is not None:
        if "{}" not in args.validation_prompt:
            logging.warning("Validation prompt provided but does not contain '{}'. It will be used as is, without the placeholder token.")
        else:
            args.validation_prompt = args.validation_prompt.format(args.placeholder_token)


    return args


# Using the same small templates for simplicity. Could be adapted if needed.
imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a photo in the style of {}",
    "a rendering in the style of {}",
    "a dark photo in the style of {}",
    "a bright photo in the style of {}",
    "a cropped photo in the style of {}",
    "a good photo in the style of {}",
    "a close-up photo in the style of {}",
    "a rendition in the style of {}",
]


class TextualInversionDataset(Dataset):
    """
    Dataset for Textual Inversion training.
    Loads images and creates prompts using templates for SD3's three tokenizers.
    """
    def __init__(
        self,
        data_root,
        tokenizer_1, # CLIP-L tokenizer
        tokenizer_2, # OpenCLIP-G tokenizer
        tokenizer_3, # T5 tokenizer
        learnable_property="object",  # [object, style]
        size=1024, # Default SD3 size
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        placeholder_token="*",
        center_crop=False,
        tokenizer_max_length=None # Optional: can pass max length if needed
    ):
        self.data_root = data_root
        self.tokenizer_1 = tokenizer_1
        self.tokenizer_2 = tokenizer_2
        self.tokenizer_3 = tokenizer_3
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p
        self.tokenizer_max_length = tokenizer_max_length # Store max length


        # Find all image files in the data directory
        self.image_paths = []
        supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
        if os.path.isfile(self.data_root): # If data_root is a file, treat it as a list of paths
            try:
                with open(self.data_root, "r") as f:
                    self.image_paths = [line.strip() for line in f if line.strip().lower().endswith(supported_extensions) and os.path.exists(line.strip())]
                if not self.image_paths:
                    raise ValueError(f"No valid image paths found in file: {self.data_root}")
            except Exception as e:
                raise ValueError(f"Could not read or process file {self.data_root}: {e}")

        elif os.path.isdir(self.data_root): # If data_root is a directory, walk through it
            for root, _, files in os.walk(self.data_root):
                for file in files:
                    if file.lower().endswith(supported_extensions):
                        self.image_paths.append(os.path.join(root, file))
        else:
            raise ValueError(f"data_root '{self.data_root}' is not a valid file or directory.")


        if not self.image_paths:
            raise ValueError(f"No images with supported extensions {supported_extensions} found in {self.data_root} (recursively or in provided file).")

        logger.info(f"Found {len(self.image_paths)} images in {self.data_root} (recursively or from file).")

        self.num_images = len(self.image_paths)
        # self.num_images = 4
        self.repeats = repeats

        self.interpolation = PIL_INTERPOLATION[interpolation]

        self.templates = imagenet_style_templates_small if learnable_property == "style" else imagenet_templates_small
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

        # Define image transformations
        # Standard SD pre-processing: resize -> (center/random) crop -> ToTensor -> normalize to [-1, 1]
        transform_list = [
            transforms.Resize(size, interpolation=self.interpolation),
        ]
        if center_crop:
            transform_list.append(transforms.CenterCrop(size))
        else:
            transform_list.append(transforms.RandomCrop(size))

        transform_list.extend([
            self.flip_transform, # Apply random flip
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]), # Normalize to [-1, 1]
        ])
        self.image_transforms = transforms.Compose(transform_list)


        # Determine max length for tokenizers if not provided
        self.max_length_1 = tokenizer_max_length or self.tokenizer_1.model_max_length
        self.max_length_2 = tokenizer_max_length or self.tokenizer_2.model_max_length
        self.max_length_3 = tokenizer_max_length or self.tokenizer_3.model_max_length


    def __len__(self):
        return self.num_images * self.repeats

    def tokenize(self, text, tokenizer, max_length):
        """Helper to tokenize text with padding and truncation."""
        return tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).input_ids[0]


    def __getitem__(self, i):
        idx = i % self.num_images
        image_path = self.image_paths[idx]
        example = {}

        try:
            image = Image.open(image_path)
            if not image.mode == "RGB":
                image = image.convert("RGB")
        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {e}. Skipping index {i} (image {idx}).")
            # Return next item recursively, careful with potential infinite loops if all fail
            if self.__len__() == 0: return None # Should not happen if check in init passes
            return self.__getitem__((i + 1) % self.__len__())


        # Format prompt with placeholder token
        text = random.choice(self.templates).format(self.placeholder_token)

        try:
            # Tokenize for all three tokenizers
            example["input_ids_1"] = self.tokenize(text, self.tokenizer_1, self.max_length_1)
            example["input_ids_2"] = self.tokenize(text, self.tokenizer_2, self.max_length_2)
            example["input_ids_3"] = self.tokenize(text, self.tokenizer_3, self.max_length_3)

            # Apply image transformations
            example["pixel_values"] = self.image_transforms(image)

        except Exception as e:
             logger.warning(f"Error processing image {image_path} or text '{text}': {e}. Skipping index {i} (image {idx}).")
             return self.__getitem__((i + 1) % self.__len__())

        return example


# Collate function to handle potential None values from dataset __getitem__ errors
def collate_fn(examples):
    examples = [e for e in examples if e is not None]
    if not examples:
        return None # Return None if the whole batch failed

    # Collate fields. Assumes default collate behavior for tensors.
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    input_ids_1 = torch.stack([example["input_ids_1"] for example in examples])
    input_ids_2 = torch.stack([example["input_ids_2"] for example in examples])
    input_ids_3 = torch.stack([example["input_ids_3"] for example in examples])

    return {
        "pixel_values": pixel_values,
        "input_ids_1": input_ids_1,
        "input_ids_2": input_ids_2,
        "input_ids_3": input_ids_3,
    }


# Copied from diffusers.examples.dreambooth.train_dreambooth_lora_sd3.py
# Minor adaptations for TI: prioritize text_input_ids_list from batch, handle max_lengths

def tokenize_prompt(tokenizer, prompt, max_length=77): # Added max_length param for flexibility
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_length, 
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids

def _encode_prompt_with_t5(
    text_encoder,
    tokenizer, 
    max_sequence_length,
    prompt=None, 
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None, 
):
    if text_input_ids is None:
        if tokenizer is None or prompt is None:
            raise ValueError("Either (tokenizer and prompt) or text_input_ids must be provided for T5.")
        current_prompt_list = [prompt] if isinstance(prompt, str) else prompt
        text_inputs_for_t5 = tokenizer(
            current_prompt_list,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        # Repeat if num_images_per_prompt > 1 and input was for single batch items
        if num_images_per_prompt > 1 and text_inputs_for_t5.input_ids.shape[0] == len(current_prompt_list):
             text_input_ids = text_inputs_for_t5.input_ids.repeat_interleave(num_images_per_prompt, dim=0)
        else:
            text_input_ids = text_inputs_for_t5.input_ids
    # If text_input_ids provided, it's assumed to be correctly shaped for (batch*num_images_per_prompt)

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    return prompt_embeds

def _encode_prompt_with_clip(
    text_encoder,
    tokenizer, 
    prompt: Union[str, List[str]], 
    max_length=77, 
    device=None,
    text_input_ids=None, 
    num_images_per_prompt: int = 1,
):
    if text_input_ids is None:
        if tokenizer is None or prompt is None:
            raise ValueError("Either (tokenizer and prompt) or text_input_ids must be provided for CLIP.")
        current_prompt_list = [prompt] if isinstance(prompt, str) else prompt
        text_inputs_for_clip = tokenizer(
            current_prompt_list,
            padding="max_length",
            max_length=max_length, 
            truncation=True,
            return_tensors="pt",
        )
        if num_images_per_prompt > 1 and text_inputs_for_clip.input_ids.shape[0] == len(current_prompt_list):
            text_input_ids = text_inputs_for_clip.input_ids.repeat_interleave(num_images_per_prompt, dim=0)
        else:
            text_input_ids = text_inputs_for_clip.input_ids
    # If text_input_ids provided, it's assumed to be correctly shaped

    prompt_embeds_output = text_encoder(text_input_ids.to(device), output_hidden_states=True)

    pooled_prompt_embeds = prompt_embeds_output[0]
    prompt_embeds = prompt_embeds_output.hidden_states[-2] 
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)
    return prompt_embeds, pooled_prompt_embeds

def encode_prompt(
    text_encoders: List[transformers.PreTrainedModel], 
    tokenizers: Optional[List[transformers.PreTrainedTokenizer]] = None, 
    prompt: Optional[Union[str, List[str]]] = None, 
    text_input_ids_list: Optional[List[torch.Tensor]] = None, 
    num_images_per_prompt: int = 1,
    device: Optional[torch.device] = None,
    clip_max_length: int = 77, 
    t5_max_sequence_length: int = 256, 
):
    if text_input_ids_list is None and (prompt is None or tokenizers is None):
        raise ValueError("Either text_input_ids_list or (prompt and tokenizers) must be provided.")

    active_prompt_list = None
    if prompt is not None:
        active_prompt_list = [prompt] if isinstance(prompt, str) else prompt
    
    if active_prompt_list is None and text_input_ids_list is not None:
        num_actual_batch_items = text_input_ids_list[0].shape[0] // num_images_per_prompt
        active_prompt_list = [""] * num_actual_batch_items
    elif active_prompt_list is None:
        raise ValueError("Prompt could not be determined for encode_prompt.")

    clip_text_encoders = text_encoders[:2]
    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []

    for i, text_encoder in enumerate(clip_text_encoders):
        current_tokenizer_for_clip = tokenizers[i] if tokenizers and len(tokenizers) > i else None
        current_text_ids_for_clip = text_input_ids_list[i] if text_input_ids_list and len(text_input_ids_list) > i else None
        
        # Determine effective tokenizer and prompt for the _encode_prompt_with_clip call
        # If text_input_ids are available, tokenizer and prompt are not used for tokenization by _encode_prompt_with_clip
        eff_tok = None if current_text_ids_for_clip is not None else current_tokenizer_for_clip
        # Pass active_prompt_list if tokenizer is to be used, otherwise None (as ids are primary)
        eff_prmpt = active_prompt_list if current_text_ids_for_clip is None and eff_tok is not None else None

        if current_text_ids_for_clip is None and (eff_tok is None or active_prompt_list is None):
            raise ValueError(f"CLIP Encoder {i}: Missing inputs. Need text_input_ids or (tokenizer and prompt).")

        p_embeds, pooled_p_embeds = _encode_prompt_with_clip(
            text_encoder=text_encoder,
            tokenizer=eff_tok,
            prompt=eff_prmpt, 
            max_length=clip_max_length,
            device=device if device is not None else text_encoder.device,
            num_images_per_prompt=num_images_per_prompt,
            text_input_ids=current_text_ids_for_clip,
        )
        clip_prompt_embeds_list.append(p_embeds)
        clip_pooled_prompt_embeds_list.append(pooled_p_embeds)

    final_clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
    final_pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

    t5_enc = text_encoders[2]
    t5_tok = tokenizers[2] if tokenizers and len(tokenizers) > 2 else None
    t5_ids = text_input_ids_list[2] if text_input_ids_list and len(text_input_ids_list) > 2 else None

    eff_t5_tok = None if t5_ids is not None else t5_tok
    eff_t5_prmpt = active_prompt_list if t5_ids is None and eff_t5_tok is not None else None
    
    if t5_ids is None and (eff_t5_tok is None or active_prompt_list is None):
        raise ValueError("T5 Encoder: Missing inputs. Need text_input_ids or (tokenizer and prompt).")

    t5_prompt_e = _encode_prompt_with_t5(
        text_encoder=t5_enc,
        tokenizer=eff_t5_tok,
        max_sequence_length=t5_max_sequence_length,
        prompt=eff_t5_prmpt,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=t5_ids,
        device=device if device is not None else t5_enc.device,
    )

    padded_clip_prompt_embeds = torch.nn.functional.pad(
        final_clip_prompt_embeds, (0, t5_prompt_e.shape[-1] - final_clip_prompt_embeds.shape[-1])
    )
    combined_prompt_embeds = torch.cat([padded_clip_prompt_embeds, t5_prompt_e], dim=-2)

    return combined_prompt_embeds, final_pooled_prompt_embeds

def main():
    args = parse_args()
    # --- DEBUG PRINT - Start ---
    # print(f"DEBUG START: args.validation_prompt = {args.validation_prompt}") # Keep for potential direct stdout check

    # Setup accelerator
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    # --- ACCELERATOR PRINT - Start ---
    accelerator.print(f"ACCELERATOR DEBUG START: args.validation_prompt = {args.validation_prompt}")

    # Security check for wandb and hub_token
    if args.report_to == "wandb" and args.hub_token is not None:
        logger.warning(
            "You cannot use both --report_to=wandb and --hub_token due to potential security risks. "
            "Please use `huggingface-cli login` for authentication."
        )
        # Decide if error or just warn - warning for now.
        # raise ValueError("Cannot use wandb and hub_token together.")


    # Enable TF32 if requested and available
    if args.allow_tf32:
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            logger.info("Enabling TF32 for faster training on Ampere GPUs.")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        else:
             logger.info("TF32 not enabled. Requires CUDA >= 11 and Ampere GPU or newer.")


    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Set seed before initializing models.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation on the main process
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            if args.hub_model_id is None:
                # Default repo name if not provided
                base_name = Path(args.pretrained_model_name_or_path).name.replace('/','_') # Handle namespace
                repo_id = f"{base_name}-ti-{args.placeholder_token}"
                logger.info(f"Hub model ID not provided, defaulting to '{repo_id}'")
            else:
                repo_id = args.hub_model_id

            try:
                repo_id = create_repo(repo_id=repo_id, exist_ok=True, token=args.hub_token).repo_id
                logger.info(f"Created or found repository on the Hub: {repo_id}")
            except Exception as e:
                 logger.error(f"Failed to create or access repository '{repo_id}': {e}. Check hub token and permissions.")
                 # Decide if this is fatal or can proceed without pushing
                 args.push_to_hub = False # Disable pushing if repo creation fails
                 logger.warning("Disabling --push_to_hub due to repository creation error.")


    # Load tokenizers using AutoTokenizer
    try:
        tokenizer_1 = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision, use_fast=False)
        tokenizer_2 = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision, use_fast=False)
        tokenizer_3 = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_3", revision=args.revision, use_fast=True)
        logger.info("Tokenizers loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load tokenizers from {args.pretrained_model_name_or_path}: {e}")
        raise # Cannot proceed without tokenizers


    # Load scheduler and models
    try:
        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler") # Changed from DDPMScheduler
        noise_scheduler_copy = copy.deepcopy(noise_scheduler) # For get_sigmas, as in Dreambooth SD3
        text_encoder_1 = CLIPTextModelWithProjection.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
        )
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
        )
        text_encoder_3 = T5EncoderModel.from_pretrained(
             args.pretrained_model_name_or_path, subfolder="text_encoder_3", revision=args.revision, variant=args.variant
        )
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
        )
        transformer = SD3Transformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
        )
        logger.info("Models (Sched, VAE, Transformer, Text Encoders) loaded successfully.")
    except Exception as e:
         logger.error(f"Failed to load models from {args.pretrained_model_name_or_path}: {e}")
         raise # Cannot proceed without models


    # Helper function to add tokens and check for errors
    def add_tokens_to_tokenizer(tokenizer, placeholder_tokens, tokenizer_name):
        num_added = tokenizer.add_tokens(placeholder_tokens)
        if num_added != len(placeholder_tokens):
            # Find which tokens already existed
            existing_tokens = []
            vocab = tokenizer.get_vocab()
            for token in placeholder_tokens:
                if token in vocab:
                    existing_tokens.append(token)
            raise ValueError(
                f"{tokenizer_name} already contained the following token(s): {existing_tokens}. "
                f"Please pass different `placeholder_token`(s) that are not already in the tokenizer vocabulary."
                f"Expected to add {len(placeholder_tokens)} tokens, but only added {num_added} new tokens."
            )
        return tokenizer.convert_tokens_to_ids(placeholder_tokens)

    # Prepare placeholder tokens (main token + potential multi-vector tokens)
    placeholder_tokens = [args.placeholder_token]
    if args.num_vectors < 1:
        raise ValueError(f"--num_vectors has to be >= 1, but is {args.num_vectors}")
    for i in range(1, args.num_vectors):
        placeholder_tokens.append(f"{args.placeholder_token}_{i}")

    # Add placeholder tokens to all tokenizers and get their IDs
    try:
        placeholder_token_ids_1 = add_tokens_to_tokenizer(tokenizer_1, placeholder_tokens, "Tokenizer 1 (CLIP-L)")
        placeholder_token_ids_2 = add_tokens_to_tokenizer(tokenizer_2, placeholder_tokens, "Tokenizer 2 (OpenCLIP-G)")
        placeholder_token_ids_3 = add_tokens_to_tokenizer(tokenizer_3, placeholder_tokens, "Tokenizer 3 (T5)")
        logger.info(f"Added placeholder tokens {placeholder_tokens} to all tokenizers.")
        logger.info(f"Corresponding IDs T1: {placeholder_token_ids_1}")
        logger.info(f"Corresponding IDs T2: {placeholder_token_ids_2}")
        logger.info(f"Corresponding IDs T3: {placeholder_token_ids_3}")
    except ValueError as e: # Catch the error raised by add_tokens_to_tokenizer
        logger.error(e)
        raise # Re-raise the error to stop execution


    # Helper function to encode initializer and check length
    def get_initializer_token_id(tokenizer, initializer_token, tokenizer_name):
        token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
        if len(token_ids) != 1:
            logger.warning(f"Initializer token '{initializer_token}' is tokenized into {len(token_ids)} IDs ({token_ids}) by {tokenizer_name}. Using the first ID ({token_ids[0]}) for initialization. This may impact quality if the token is not effectively a single unit.")
            # You might want to raise an error here instead for stricter control:
            # raise ValueError(f"Initializer token '{initializer_token}' must be a single token in {tokenizer_name}, but got {len(token_ids)} IDs: {token_ids}")
        return token_ids[0]

    # Convert the initializer token to IDs for all tokenizers
    try:
        initializer_token_id_1 = get_initializer_token_id(tokenizer_1, args.initializer_token, "Tokenizer 1")
        initializer_token_id_2 = get_initializer_token_id(tokenizer_2, args.initializer_token, "Tokenizer 2")
        initializer_token_id_3 = get_initializer_token_id(tokenizer_3, args.initializer_token, "Tokenizer 3")
        logger.info(f"Initializer token '{args.initializer_token}' found with IDs: T1={initializer_token_id_1}, T2={initializer_token_id_2}, T3={initializer_token_id_3}")
    except Exception as e: # Catch errors during encoding (e.g., token not found)
        logger.error(f"Failed to encode initializer token '{args.initializer_token}' in one or more tokenizers: {e}")
        raise


    # Resize the token embeddings for all text encoders AFTER adding tokens
    text_encoder_1.resize_token_embeddings(len(tokenizer_1))
    text_encoder_2.resize_token_embeddings(len(tokenizer_2))
    text_encoder_3.resize_token_embeddings(len(tokenizer_3))
    logger.info("Resized token embeddings for all text encoders.")


    # Helper function to get embedding layer, handling T5's 'shared' attribute
    def get_embedding_layer(text_encoder):
        if hasattr(text_encoder, "get_input_embeddings"):
            return text_encoder.get_input_embeddings()
        elif hasattr(text_encoder, "shared"): # T5 specific
            return text_encoder.shared
        else:
            raise AttributeError(f"Could not find embedding layer for encoder {type(text_encoder)}")


    # Initialize the newly added placeholder token embeddings
    try:
        embedding_layer_1 = get_embedding_layer(text_encoder_1)
        embedding_layer_2 = get_embedding_layer(text_encoder_2)
        embedding_layer_3 = get_embedding_layer(text_encoder_3)

        token_embeds_1 = embedding_layer_1.weight.data
        token_embeds_2 = embedding_layer_2.weight.data
        token_embeds_3 = embedding_layer_3.weight.data

        with torch.no_grad():
            for token_id in placeholder_token_ids_1:
                token_embeds_1[token_id] = token_embeds_1[initializer_token_id_1].clone()
            for token_id in placeholder_token_ids_2:
                token_embeds_2[token_id] = token_embeds_2[initializer_token_id_2].clone()
            for token_id in placeholder_token_ids_3:
                token_embeds_3[token_id] = token_embeds_3[initializer_token_id_3].clone()
        logger.info("Initialized placeholder token embeddings using initializer token embeddings.")
    except Exception as e:
        logger.error(f"Failed to initialize placeholder embeddings: {e}")
        raise


    # Freeze VAE and Transformer (main diffusion model)
    vae.requires_grad_(False)
    transformer.requires_grad_(False)
    logger.info("Froze VAE and Transformer parameters.")

    # Freeze all parameters except for the token embeddings in text encoders
    # This requires careful handling to ensure only the embedding layer weights are trainable
    def freeze_encoder_except_embeddings(text_encoder, encoder_name):
        logger.info(f"Freezing non-embedding parameters for {encoder_name}")
        # Freeze everything first
        for param in text_encoder.parameters():
            param.requires_grad = False
        # Unfreeze only the embedding layer
        embedding_layer = get_embedding_layer(text_encoder)
        for param in embedding_layer.parameters():
             param.requires_grad = True
        logger.info(f"Unfroze embedding parameters for {encoder_name}")

    try:
        freeze_encoder_except_embeddings(text_encoder_1, "Text Encoder 1 (CLIP-L)")
        freeze_encoder_except_embeddings(text_encoder_2, "Text Encoder 2 (OpenCLIP-G)")
        freeze_encoder_except_embeddings(text_encoder_3, "Text Encoder 3 (T5)")
    except Exception as e:
        logger.error(f"Error freezing text encoder parameters: {e}")
        raise


    if args.gradient_checkpointing:
        # Enable gradient checkpointing only on models with trainable parameters (text encoders)
        text_encoder_1.gradient_checkpointing_enable()
        text_encoder_2.gradient_checkpointing_enable()
        text_encoder_3.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing for text encoders.")
        # Note: transformer.gradient_checkpointing_enable() is not needed as it's frozen


    # Scale learning rate if requested
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
        logger.info(f"Scaled learning rate to {args.learning_rate}")


    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_class = bnb.optim.AdamW8bit
            logger.info("Using 8-bit Adam optimizer.")
        except ImportError:
            logger.error("bitsandbytes not found. Please install with `pip install bitsandbytes` to use 8-bit Adam. Falling back to regular AdamW.")
            optimizer_class = torch.optim.AdamW
    else:
        optimizer_class = torch.optim.AdamW
        logger.info("Using regular AdamW optimizer.")


    # Get parameters to optimize (only the embedding layers' parameters)
    params_to_optimize = [
        {"params": get_embedding_layer(text_encoder_1).parameters()},
        {"params": get_embedding_layer(text_encoder_2).parameters()},
        {"params": get_embedding_layer(text_encoder_3).parameters()},
    ]

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and DataLoader creation
    try:
        train_dataset = TextualInversionDataset(
            data_root=args.train_data_dir,
            tokenizer_1=tokenizer_1,
            tokenizer_2=tokenizer_2,
            tokenizer_3=tokenizer_3,
            size=args.resolution,
            placeholder_token=args.placeholder_token, # Use the main token
            repeats=args.repeats,
            learnable_property=args.learnable_property,
            center_crop=args.center_crop,
            flip_p=0.5, # Standard default
            set="train",
            # tokenizer_max_length=args.tokenizer_max_length # Pass if arg exists
        )
        logger.info(f"Created training dataset with {len(train_dataset)} examples (incl. repeats).")
    except Exception as e:
        logger.error(f"Failed to create training dataset: {e}")
        raise

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn, # Use custom collate_fn to handle errors
        num_workers=args.dataloader_num_workers,
        pin_memory=True, # Enable pin_memory for potentially faster transfer
    )


    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
        logger.info(f"max_train_steps not provided, setting to {args.max_train_steps} ({args.num_train_epochs} epochs * {num_update_steps_per_epoch} steps/epoch)")
    else:
        overrode_max_train_steps = False
        # Recalculate epochs based on max_train_steps
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        logger.info(f"max_train_steps provided ({args.max_train_steps}), running for {args.num_train_epochs} epochs.")


    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes, # Scale warmup steps
        num_training_steps=args.max_train_steps * accelerator.num_processes, # Scale total steps
        num_cycles=args.lr_num_cycles,
    )

    # Prepare models and optimizer with accelerator
    # Note: We only prepare the models with trainable parameters (text encoders) and the optimizer/dataloader/scheduler.
    # Frozen models (VAE, Transformer) are moved to device manually later.
    text_encoder_1, text_encoder_2, text_encoder_3, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        text_encoder_1, text_encoder_2, text_encoder_3, optimizer, train_dataloader, lr_scheduler
    )
    logger.info("Prepared text encoders, optimizer, dataloader, and LR scheduler with Accelerator.")


    # Move frozen models to device and set dtype
    # VAE often works best in fp32 or bf16. Check SD3 recommendations.
    # Using bf16 if available and accelerator uses it, otherwise fp32.
    vae_dtype = torch.bfloat16 if accelerator.mixed_precision == "bf16" else torch.float32
    weight_dtype = torch.float32 # Default fallback
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    try:
        vae.to(accelerator.device, dtype=vae_dtype)
        transformer.to(accelerator.device, dtype=weight_dtype) # Use main training dtype for transformer
        logger.info(f"Moved VAE (dtype: {vae_dtype}) and Transformer (dtype: {weight_dtype}) to device: {accelerator.device}")
    except Exception as e:
         logger.error(f"Failed to move VAE or Transformer to device: {e}")
         raise


    # Recalculate steps/epochs *again* after preparing dataloader, as it might change length with distributed setup.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps: # If we calculated max_train_steps initially
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    logger.info(f"Re-calculated training parameters: {args.num_train_epochs} epochs, {num_update_steps_per_epoch} update steps per epoch.")


    # Initialize trackers on main process
    if accelerator.is_main_process:
        # --- START MODIFICATION ---
        # current_run_output_dir will be args.output_dir unless wandb is active
        # and a wandb-specific subdirectory is successfully created.
        current_run_output_dir = args.output_dir
        # path_to_log_to_wandb will store the actual path used, for logging to wandb.
        # Default to the absolute version of the base output_dir.
        path_to_log_to_wandb = os.path.abspath(args.output_dir)
        is_wandb_effectively_active = False # Flag to determine if wandb logging should occur
        # --- END MODIFICATION ---

        run_name = f"sd3-ti_{args.placeholder_token}_lr{args.learning_rate}_steps{args.max_train_steps}"
        
        accelerator.init_trackers("textual_inversion_sd3", config=vars(args), init_kwargs={"wandb": {"name": run_name}})
        logger.info(f"Initialized trackers (report_to={args.report_to})")

        # --- START MODIFICATION (Revising the check for wandb activity) ---
        # Determine if wandb is active and attempt to create a run-specific directory.
        wandb_tracker_run_obj = None
        try:
            # This will succeed if "wandb" is among the configured log_with options
            # and was successfully initialized by init_trackers.
            wandb_tracker_run_obj = accelerator.get_tracker("wandb", unwrap=True)
        except ValueError:
            # This occurs if "wandb" was not in log_with or failed to initialize.
            logger.info("WandB tracker not configured or found in accelerator. Artifacts will be saved to base output directory.")
            # current_run_output_dir remains args.output_dir
            # path_to_log_to_wandb remains os.path.abspath(args.output_dir)
            # is_wandb_effectively_active remains False (as initialized)

        if wandb_tracker_run_obj:
            is_wandb_effectively_active = True # Wandb is active for tracking purposes
            if hasattr(wandb_tracker_run_obj, 'id') and wandb_tracker_run_obj.id:
                wandb_id = wandb_tracker_run_obj.id
                specific_run_dir = os.path.join(args.output_dir, f"wandb-{wandb_id}")
                try:
                    os.makedirs(specific_run_dir, exist_ok=True)
                    current_run_output_dir = specific_run_dir # Update to the specific run directory
                    path_to_log_to_wandb = os.path.abspath(current_run_output_dir) # Update path to log
                    logger.info(f"WandB run ID: {wandb_id}. Artifacts for this run will be saved to: {current_run_output_dir}")
                except OSError as e:
                    logger.warning(f"Could not create directory {specific_run_dir}: {e}. Using base output directory {args.output_dir}. Path logged to WandB will be {path_to_log_to_wandb} (which is the abspath of the base output dir).")
                    # current_run_output_dir remains args.output_dir
                    # path_to_log_to_wandb (already abspath of args.output_dir) is correct for this fallback
            else:
                logger.warning(f"WandB tracker is active, but could not retrieve run ID. Using base output directory {args.output_dir}. Path logged to WandB will be {path_to_log_to_wandb} (which is the abspath of the base output dir).")
                # current_run_output_dir remains args.output_dir
                # path_to_log_to_wandb (already abspath of args.output_dir) is correct
        # --- END MODIFICATION ---

    global_step = 0
    first_epoch = 0

    # --- START MODIFICATION ---
    # Log the output path (potentially wandb-specific) after global_step is initialized.
    # Use the 'is_wandb_effectively_active' flag set after attempting to get the wandb tracker.
    if accelerator.is_main_process and is_wandb_effectively_active:
        # 'path_to_log_to_wandb' was set up (or confirmed as default) in the block above.
        # It holds the absolute path to current_run_output_dir (either base or wandb-specific).
        try:
            accelerator.log({"full_run_output_path": path_to_log_to_wandb}, step=global_step) # global_step is 0 here
            logger.info(f"Logged full_run_output_path: {path_to_log_to_wandb} to WandB (at step {global_step}).")
        except Exception as e: # Catch any potential error during logging itself
            logger.error(f"Error logging full_run_output_path to WandB: {e}")
    # --- END MODIFICATION ---

    # Load state if resuming checkpoint
    if args.resume_from_checkpoint:
        checkpoint_path = None
        if args.resume_from_checkpoint == "latest":
            # Find the latest checkpoint directory
            dirs = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
            if dirs:
                dirs.sort(key=lambda x: int(x.split("-")[1]))
                checkpoint_path = os.path.join(args.output_dir, dirs[-1])
        else:
            checkpoint_path = args.resume_from_checkpoint

        if checkpoint_path and os.path.isdir(checkpoint_path):
            logger.info(f"Resuming from checkpoint {checkpoint_path}")
            try:
                accelerator.load_state(checkpoint_path)
                global_step = int(checkpoint_path.split("-")[-1]) # Extract step from dir name
                resume_step = global_step * args.gradient_accumulation_steps
                first_epoch = global_step // num_update_steps_per_epoch
                logger.info(f"Resumed training from global step {global_step}, epoch {first_epoch}.")
            except Exception as e:
                 logger.error(f"Failed to load checkpoint state from {checkpoint_path}: {e}. Starting training from scratch.")
                 global_step = 0
                 first_epoch = 0
                 args.resume_from_checkpoint = None # Prevent trying again
        else:
            logger.warning(f"Checkpoint '{args.resume_from_checkpoint}' not found or not a directory. Starting training from scratch.")
            args.resume_from_checkpoint = None # Reset arg if path invalid

    initial_global_step = global_step

    progress_bar = tqdm(
        range(initial_global_step, args.max_train_steps),
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    # Keep original embeddings as reference to restore non-trained embeddings
    # Get embeddings from potentially wrapped models
    orig_embeds_params_1 = get_embedding_layer(accelerator.unwrap_model(text_encoder_1)).weight.data.clone()
    orig_embeds_params_2 = get_embedding_layer(accelerator.unwrap_model(text_encoder_2)).weight.data.clone()
    orig_embeds_params_3 = get_embedding_layer(accelerator.unwrap_model(text_encoder_3)).weight.data.clone()

    # Helper function for sigmas from train_dreambooth_lora_sd3.py
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        # Uses noise_scheduler_copy and accelerator (available in main scope)
        # Exact match to Dreambooth SD3:
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device) # Ensure timesteps are on accelerator.device for comparison
        
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # Training loop
    for epoch in range(first_epoch, args.num_train_epochs):
        text_encoder_1.train()
        text_encoder_2.train()
        text_encoder_3.train()

        for step, batch in enumerate(train_dataloader):
            # Handle potential errors from collate_fn
            if batch is None:
                 logger.warning(f"Skipping step {step} in epoch {epoch} due to batch collation error (likely image loading issues).")
                 # If skipping steps, progress bar might not reach max_train_steps exactly.
                 # Consider how to handle this or ensure dataset is clean.
                 continue


            # Accumulate gradients across text encoders
            with accelerator.accumulate([text_encoder_1, text_encoder_2, text_encoder_3]):
                # --- ACCELERATOR PRINT - Loop ---
                accelerator.print(f"ACCELERATOR DEBUG LOOP: args.validation_prompt = {args.validation_prompt}")
                # 1. Encode Images to Latents (using VAE in correct precision and no_grad)
                with torch.no_grad():
                    # Move batch to device inside the training loop? Accelerator might handle this. Let's assume batch is on device.
                    # pixel_values = batch["pixel_values"].to(accelerator.device, dtype=vae_dtype) # Ensure correct device and dtype
                    raw_latents = vae.encode(batch["pixel_values"].to(dtype=vae_dtype)).latent_dist.sample()
                    latents = (raw_latents - vae.config.shift_factor) * vae.config.scaling_factor # Corrected: Apply shift_factor then scaling_factor
                    latents = latents.to(dtype=weight_dtype) # Cast back to training precision if needed
    
                # 2. Sample Noise and Timesteps
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                # timesteps = timesteps.long()

                # New timestep sampling from train_dreambooth_lora_sd3.py
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                    device=latents.device # Ensure u is on the same device as latents
                )
                # noise_scheduler_copy should have config and timesteps attributes
                if not hasattr(noise_scheduler_copy, 'config') or not hasattr(noise_scheduler_copy, 'timesteps'):
                    raise AttributeError("noise_scheduler_copy is missing 'config' or 'timesteps' attribute.")
                
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                # Ensure timesteps from noise_scheduler_copy are moved to the correct device
                timesteps = noise_scheduler_copy.timesteps.to(device=latents.device)[indices]

                # 3. Add Noise to Latents (Forward Process)
                # noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps) # Error: 'FlowMatchEulerDiscreteScheduler' object has no attribute 'add_noise'
                
                # Manual noising for sigma-based schedulers: z_t = z_0 + epsilon * sigma_t
                # Ensure sigmas are on the correct device and dtype
                # .sigmas should be populated by from_pretrained from the scheduler_config.json
                # if not hasattr(noise_scheduler, 'sigmas') or noise_scheduler.sigmas is None:
                #     raise AttributeError("noise_scheduler does not have 'sigmas' populated. This should not happen if loaded from SD3 config.")

                # current_scheduler_sigmas = noise_scheduler.sigmas.to(device=latents.device, dtype=latents.dtype)
                
                # # 'timesteps' are already integer indices for the sigmas array
                # sigma_t_batch = current_scheduler_sigmas[timesteps]
                
                # # Reshape sigmas to be broadcastable for element-wise multiplication: (bsz, 1, 1, 1)
                # sigma_t_batch = sigma_t_batch.reshape(bsz, 1, 1, 1) 
                
                # Use get_sigmas helper and align with train_dreambooth_lora_sd3.py
                sigmas = get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
                noisy_latents = (1.0 - sigmas) * latents + sigmas * noise

                # 4. Encode Prompts using the new encode_prompt function
                # The batch already contains input_ids_1, input_ids_2, input_ids_3 from the dataset
                text_input_ids_list = [batch["input_ids_1"], batch["input_ids_2"], batch["input_ids_3"]]
                
                # The `encode_prompt` function (copied from Dreambooth script) will handle
                # passing these to the respective text encoders and combining their outputs.
                # `num_images_per_prompt` is 1 for TI training steps.
                # `prompt` can be None as text_input_ids_list is primary.
                # `tokenizers` list is passed for completeness if `encode_prompt` has fallback logic,
                # but it should primarily use the provided text_input_ids_list.
                prompt_embeds, pooled_prompt_embeds = encode_prompt(
                    text_encoders=[text_encoder_1, text_encoder_2, text_encoder_3],
                    tokenizers=[tokenizer_1, tokenizer_2, tokenizer_3], # For fallback/completeness
                    prompt=None, # Debug text from batch could go here, e.g., batch.get("captions", None)
                    text_input_ids_list=text_input_ids_list,
                    num_images_per_prompt=1, # Standard for TI training step
                    device=accelerator.device,
                    # Max lengths for tokenizers if they were to be used internally by encode_prompt:
                    # clip_max_length=tokenizer_1.model_max_length, # Example: or args.clip_max_length
                    # t5_max_sequence_length=tokenizer_3.model_max_length # Example: or args.t5_max_length
                )
                
                # Ensure embeddings are in the correct training dtype (handled by Accelerator or explicitly cast if needed)
                prompt_embeds = prompt_embeds.to(dtype=weight_dtype)
                pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=weight_dtype)

                # 5. Combine Embeddings for SD3 Transformer (This logic is now inside encode_prompt)
                # NO LONGER NEEDED HERE:
                # # Concatenate CLIP-L and OpenCLIP-G hidden states
                # clip_prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
                # # Pad the concatenated CLIP embeddings to match the T5 embedding dimension
                # clip_prompt_embeds = torch.nn.functional.pad(
                #      clip_prompt_embeds, (0, prompt_embeds_3.shape[-1] - clip_prompt_embeds.shape[-1])
                # )
                # # Concatenate the padded CLIP embeddings and T5 embeddings along the sequence length dim
                # prompt_embeds = torch.cat([clip_prompt_embeds, prompt_embeds_3], dim=-2)
                # # Concatenate the pooled outputs from CLIP-L and OpenCLIP-G ONLY
                # pooled_prompt_embeds = torch.cat([pooled_prompt_embeds_1, pooled_prompt_embeds_2], dim=-1)
                # # Ensure embeddings are in the correct training dtype
                # prompt_embeds = prompt_embeds.to(dtype=weight_dtype)
                # pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=weight_dtype)

                # 6. Predict Noise using Transformer
                # Pass the correctly combined embeddings from encode_prompt
                model_pred = transformer(
                    hidden_states=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,      # Combined padded CLIP + T5 hidden states
                    pooled_projections=pooled_prompt_embeds, # Combined CLIP L + G pooled outputs
                    return_dict=False,
                )[0]


                # 7. Calculate Loss
                # User-specified target: original latents (z_0) minus noisy latents (z_t).
                # latents (z_0) = clean VAE output
                # noisy_latents (z_t) = latents + noise * sigma_t
                # So, target = latents - noisy_latents = latents - (latents + noise * sigma_t_batch) = -noise * sigma_t_batch
                target = noise-latents

                # Loss calculation - use float32 for stability
                # loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Add loss weighting consistent with train_dreambooth_lora_sd3.py
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    dim=1, # ensure correct dimension for mean
                )
                loss = loss.mean()

                # 8. Backward Pass and Optimizer Step
                accelerator.backward(loss)

                # --- DIAGNOSTICS: GRADIENT NORMS (Before Optimizer Step) ---
                if accelerator.is_main_process:
                    diag_logs = {}
                    with torch.no_grad():
                        # Text Encoder 1
                        emb_layer_1 = get_embedding_layer(accelerator.unwrap_model(text_encoder_1))
                        if emb_layer_1.weight.grad is not None:
                            placeholder_grads_1 = emb_layer_1.weight.grad[placeholder_token_ids_1]
                            diag_logs["grad_norm_emb1_placeholder"] = torch.linalg.norm(placeholder_grads_1).item()
                        
                        # Text Encoder 2
                        emb_layer_2 = get_embedding_layer(accelerator.unwrap_model(text_encoder_2))
                        if emb_layer_2.weight.grad is not None:
                            placeholder_grads_2 = emb_layer_2.weight.grad[placeholder_token_ids_2]
                            diag_logs["grad_norm_emb2_placeholder"] = torch.linalg.norm(placeholder_grads_2).item()

                        # Text Encoder 3
                        emb_layer_3 = get_embedding_layer(accelerator.unwrap_model(text_encoder_3))
                        if emb_layer_3.weight.grad is not None:
                            placeholder_grads_3 = emb_layer_3.weight.grad[placeholder_token_ids_3]
                            diag_logs["grad_norm_emb3_placeholder"] = torch.linalg.norm(placeholder_grads_3).item()
                    if diag_logs: # Only log if there's something (grads might be None initially or in edge cases)
                        # accelerator.log(diag_logs, step=global_step) # Changed to print
                        print(f"Step {global_step} [Grad Norms]: {diag_logs}")
                # --- END DIAGNOSTICS ---

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                # --- DIAGNOSTICS: PLACEHOLDER EMBEDDING NORMS (After Optimizer Step) ---
                if accelerator.is_main_process:
                    diag_logs = {}
                    with torch.no_grad():
                        # Text Encoder 1
                        emb_layer_1_weight = get_embedding_layer(accelerator.unwrap_model(text_encoder_1)).weight
                        placeholder_embeds_1 = emb_layer_1_weight[placeholder_token_ids_1]
                        diag_logs["norm_emb1_placeholder"] = torch.linalg.norm(placeholder_embeds_1).item()

                        # Text Encoder 2
                        emb_layer_2_weight = get_embedding_layer(accelerator.unwrap_model(text_encoder_2)).weight
                        placeholder_embeds_2 = emb_layer_2_weight[placeholder_token_ids_2]
                        diag_logs["norm_emb2_placeholder"] = torch.linalg.norm(placeholder_embeds_2).item()

                        # Text Encoder 3
                        emb_layer_3_weight = get_embedding_layer(accelerator.unwrap_model(text_encoder_3)).weight
                        placeholder_embeds_3 = emb_layer_3_weight[placeholder_token_ids_3]
                        diag_logs["norm_emb3_placeholder"] = torch.linalg.norm(placeholder_embeds_3).item()
                    # accelerator.log(diag_logs, step=global_step) # Changed to print
                    print(f"Step {global_step} [Placeholder Norms]: {diag_logs}")
                # --- END DIAGNOSTICS ---


                # 9. Restore Original Embeddings for non-placeholder tokens
                # This step prevents drift in the frozen parts of the embedding table.
                with torch.no_grad():
                    # Get placeholder token IDs again (they might change if tokenizer reloaded?) - Safer to reuse from start
                    # Create boolean masks for non-placeholder tokens
                    index_no_updates_1 = torch.ones((len(tokenizer_1),), dtype=torch.bool, device=accelerator.device)
                    index_no_updates_1[placeholder_token_ids_1] = False # Set placeholder indices to False

                    index_no_updates_2 = torch.ones((len(tokenizer_2),), dtype=torch.bool, device=accelerator.device)
                    index_no_updates_2[placeholder_token_ids_2] = False

                    index_no_updates_3 = torch.ones((len(tokenizer_3),), dtype=torch.bool, device=accelerator.device)
                    index_no_updates_3[placeholder_token_ids_3] = False


                    # Restore original weights for non-updated indices
                    emb_layer_1 = get_embedding_layer(accelerator.unwrap_model(text_encoder_1))
                    emb_layer_1.weight[index_no_updates_1] = orig_embeds_params_1[index_no_updates_1].to(emb_layer_1.weight.dtype)

                    emb_layer_2 = get_embedding_layer(accelerator.unwrap_model(text_encoder_2))
                    emb_layer_2.weight[index_no_updates_2] = orig_embeds_params_2[index_no_updates_2].to(emb_layer_2.weight.dtype)

                    emb_layer_3 = get_embedding_layer(accelerator.unwrap_model(text_encoder_3))
                    emb_layer_3.weight[index_no_updates_3] = orig_embeds_params_3[index_no_updates_3].to(emb_layer_3.weight.dtype)
                
                # --- DIAGNOSTICS: NON-PLACEHOLDER EMBEDDING RESTORATION CHECK ---
                if accelerator.is_main_process:
                    diag_logs = {}
                    with torch.no_grad():
                        # Text Encoder 1
                        current_emb_layer_1_weight = get_embedding_layer(accelerator.unwrap_model(text_encoder_1)).weight
                        current_non_placeholder_1 = current_emb_layer_1_weight[index_no_updates_1]
                        original_non_placeholder_1 = orig_embeds_params_1[index_no_updates_1].to(current_emb_layer_1_weight.device, dtype=current_emb_layer_1_weight.dtype)
                        diag_logs["norm_diff_emb1_non_placeholder"] = torch.linalg.norm(current_non_placeholder_1 - original_non_placeholder_1).item()

                        # Text Encoder 2
                        current_emb_layer_2_weight = get_embedding_layer(accelerator.unwrap_model(text_encoder_2)).weight
                        current_non_placeholder_2 = current_emb_layer_2_weight[index_no_updates_2]
                        original_non_placeholder_2 = orig_embeds_params_2[index_no_updates_2].to(current_emb_layer_2_weight.device, dtype=current_emb_layer_2_weight.dtype)
                        diag_logs["norm_diff_emb2_non_placeholder"] = torch.linalg.norm(current_non_placeholder_2 - original_non_placeholder_2).item()

                        # Text Encoder 3
                        current_emb_layer_3_weight = get_embedding_layer(accelerator.unwrap_model(text_encoder_3)).weight
                        current_non_placeholder_3 = current_emb_layer_3_weight[index_no_updates_3]
                        original_non_placeholder_3 = orig_embeds_params_3[index_no_updates_3].to(current_emb_layer_3_weight.device, dtype=current_emb_layer_3_weight.dtype)
                        diag_logs["norm_diff_emb3_non_placeholder"] = torch.linalg.norm(current_non_placeholder_3 - original_non_placeholder_3).item()
                    # accelerator.log(diag_logs, step=global_step) # Changed to print
                    print(f"Step {global_step} [Non-Placeholder Diff Norms]: {diag_logs}")
                # --- END DIAGNOSTICS ---


            # End of gradient accumulation block

            # Check if optimization step performed and handle logging/saving/validation
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Log metrics
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                # Save learned embeddings periodically on main process
                if global_step % args.save_steps == 0 and accelerator.is_main_process:
                    logger.info(f"Saving learned embeddings at step {global_step}...")
                    save_path_root = current_run_output_dir # MODIFIED
                    save_progress(
                        text_encoder_1, placeholder_token_ids_1, accelerator, args,
                        os.path.join(save_path_root, f"learned_embeds_t1-steps-{global_step}.safetensors"),
                        safe_serialization=True, embedding_layer=get_embedding_layer(text_encoder_1), text_encoder_name="text_encoder_1"
                    )
                    save_progress(
                        text_encoder_2, placeholder_token_ids_2, accelerator, args,
                        os.path.join(save_path_root, f"learned_embeds_t2-steps-{global_step}.safetensors"),
                        safe_serialization=True, embedding_layer=get_embedding_layer(text_encoder_2), text_encoder_name="text_encoder_2"
                    )
                    save_progress(
                        text_encoder_3, placeholder_token_ids_3, accelerator, args,
                        os.path.join(save_path_root, f"learned_embeds_t3-steps-{global_step}.safetensors"),
                        safe_serialization=True, embedding_layer=get_embedding_layer(text_encoder_3), text_encoder_name="text_encoder_3"
                    )

                # Save training state checkpoints on main process
                if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    save_path = os.path.join(current_run_output_dir, f"checkpoint-{global_step}") # MODIFIED
                    # Handle checkpoint limits
                    if args.checkpoints_total_limit is not None:
                        checkpoints = sorted(
                            [d for d in os.listdir(current_run_output_dir) if d.startswith("checkpoint-")], # MODIFIED
                            key=lambda x: int(x.split("-")[1])
                        )
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[:num_to_remove]
                            logger.info(f"Reached checkpoint limit ({args.checkpoints_total_limit}). Removing {len(removing_checkpoints)} oldest checkpoints:")
                            for ckpt_to_remove in removing_checkpoints:
                                shutil.rmtree(os.path.join(current_run_output_dir, ckpt_to_remove)) # MODIFIED
                                logger.info(f" Removed {ckpt_to_remove}")

                    try:
                       accelerator.save_state(save_path)
                       logger.info(f"Saved training state to {save_path}")
                    except Exception as e:
                         logger.error(f"Failed to save checkpoint to {save_path}: {e}")


                # Run validation on main process
                # --- ACCELERATOR PRINT 1 ---
                accelerator.print(f"ACCELERATOR DEBUG: Step {global_step}, Check Validation? Prompt: {args.validation_prompt is not None}, Modulo: {global_step % args.validation_steps == 0}, Force_Step1: {global_step == 1}, Main Process: {accelerator.is_main_process}")
                if args.validation_prompt is not None and ((global_step % args.validation_steps == 0) or (global_step == 1)) and accelerator.is_main_process:
                    print(f"Running validation at step {global_step}...")
                    # Temporarily set models to eval mode
                    text_encoder_1.eval()
                    text_encoder_2.eval()
                    text_encoder_3.eval()
                    # Transformer and VAE are already frozen and likely in eval mode, but good practice:
                    transformer.eval()
                    vae.eval()

                    # Store validation images
                    validation_images = []
                    try:
                        with torch.no_grad():
                            validation_images = log_validation(
                                text_encoder_1, text_encoder_2, text_encoder_3,
                                tokenizer_1, tokenizer_2, tokenizer_3,
                                transformer, vae,
                                args, accelerator, weight_dtype, epoch # Pass epoch for logging
                            )
                        print(f"Validation finished at step {global_step}.")
                    except Exception as e:
                        print(f"Validation failed at step {global_step}: {e}")
                    finally:
                        # Ensure models return to training mode
                        text_encoder_1.train()
                        text_encoder_2.train()
                        text_encoder_3.train()


            # Check if max training steps reached
            if global_step >= args.max_train_steps:
                break  # Exit inner (step) loop

        # Check again after epoch finishes
        if global_step >= args.max_train_steps:
            logger.info(f"Reached max_train_steps ({args.max_train_steps}). Finishing training.")
            break  # Exit outer (epoch) loop

    # End of training loop

    # Final cleanup and saving
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        logger.info("Training finished. Performing final saves.")
        # Final validation run
        final_images = []
        if args.validation_prompt:
            logger.info("Running final validation...")
            text_encoder_1.eval()
            text_encoder_2.eval()
            text_encoder_3.eval()
            transformer.eval()
            vae.eval()
            try:
                with torch.no_grad():
                    final_images = log_validation(
                        text_encoder_1, text_encoder_2, text_encoder_3,
                        tokenizer_1, tokenizer_2, tokenizer_3,
                        transformer, vae,
                        args, accelerator, weight_dtype, epoch="final", is_final_validation=True
                    )
                logger.info("Final validation finished.")
            except Exception as e:
                logger.error(f"Final validation failed: {e}")


        # Save final embeddings
        save_path_root = current_run_output_dir # MODIFIED
        logger.info("Saving final learned embeddings...")
        save_progress(
            text_encoder_1, placeholder_token_ids_1, accelerator, args,
            os.path.join(save_path_root, "learned_embeds_t1.safetensors"),
            safe_serialization=True, embedding_layer=get_embedding_layer(text_encoder_1), text_encoder_name="text_encoder_1 (final)"
        )
        save_progress(
            text_encoder_2, placeholder_token_ids_2, accelerator, args,
            os.path.join(save_path_root, "learned_embeds_t2.safetensors"),
            safe_serialization=True, embedding_layer=get_embedding_layer(text_encoder_2), text_encoder_name="text_encoder_2 (final)"
        )
        save_progress(
            text_encoder_3, placeholder_token_ids_3, accelerator, args,
            os.path.join(save_path_root, "learned_embeds_t3.safetensors"),
            safe_serialization=True, embedding_layer=get_embedding_layer(text_encoder_3), text_encoder_name="text_encoder_3 (final)"
        )

        # Optionally save full pipeline (marked experimental for SD3 TI)
        # if args.save_as_full_pipeline: ... (code omitted for brevity, complex for TI)

        # Push to Hub if requested
        if args.push_to_hub:
            logger.info("Pushing final results to Hugging Face Hub...")
            # Use the repo_id determined earlier
            hub_repo_id = repo_id if 'repo_id' in locals() else args.hub_model_id # Ensure repo_id exists

            if hub_repo_id:
                 # Create model card
                 try:
                     save_model_card(
                        hub_repo_id,
                        images=final_images, # Use images from final validation
                        base_model=args.pretrained_model_name_or_path,
                        repo_folder=current_run_output_dir, # MODIFIED
                    )
                     logger.info("Generated README.md (model card).")
                 except Exception as e:
                      logger.error(f"Failed to generate model card: {e}")

                 # Upload folder contents
                 try:
                    upload_folder(
                        repo_id=hub_repo_id,
                        folder_path=current_run_output_dir, # MODIFIED
                        commit_message=f"Add SD3 textual inversion weights for {args.placeholder_token}",
                        ignore_patterns=["step_*", "epoch_*", "checkpoint-*"], # Ignore intermediate files
                        token=args.hub_token # Pass token if provided
                    )
                    logger.info(f"Successfully pushed results to hub repository: {hub_repo_id}")
                 except Exception as e:
                    logger.error(f"Error pushing to hub repository '{hub_repo_id}': {e}")
            else:
                 logger.warning("Cannot push to hub: Hub repository ID is not defined (check --hub_model_id or initial repo creation).")


    accelerator.end_training()
    logger.info("Accelerator ended training.")


if __name__ == "__main__":
    main() 