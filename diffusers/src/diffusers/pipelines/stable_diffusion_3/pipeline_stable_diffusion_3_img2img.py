# Copyright 2024 Stability AI and The HuggingFace Team. All rights reserved.
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
# limitations under the License.

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import PIL.Image
import torch
import numpy as np
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

from ...image_processor import PipelineImageInput, VaeImageProcessor
from ...loaders import FromSingleFileMixin, SD3LoraLoaderMixin
from ...models.autoencoders import AutoencoderKL
from ...models.transformers import SD3Transformer2DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import StableDiffusion3PipelineOutput
import os, pickle
import copy

from ...training_utils import compute_loss_weighting_for_sd3, compute_density_for_timestep_sampling

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch

        >>> from diffusers import AutoPipelineForImage2Image
        >>> from diffusers.utils import load_image

        >>> device = "cuda"
        >>> model_id_or_path = "stabilityai/stable-diffusion-3-medium-diffusers"
        >>> pipe = AutoPipelineForImage2Image.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
        >>> pipe = pipe.to(device)

        >>> url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
        >>> init_image = load_image(url).resize((1024, 1024))

        >>> prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"

        >>> images = pipe(prompt=prompt, image=init_image, strength=0.95, guidance_scale=7.5).images[0]
        ```
"""


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        # print("here") # here
        # exit(0)
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        # return encoder_output.latent_dist.mode()
        raise NotImplementedError("Argmax sampling is not supported for VQ-VAE models")
    elif hasattr(encoder_output, "latents"):
        # return encoder_output.latents
        raise NotImplementedError("latents sampling is not supported for VQ-VAE models")
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else: # here I think 
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        
    return timesteps, num_inference_steps


class StableDiffusion3Img2ImgPipeline(DiffusionPipeline, SD3LoraLoaderMixin, FromSingleFileMixin):
    r"""
    Args:
        transformer ([`SD3Transformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModelWithProjection`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant,
            with an additional added projection layer that is initialized with a diagonal matrix with the `hidden_size`
            as its dimension.
        text_encoder_2 ([`CLIPTextModelWithProjection`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the
            [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
            variant.
        text_encoder_3 ([`T5EncoderModel`]):
            Frozen text-encoder. Stable Diffusion 3 uses
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel), specifically the
            [t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`CLIPTokenizer`):
            Second Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_3 (`T5TokenizerFast`):
            Tokenizer of class
            [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->text_encoder_3->transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds", "negative_pooled_prompt_embeds"]

    def __init__(
        self,
        transformer: SD3Transformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer_2: CLIPTokenizer,
        text_encoder_3: T5EncoderModel,
        tokenizer_3: T5TokenizerFast,
        resize: Optional[int] = 1024,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            text_encoder_3=text_encoder_3,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            tokenizer_3=tokenizer_3,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, vae_latent_channels=self.vae.config.latent_channels
        )
        # print(self.vae_scale_factor) # 8
        # exit(0)
        self.tokenizer_max_length = self.tokenizer.model_max_length
        self.default_sample_size = self.transformer.config.sample_size
        # try:
        self.resize = resize
        if self.resize == 1024:
            # print("here1")
            if os.path.exists(os.path.expanduser("~/diffusers/noise_samples_sd3.pkl")):
                    # print("here1")
                with open(os.path.expanduser("~/diffusers/noise_samples_sd3.pkl"), "rb") as f:
                    self.noise_samples = pickle.load(f)
                    assert self.noise_samples.shape == (1000,16, 128, 128)
                    print('Loaded noise samples from cache with shape: ', self.noise_samples.shape)
            elif os.path.exists(os.path.join(os.getcwd(),"diffusers/noise_samples_sd3.pkl")):
                # print("here2")
                with open(os.path.join(os.getcwd(),"diffusers/noise_samples_sd3.pkl"), "rb") as f:
                    self.noise_samples = pickle.load(f)
                    assert self.noise_samples.shape == (1000,16, 128, 128)
                    print('Loaded noise samples from cache with shape: ', self.noise_samples.shape)        
                        # pickle.dump(self.noise_samples, f))
            else:
                # if not, generate noise_samples a list of random gaussian samples of shape (4,64,64) and cache them
                self.noise_samples = torch.randn(1000,16, 128, 128)
                try:
                    with open(os.path.expanduser("~/diffusers/noise_samples_sd3.pkl"), "wb") as f:
                        pickle.dump(self.noise_samples, f)
                except FileNotFoundError:
                    current_directory = os.getcwd()
                    with open(os.path.join(current_directory,"diffusers/noise_samples_sd3.pkl"), "wb") as f:
                        pickle.dump(self.noise_samples, f)
                print('Generated noise samples with shape: ', self.noise_samples.shape)
        elif self.resize == 512:
            print("here2")
            if os.path.exists(os.path.expanduser("~/diffusers/noise_samples_sd3_2.pkl")):
                    # print("here1")
                with open(os.path.expanduser("~/diffusers/noise_samples_sd3_2.pkl"), "rb") as f:
                    self.noise_samples = pickle.load(f)
                    assert self.noise_samples.shape == (1000,16, 64, 64)
                    print('Loaded noise samples from cache with shape: ', self.noise_samples.shape)
            elif os.path.exists(os.path.join(os.getcwd(),"diffusers/noise_samples_sd3_2.pkl")):
                # print("here2")
                with open(os.path.join(os.getcwd(),"diffusers/noise_samples_sd3_2.pkl"), "rb") as f:
                    self.noise_samples = pickle.load(f)
                    assert self.noise_samples.shape == (1000,16, 64, 64)
                    print('Loaded noise samples from cache with shape: ', self.noise_samples.shape)        
                        # pickle.dump(self.noise_samples, f))
            else:
                # if not, generate noise_samples a list of random gaussian samples of shape (4,64,64) and cache them
                self.noise_samples = torch.randn(1000,16, 64, 64)
                try:
                    with open(os.path.expanduser("~/diffusers/noise_samples_sd3_2.pkl"), "wb") as f:
                        pickle.dump(self.noise_samples, f)
                except FileNotFoundError:
                    current_directory = os.getcwd()
                    with open(os.path.join(current_directory,"diffusers/noise_samples_sd3_2.pkl"), "wb") as f:
                        pickle.dump(self.noise_samples, f)
                print('Generated noise samples with shape: ', self.noise_samples.shape)
    # Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3.StableDiffusion3Pipeline._get_t5_prompt_embeds
        else: 
            raise ValueError("Resize should be 1024 or 512")
        # exit(0)
    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        noise_scheduler_copy = copy.deepcopy(self.scheduler)
        sigmas = noise_scheduler_copy.sigmas.to(device=self._execution_device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(self._execution_device)
        timesteps = timesteps.to(self._execution_device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 256,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if self.text_encoder_3 is None:
            return torch.zeros(
                (
                    batch_size * num_images_per_prompt,
                    self.tokenizer_max_length,
                    self.transformer.config.joint_attention_dim,
                ),
                device=device,
                dtype=dtype,
            )

        text_inputs = self.tokenizer_3(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_3(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_3.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder_3(text_input_ids.to(device))[0]

        dtype = self.text_encoder_3.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3.StableDiffusion3Pipeline._get_clip_prompt_embeds
    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        clip_skip: Optional[int] = None,
        clip_model_index: int = 0,
    ):
        device = device or self._execution_device

        clip_tokenizers = [self.tokenizer, self.tokenizer_2]
        clip_text_encoders = [self.text_encoder, self.text_encoder_2]

        tokenizer = clip_tokenizers[clip_model_index]
        text_encoder = clip_text_encoders[clip_model_index]

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )
        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
        
        pooled_prompt_embeds = prompt_embeds[0]

        if clip_skip is None:
            prompt_embeds = prompt_embeds.hidden_states[-2]
        else:
            prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds, pooled_prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3.StableDiffusion3Pipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]],
        prompt_3: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        clip_skip: Optional[int] = None,
        max_sequence_length: int = 256,
        lora_scale: Optional[float] = None,
    ):
        r"""

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in all text-encoders
            prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_3` and `text_encoder_3`. If not defined, `prompt` is
                used in all text-encoders
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in all the text-encoders.
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_3` and
                `text_encoder_3`. If not defined, `negative_prompt` is used in both text-encoders
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, SD3LoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)
            if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            prompt_3 = prompt_3 or prompt
            prompt_3 = [prompt_3] if isinstance(prompt_3, str) else prompt_3

            prompt_embed, pooled_prompt_embed = self._get_clip_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip,
                clip_model_index=0,
            )
            prompt_2_embed, pooled_prompt_2_embed = self._get_clip_prompt_embeds(
                prompt=prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip,
                clip_model_index=1,
            )
            clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)

            t5_prompt_embed = self._get_t5_prompt_embeds(
                prompt=prompt_3,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )
            # print(clip_prompt_embeds.shape) # torch.Size([16, 77, 2048])
            # print(t5_prompt_embed.shape) # torch.Size([16, 256, 4096])
            # exit(0)

            clip_prompt_embeds = torch.nn.functional.pad(
                clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
            )
            # print(clip_prompt_embeds.shape) # torch.Size([16, 77, 4096])
            # exit(0)

            prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
            # print(prompt_embeds.shape) # torch.Size([16, 333, 4096])
            # exit(0)
            pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt
            negative_prompt_3 = negative_prompt_3 or negative_prompt

            # normalize str to list
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_2 = (
                batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
            )
            negative_prompt_3 = (
                batch_size * [negative_prompt_3] if isinstance(negative_prompt_3, str) else negative_prompt_3
            )

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embed, negative_pooled_prompt_embed = self._get_clip_prompt_embeds(
                negative_prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=None,
                clip_model_index=0,
            )
            negative_prompt_2_embed, negative_pooled_prompt_2_embed = self._get_clip_prompt_embeds(
                negative_prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=None,
                clip_model_index=1,
            )
            negative_clip_prompt_embeds = torch.cat([negative_prompt_embed, negative_prompt_2_embed], dim=-1)

            t5_negative_prompt_embed = self._get_t5_prompt_embeds(
                prompt=negative_prompt_3,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

            negative_clip_prompt_embeds = torch.nn.functional.pad(
                negative_clip_prompt_embeds,
                (0, t5_negative_prompt_embed.shape[-1] - negative_clip_prompt_embeds.shape[-1]),
            )

            negative_prompt_embeds = torch.cat([negative_clip_prompt_embeds, t5_negative_prompt_embed], dim=-2)
            negative_pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embed, negative_pooled_prompt_2_embed], dim=-1
            )

        if self.text_encoder is not None:
            if isinstance(self, SD3LoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, SD3LoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def check_inputs(
        self,
        prompt,
        prompt_2,
        prompt_3,
        strength,
        negative_prompt=None,
        negative_prompt_2=None,
        negative_prompt_3=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        if strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_3 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_3`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")
        elif prompt_3 is not None and (not isinstance(prompt_3, str) and not isinstance(prompt_3, list)):
            raise ValueError(f"`prompt_3` has to be of type `str` or `list` but is {type(prompt_3)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_2 is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_2`: {negative_prompt_2} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_3 is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_3`: {negative_prompt_3} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )

        if negative_prompt_embeds is not None and negative_pooled_prompt_embeds is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_pooled_prompt_embeds` also have to be passed. Make sure to generate `negative_pooled_prompt_embeds` from the same text encoder that was used to generate `negative_prompt_embeds`."
            )

        if max_sequence_length is not None and max_sequence_length > 512:
            raise ValueError(f"`max_sequence_length` cannot be greater than 512 but is {max_sequence_length}")

    def get_timesteps(self, num_inference_steps, strength, device, itm=False):
        # get the original timestep using init_timestep
        init_timestep = min(num_inference_steps * strength, num_inference_steps)

        t_start = int(max(num_inference_steps - init_timestep, 0))
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        if not itm:
            if hasattr(self.scheduler, "set_begin_index"):
                self.scheduler.set_begin_index(t_start * self.scheduler.order)

        return timesteps, num_inference_steps - t_start
    
    def reinit_noise(self):
        if hasattr(self, 'timesteps') and self.timesteps is not None:
            # get noises shape and re-init the timesteps we're using already.
            noise_shape = (len(self.timesteps), *self.noise_samples[0].shape)
            
            self.noise_samples[:len(self.timesteps)] = torch.randn(noise_shape, device=self.device)


    def prepare_latents(self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None, sampling_step=None, presaved_latents=None):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )
        if presaved_latents is None:
            image = image.to(device=device, dtype=dtype)

            batch_size = batch_size * num_images_per_prompt
            # if image.shape[1] == self.vae.config.latent_channels:
            #     init_latents = image

            # else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            if isinstance(generator, list):
                # print("here?") # not here
                # exit(0)
                    init_latents = [
                        retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                        for i in range(batch_size)
                    ]
                    init_latents = torch.cat(init_latents, dim=0)
                    # print("here1")
            else:
                    # print(self.vae.__class__.__name__)
                    # exit(0)
                    # print("here2") # here
                    # print("image" , torch.mean(image, dim=(1,2,3)))
                    # print(image.shape) # # torch.Size([4, 3, 1024, 1024])
                    # exit(0)
                    init_latents = retrieve_latents(self.vae.encode(image), generator=generator)
            # exit(0)
            init_latents = (init_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
            # print(torch.mean(init_latents, dim=(1,2,3)))
            
            if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
                
                # expand init_latents for batch_size
                additional_image_per_prompt = batch_size // init_latents.shape[0]
                init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
            elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
                raise ValueError(
                    f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
                )
            else:
                # print("here?") # here
                # exit(0)
                init_latents = torch.cat([init_latents], dim=0)
        else:
            init_latents = presaved_latents
        shape = init_latents.shape
        noise = self.noise_samples[sampling_step] # (16,64,64)
        noise = noise.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # (B, 16, 64, 64)
        noise = noise.to(device=device, dtype=dtype)

        # noise = randn_tensor((1, shape[1],shape[2],shape[3]), generator=generator, device=device, dtype=dtype)
        # noise = noise.repeat(batch_size, 1, 1, 1)

        init_latents = init_latents.to(device=device, dtype=dtype)
        if self.scheduler.__class__.__name__ != "FlowMatchEulerDiscreteScheduler":
            scaled_latents = self.scheduler.add_noise(init_latents, noise, timestep)
            sigma = None
        else: scaled_latents, sigma = self.scheduler.scale_noise(init_latents, timestep, noise)
        # print(init_latents.shape) # torch.Size([4, 16, 64, 64])
        # print(noise.shape)
        # exit(0)
        # else:
        #     init_latents = self.scheduler.add_noise(original_samples = init_latents, timesteps = timestep, noise = noise)
        
        return scaled_latents, noise, init_latents, sigma

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def clip_skip(self):
        return self._clip_skip

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    def sampled_timesteps(self, num_inference_steps, strength, sampling_steps, device, later_timesteps=False):
        if not later_timesteps:
            rate = int(1000 / sampling_steps)
            init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
            t_start = max(num_inference_steps - init_timestep, 0)
            t_end = len(self.scheduler.timesteps) - t_start
            timesteps_indices = np.arange(t_start * self.scheduler.order, t_end, rate)
            timesteps = self.scheduler.timesteps[timesteps_indices].tolist()[::-1]
        else:
            # rate = int(1000 / sampling_steps)
            # init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
            # # t_start = max(num_inference_steps - init_timestep, 0)
            # t_start = max(num_inference_steps - init_timestep, 500)
            # t_end = len(self.scheduler.timesteps) - t_start

            # # print(t_start, t_end) #20 #980
            # # exit(0) 
            # timesteps_indices = np.arange(t_start * self.scheduler.order, t_end, rate)
            # timesteps = self.scheduler.timesteps[timesteps_indices].tolist()
            # Get the timesteps as a numpy array
            timesteps_array = np.array(self.scheduler.timesteps)

            # Select timesteps within the 500 to 1000 range
            valid_timesteps = timesteps_array[(timesteps_array >= 500) & (timesteps_array <= 1000)]

            # If fewer than 30 are available, handle gracefully
            if len(valid_timesteps) < 30:
                raise ValueError(f"Only {len(valid_timesteps)} timesteps available in range 500-1000")

            # Select 30 timesteps evenly spaced within the valid range
            selected_indices = np.linspace(0, len(valid_timesteps) - 1, 30).astype(int)
            timesteps = valid_timesteps[selected_indices].tolist()
            # print(timesteps)
            # exit(0)
        # print(timesteps)
        # [65.965576171875, 151.0791473388672, 226.65536499023438, 294.21221923828125, 354.96185302734375, 409.88372802734375, 459.778076171875, 505.3050537109375, 547.0139770507812, 585.3658447265625, 620.7503051757812, 653.4988403320312, 683.8955078125, 712.1848754882812, 738.5786743164062, 763.2612915039062, 786.3938598632812, 808.1181030273438, 828.55859375, 847.8260498046875, 866.0185546875, 883.2236328125, 899.5195922851562, 914.9765625, 929.6577758789062, 943.6200561523438, 956.9152221679688, 969.5897827148438, 981.6862182617188, 993.2432861328125]
        return timesteps
    
    def presave_latents(self, image):
        init_latents = retrieve_latents(self.vae.encode(image), generator=None)
        init_latents = (init_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        # init_latents = retrieve_latents(self.vae.encode(image), generator=None)
        # init_latents = self.vae.config.scaling_factor * init_latents
        return init_latents

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        image: PipelineImageInput = None,
        strength: float = 0.6,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
        sampling_steps: int = 250,
        unconditional=False,
        middle_step=False,
        time_weighting=None,
        imgs_visulalize=None,
        presaved_latents=None,
        save_noise = None,
        save_noises=False,
        use_normed_classifier=False,
        use_euler=False,
        later_timesteps = False
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead
            prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_3` and `text_encoder_3`. If not defined, `prompt` is
                will be used instead
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used instead
            negative_prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_3` and
                `text_encoder_3`. If not defined, `negative_prompt` is used instead
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_3.StableDiffusion3PipelineOutput`] instead of
                a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
               A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 256): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion_3.StableDiffusion3PipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion_3.StableDiffusion3PipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            prompt_3,
            strength,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=self.do_classifier_free_guidance or unconditional,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            clip_skip=self.clip_skip,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        if self.do_classifier_free_guidance or unconditional:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
        # print(image.shape)
        # print(torch.mean(image, dim=(1,2,3)))
        # 3. Preprocess image
        if presaved_latents is not None:
            image = presaved_latents
        else:
            image = self.image_processor.preprocess(image)
        # print(image)
        # print(type(image))
        # exit(0)
        # print()

        # 4. Prepare timesteps
        
        # print(f"second timestep: {timesteps}")
        # exit(0)
        # latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        # # print(timesteps)
        # # exit(0)
        # # 5. Prepare latent variables

        # # latent_timestep = torch.tensor([t] * batch_size, device=device)
        #         # print(latent_timestep)
        #         # exit(0)
        
        # # rate = int(1000 / sampling_steps)
        # # timesteps = list(range(10, 990, rate))
        
        # if latents is None:
        #     latents = self.prepare_latents(
        #         image,
        #         latent_timestep,
        #         batch_size,
        #         num_images_per_prompt,
        #         prompt_embeds.dtype,
        #         device,
        #         generator,
        #     )[0]


        dists = []
        # num_inference_steps= sampling_steps
        
        if time_weighting is None:
            if not later_timesteps:
                self.timesteps = self.sampled_timesteps(num_inference_steps, strength, sampling_steps, device)
            else:
                self.timesteps = self.sampled_timesteps(num_inference_steps, strength, sampling_steps, device, later_timesteps=True)
        #     timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
            # timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device, itm=True)
#             tensor([1000.0000,  993.1244,  986.0571,  978.7898,  971.3141,  963.6208,
#          955.7004,  947.5425,  939.1364,  930.4705,  921.5326,  912.3097,
#          902.7879,  892.9525,  882.7878,  872.2768,  861.4016,  850.1429,
#          838.4802,  826.3912,  813.8521,  800.8373,  787.3193,  773.2682,
#          758.6519,  743.4356,  727.5817,  711.0490,  693.7931,  675.7655,
#          656.9132,  637.1784,  616.4975,  594.8009,  572.0118,  548.0456,
#          522.8087,  496.1973,  468.0958,  438.3757,  406.8929,  373.4857,
#          337.9722,  300.1466,  259.7757,  216.5935,  170.2957,  120.5327,
#           66.9002,    8.9286], device='cuda:0')
            
# 50    
        else:    
            # print("here?")
            u = compute_density_for_timestep_sampling(
                    weighting_scheme=time_weighting,
                    batch_size=num_inference_steps,
                    logit_mean=0.0,
                    logit_std=1.0,
                    mode_scale=1.29,
                )
            # tensor([794.4498, 822.4932, 562.5000, 600.7194, 804.9120, 961.5936, 811.2947,
            # 660.1344, 679.4310, 688.3116, 727.6005, 465.5172, 495.9839, 793.7852,
            # 895.1613, 920.4018, 630.9385, 576.3546, 622.8069, 934.3891, 808.7557,
            # 320.7547, 653.4988, 778.8463, 781.6093, 714.6596, 428.5715, 380.5970,
            # 538.4615, 874.9999, 542.1456, 461.2188, 830.9544, 701.2780, 791.1153,
            # 816.3173, 635.9446, 816.9398, 709.6945, 824.9324, 842.1053, 839.7888,
            # 933.1067, 911.7646, 328.1250, 839.2070, 919.0551, 713.8365, 763.2613,
            # 831.5508], device='cuda:0')
            indices = (u * self.scheduler.config.num_train_timesteps).long()
            self.timesteps = self.scheduler.timesteps[indices].to(device=device)
        # print(len(self.timesteps))
        # exit(0)
        # print(timesteps)
        # exit(0)
        # print(timesteps)
        # print(len(timesteps)) # 30
        # exit(0)
        # print(type(generator))
        # exit(0)
        target_gaussian_noises_per_timestep = {}
        cond_noises_per_timestep = {}
        uncond_noises_per_timestep = {}
        for sampling_step, t in enumerate(self.timesteps):
        # num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        # self._num_timesteps = len(timesteps)
        # with self.progress_bar(total=num_inference_steps) as progress_bar:
            # for sampling_step, t in enumerate(timesteps):
                # if self.interrupt:
                #     continue

                # expand the latents if we are doing classifier free guidance
                
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                
                latent_timestep = torch.tensor([t] * batch_size, device=device)
                latents, noise, init_latents, sigmas = self.prepare_latents(
                        image,
                        latent_timestep,
                        batch_size,
                        num_images_per_prompt,
                        prompt_embeds.dtype,
                        device,
                        generator,
                        sampling_step = sampling_step,
                        presaved_latents=presaved_latents
                    )
                # print(f"sigmas1 {sigmas}")
                

                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance or unconditional else latents
                t_org = t
                t = torch.tensor(t, dtype=torch.float, device=device)
                timestep = t.expand(latent_model_input.shape[0])

                if self.scheduler.__class__.__name__ != "FlowMatchEulerDiscreteScheduler":
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t, itm=True) # different
                
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # from math import ceil, sqrt
                # import matplotlib.pyplot as plt
                # from matplotlib.colors import LinearSegmentedColormap
                # import numpy as np

                # # Define a custom colormap from white to red
                # white_to_red = LinearSegmentedColormap.from_list("white_to_red", ["white", "red"])

                # dist = torch.norm(noise_pred - (noise- init_latents), keepdim=True, p=2, dim = 1)
                # # print(dist.shape) # torch.Size([4, 1, 64, 64])
                # # exit(0)
                
                # # Example values for illustration
                # num_samples = dist.shape[0]
                # rows, cols = 2, num_samples  # Adjust according to the desired grid
                
                # fig, axes = plt.subplots(rows, cols, figsize=(8, 8))


                # for i in range(num_samples):
                #     # Select sample and squeeze channel dimension
                #     sample = dist[i, 0, :, :].cpu().numpy()  # Use [0, :, :] to handle the channel dimension
                #     axes[0, i].imshow(sample, cmap=white_to_red)  # Show sample in the first row
                #     axes[0, i].axis('off')
                #     axes[0, i].set_title(f"{prompt[i]}", fontsize=8)
                #     if imgs_visulalize is not None:
                #         img_vis = imgs_visulalize[0][i].cpu().numpy().transpose(1, 2, 0)  # Convert to numpy array
                #         axes[1, i].imshow(img_vis)  # Show imgs_visualize in the second row
                #         axes[1, i].axis('off')
                #         axes[1, i].set_title(f"mean_{np.mean(sample)}\nmin_{np.min(sample)}\nmax_{np.max(sample)}", fontsize=8)
                #         # axes[1, i].set_title(f"Visualize {prompt[i]}")

                #     else: axes[1, i].axis('off')

                # plt.tight_layout()
                # task_name = f"{imgs_visulalize[1][3]}_sd3_{time_weighting}"
                # os.makedirs(f"{task_name}_images", exist_ok=True)
                # fig.savefig(f"{task_name}_images/Batch_{imgs_visulalize[1][0]}_Text{imgs_visulalize[1][1]}_Image{imgs_visulalize[1][2]}_time_{t}.png", dpi=300, bbox_inches='tight')  # Save the first prompt's name
                # plt.close(fig)  # Close the figure to free memory

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    weighting = compute_loss_weighting_for_sd3(weighting_scheme=time_weighting, sigmas=sigmas)
                    target = noise - init_latents
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    
                    if use_normed_classifier:
                        dist = weighting.float().flatten() * torch.norm((noise_pred - target).reshape(target.shape[0], -1), p=2, dim=1)
                    else:
                        dist = weighting.float().flatten() * ((noise_pred - target) ** 2).reshape(target.shape[0], -1).mean(dim=-1)
                    dists.append(dist)
                    
                    if save_noises:
                        if t_org not in target_gaussian_noises_per_timestep:
                            # only save once since it's always same
                            target_gaussian_noises_per_timestep[t_org] = target
                        cond_noises_per_timestep[t_org] = noise_pred_text
                        uncond_noises_per_timestep[t_org] = noise_pred_uncond

                elif unconditional:
                    # assert not use_normed_classifier, "Not implemented use normed classifier for unconditional"
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    weighting = compute_loss_weighting_for_sd3(weighting_scheme=time_weighting, sigmas=sigmas)
                    target = noise - init_latents
                    if use_normed_classifier:
                        dist = weighting.float().flatten() * torch.norm((noise_pred_text - target).reshape(target.shape[0], -1), p=2, dim=1)
                        base_dist = weighting.float().flatten() * torch.norm((noise_pred_uncond - target).reshape(target.shape[0], -1), p=2, dim=1)
                    else:
                        dist = weighting.float().flatten() * ((noise_pred_text - target) ** 2).reshape(target.shape[0], -1).mean(dim=-1)
                        base_dist = weighting.float().flatten() * ((noise_pred_uncond - target) ** 2).reshape(target.shape[0], -1).mean(dim=-1)

                    dists.append(dist-base_dist)

                else:
                    try:
                        weighting = compute_loss_weighting_for_sd3(weighting_scheme=time_weighting, sigmas=sigmas)
                    except:
                        weighting = torch.ones((batch_size, 1), device=device)
                    target = noise - init_latents
                    if use_normed_classifier:
                        dist = weighting.float().flatten() * torch.norm((noise_pred - target).reshape(target.shape[0], -1), p=2, dim=1)
                    else:
                        dist = weighting.float().flatten() * ((noise_pred - target) ** 2).reshape(target.shape[0], -1).mean(dim=-1)
                    dists.append(dist)

        dists = torch.stack(dists).permute(1, 0)

        return dists, cond_noises_per_timestep, uncond_noises_per_timestep, target_gaussian_noises_per_timestep
        #             noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

        #         # compute the previous noisy sample x_t -> x_t-1
        #         latents_dtype = latents.dtype
                
        #         latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        #         if latents.dtype != latents_dtype:
        #             if torch.backends.mps.is_available():
        #                 # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
        #                 latents = latents.to(latents_dtype)

        #         if callback_on_step_end is not None:
        #             callback_kwargs = {}
        #             for k in callback_on_step_end_tensor_inputs:
        #                 callback_kwargs[k] = locals()[k]
        #             callback_outputs = callback_on_step_end(self, sampling_step, t, callback_kwargs)

        #             latents = callback_outputs.pop("latents", latents)
        #             prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
        #             negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
        #             negative_pooled_prompt_embeds = callback_outputs.pop(
        #                 "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
        #             )

        #         # call the callback, if provided
        #         # if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
        #         #     progress_bar.update()

        #         if XLA_AVAILABLE:
        #             xm.mark_step()

        # if output_type == "latent":
        #     image = latents

        # else:
        #     latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        #     image = self.vae.decode(latents, return_dict=False)[0]
        #     image = self.image_processor.postprocess(image, output_type=output_type)

        # # Offload all models
        # self.maybe_free_model_hooks()

        # if not return_dict:
        #     return (image,)

        # return StableDiffusion3PipelineOutput(images=image)