# Copyright 2024 Black Forest Labs and The HuggingFace Team. All rights reserved.
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
import re
from typing import Any, Callable, Dict, List, Literal, Optional, Union
import cv2
import numpy as np
import PIL.Image
import torch
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FluxLoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.transformers import FluxTransformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers import DiffusionPipeline
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.pipelines.flux.pipeline_flux_inpaint import FluxInpaintPipeline
import torch.nn.functional as F
import torchvision
import os
from cache_functions import cache_init
from region_utils.ui_utils import region_pair_to_pts
from region_utils.drag import blur_source,reverse_and_repeat_every_n_elements


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import FluxInpaintPipeline
        >>> from diffusers.utils import load_image

        >>> pipe = FluxInpaintPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")
        >>> prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
        >>> img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
        >>> mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
        >>> source = load_image(img_url)
        >>> mask = load_image(mask_url)
        >>> image = pipe(prompt=prompt, image=source, ref_segment=mask).images[0]
        >>> image.save("flux_inpainting.png")
        ```
"""
def process_point_pairs(source_pts, target_pts, height, width, kernel_size=3):
    
    def process_points(pts, h, w):
        mask = np.zeros((h//16, w//16))
        for pt in pts:
            x, y = pt
            mask[x, y] = 1
            
        kernel = np.ones((kernel_size,kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        return torch.tensor(np.argwhere(mask > 0))
    
    source_pts = process_points(source_pts, height, width)
    target_pts = process_points(target_pts, height, width)
    
    
    return source_pts, target_pts

# Copied from diffusers.pipelines.flux.pipeline_flux.calculate_shift
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu



# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
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
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps



def unregister(*handlers):
    for handler in handlers:
        handler.remove()
    torch.cuda.empty_cache()

class FluxDragEditPipeline(FluxInpaintPipeline):
    r"""
    The Flux pipeline for image composition.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    Args:
        transformer ([`FluxTransformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        text_encoder_2 ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`T5TokenizerFast`):
            Second Tokenizer of class
            [T5TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5TokenizerFast).
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds"]
    def controlled_forward_ode(self, 
                               image_latents,
                               latent_image_ids,
                               sigmas,
                               gamma,
                               null_prompt_embeds,
                               null_pooled_prompt_embeds,
                               null_text_ids,
                               timesteps,
                               guidance,
                               height,
                               width,
                               joint_attention_kwargs,
                               skip_T=3,
                               ):

        """
        Eq 8 dY_t = [u_t(Y_t) + γ(u_t(Y_t|y_1) - u_t(Y_t))]dt
        """
        device = image_latents.device
        batch_size = image_latents.shape[0]
        Y_t = image_latents.clone()
        y_1 = torch.randn_like(Y_t)
        N = len(sigmas)
        inv_latents_trajectory=[]
        current = joint_attention_kwargs['current']
        if guidance is not None:
            guidance = guidance.expand(batch_size)
        
        for i in range(N-1): # enumerate(timesteps):
            current['step'] = i
            t_i = torch.tensor(i / (N), dtype=Y_t.dtype, device=device)
            dt = torch.tensor(1 / (N), dtype=Y_t.dtype, device=device)
            if i % skip_T == 0 or i == N-2:
                u_t_i = self.transformer(
                    hidden_states=Y_t, 
                    timestep=t_i.repeat(batch_size), 
                    guidance=guidance,
                    pooled_projections=null_pooled_prompt_embeds,
                    encoder_hidden_states=null_prompt_embeds,
                    txt_ids=null_text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=joint_attention_kwargs,
                    return_dict=False,                
                )[0]
                
            u_t_i_cond = (y_1 - Y_t) / (1 - t_i)
            u_hat_t_i = u_t_i + gamma * (u_t_i_cond - u_t_i)
            Y_t = Y_t + u_hat_t_i * dt # (sigmas[i] - sigmas[i+1])
            inv_latents_trajectory.append(Y_t)
        return {'inv_latents':Y_t, 'inv_latents_trajectory':inv_latents_trajectory}
    
        
    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModel,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels)) if hasattr(self, "vae") and self.vae is not None else 16
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            vae_latent_channels=self.vae.config.latent_channels,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )
        self.default_sample_size = 64

    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._get_t5_prompt_embeds
    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer_2)

        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_2.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])

        prompt_embeds = self.text_encoder_2(text_input_ids.to(device), output_hidden_states=False)[0]

        dtype = self.text_encoder_2.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._get_clip_prompt_embeds
    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
    ):
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
        prompt_embeds = self.text_encoder(text_input_ids.to(device), output_hidden_states=False)

        # Use pooled output of CLIPTextModel
        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds

    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 512,
        lora_scale: Optional[float] = None,
    ):
        r"""

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in all text-encoders
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, FluxLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)
            if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            # We only use the pooled prompt output from the CLIPTextModel
            pooled_prompt_embeds = self._get_clip_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
            )
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

        if self.text_encoder is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        dtype = self.text_encoder.dtype if self.text_encoder is not None else self.transformer.dtype
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

        return prompt_embeds, pooled_prompt_embeds, text_ids

    # Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3_inpaint.StableDiffusion3InpaintPipeline._encode_vae_image
    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator)

        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        return image_latents

    # Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3_img2img.StableDiffusion3Img2ImgPipeline.get_timesteps
    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(num_inference_steps * strength, num_inference_steps)

        t_start = int(max(num_inference_steps - init_timestep, 0))
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        sigmas = self.scheduler.sigmas[t_start * self.scheduler.order :]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)

        return timesteps, num_inference_steps - t_start, sigmas
    
    def check_inputs(
        self,
        *args,
        **kwargs
    ):
        return
    @staticmethod
    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._prepare_latent_image_ids
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height // 2, width // 2, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)
    @staticmethod
    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._pack_latents
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    @staticmethod
    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._unpack_latents
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        height = height // vae_scale_factor
        width = width // vae_scale_factor

        latents = latents.view(batch_size, height, width, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height * 2, width * 2)

        return latents
    def prepare_latents(
        self,
        image,
        timestep,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        height = 2 * (int(height) // self.vae_scale_factor)
        width = 2 * (int(width) // self.vae_scale_factor)

        shape = (batch_size, num_channels_latents, height, width)
        latent_image_ids = self._prepare_latent_image_ids(batch_size, height, width, device, dtype)

        image = image.to(device=device, dtype=dtype)
        image_latents = self._encode_vae_image(image=image, generator=generator)

        if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
            # expand init_latents for batch_size
            additional_image_per_prompt = batch_size // image_latents.shape[0]
            image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            image_latents = torch.cat([image_latents], dim=0)

        if latents is None:
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = self.scheduler.scale_noise(image_latents, timestep, noise)
        else:
            noise = latents.to(device)
            latents = noise

        noise = self._pack_latents(noise, batch_size, num_channels_latents, height, width)
        image_latents = self._pack_latents(image_latents, batch_size, num_channels_latents, height, width)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
        return latents, noise, image_latents, latent_image_ids
    def prepare_mask_latents(
        self,
        mask,
        masked_image,
        batch_size,
        num_channels_latents,
        num_images_per_prompt,
        height,
        width,
        dtype,
        device,
        generator,
    ):
        height = 2 * (int(height) // self.vae_scale_factor)
        width = 2 * (int(width) // self.vae_scale_factor)
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(mask, size=(height, width))
        mask = mask.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt

        masked_image = masked_image.to(device=device, dtype=dtype)
        # print(f"masked_image.shape: {masked_image.shape}")
        if masked_image.shape[1] == 16:
            masked_image_latents = masked_image
        else:
            masked_image_latents = retrieve_latents(self.vae.encode(masked_image), generator=generator)

        masked_image_latents = (masked_image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
        if masked_image_latents.shape[0] < batch_size:
            if not batch_size % masked_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            masked_image_latents = masked_image_latents.repeat(batch_size // masked_image_latents.shape[0], 1, 1, 1)

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)

        masked_image_latents = self._pack_latents(
            masked_image_latents,
            batch_size,
            num_channels_latents,
            height,
            width,
        )
        mask = self._pack_latents(
            mask.repeat(1, num_channels_latents, 1, 1),
            batch_size,
            num_channels_latents,
            height,
            width,
        )

        return mask, masked_image_latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt
    
    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def gen(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        neg_prompt: Optional[Union[str, List[str]]] = None,
        neg_prompt_2: Optional[Union[str, List[str]]] = None,
        
        main_image: PipelineImageInput = None,
        src_region: PipelineImageInput = None,
        tgt_region: PipelineImageInput = None,
        src_pts: Optional[torch.Tensor] = None,
        tgt_pts: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        
        height: Optional[int] = None,
        width: Optional[int] = None,
        eta: float = 1.0,
        gamma: float = 1.0,
        start_timestep: int = 0,
        stop_timestep: int = 4,
        alpha=0.9,
        t_prime_ratio: int = 0.8,
        skip_T: int = 1,
        drag_class: Optional[Literal['copy', 'cut']] = 'copy',
        padding_mask_crop: Optional[int] = None,
        strength: float = 1.0,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
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
            main_image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image`, numpy array or tensor representing an image batch to be used as the starting point. For both
                numpy array and pytorch tensor, the expected value range is between `[0, 1]` If it's a tensor or a list
                or tensors, the expected shape should be `(B, C, H, W)` or `(C, H, W)`. If it is a numpy array or a
                list of arrays, the expected shape should be `(B, H, W, C)` or `(H, W, C)` It can also accept image
                latents as `image`, but if passing latents directly it is not encoded again.
            ref_image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image`, numpy array or tensor representing an image batch to be used as the starting point. For both
                numpy array and pytorch tensor, the expected value range is between `[0, 1]` If it's a tensor or a list
                or tensors, the expected shape should be `(B, C, H, W)` or `(C, H, W)`. If it is a numpy array or a
                list of arrays, the expected shape should be `(B, H, W, C)` or `(H, W, C)` It can also accept image
                latents as `image`, but if passing latents directly it is not encoded again.
            ref_segment (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image`, numpy array or tensor representing an image batch to mask `image`. White pixels in the mask
                are repainted while black pixels are preserved. If `ref_segment` is a PIL image, it is converted to a
                single channel (luminance) before use. If it's a numpy array or pytorch tensor, it should contain one
                color channel (L) instead of 3, so the expected shape for pytorch tensor would be `(B, 1, H, W)`, `(B,
                H, W)`, `(1, H, W)`, `(H, W)`. And for numpy array would be for `(B, H, W, 1)`, `(B, H, W)`, `(H, W,
                1)`, or `(H, W)`.
            ref_segment_latent (`torch.Tensor`, `List[torch.Tensor]`):
                `Tensor` representing an image batch to mask `image` generated by VAE. If not provided, the mask
                latents tensor will ge generated by `ref_segment`.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            padding_mask_crop (`int`, *optional*, defaults to `None`):
                The size of margin in the crop to be applied to the image and masking. If `None`, no crop is applied to
                image and ref_segment. If `padding_mask_crop` is not `None`, it will first find a rectangular region
                with the same aspect ration of the image and contains all masked area, and then expand that area based
                on `padding_mask_crop`. The image and ref_segment will then be cropped based on the expanded area before
                resizing to the original image size for inpainting. This is useful when the masked area is small while
                the image is large and contain information irrelevant for inpainting, such as background.
            strength (`float`, *optional*, defaults to 1.0):
                Indicates extent to transform the reference `image`. Must be between 0 and 1. `image` is used as a
                starting point and more noise is added the higher the `strength`. The number of denoising steps depends
                on the amount of noise initially added. When `strength` is 1, added noise is maximum and the denoising
                process runs for the full number of iterations specified in `num_inference_steps`. A value of 1
                essentially ignores `image`.
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
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
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
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.

            fresh_ratio, type=float, default=0.30)
            cache_type, type=str, choices=['random', 'attention','similarity','norm', 'compress'], default='attention')
            ratio_scheduler, type=str, default='ToCa', choices=['linear', 'cosine', 'exp', 'constant','linear-mode','layerwise','ToCa'])
            force_fresh", type=str, choices=['global', 'local'], default='global',
                        help="Force fresh strategy. global: fresh all tokens. local: fresh tokens acheiving fresh step threshold.")
            fresh_threshold: int = 3,
            soft_fresh_weight: float = 0.25,

        Examples:

        Returns:
            [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
            is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
            images.
        """
        main_latents = None
        
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        
        # init neg_prompt
        if neg_prompt is None:
            neg_prompt = "bad quality, worst quality, text, signature, watermark, extra limbs, low quality, extra person,unharmonious"
        
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            strength=strength,
            height=height,
            width=width,
            output_type=output_type,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            padding_mask_crop=padding_mask_crop,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Preprocess mask and image
        if padding_mask_crop is not None:
            # crops_coords = self.mask_processor.get_crop_region(ref_segment, width, height, pad=padding_mask_crop)
            resize_mode = "fill"
        else:
            crops_coords = None
            resize_mode = "default"

        init_image = self.image_processor.preprocess(
            main_image, height=height, width=width, crops_coords=crops_coords, resize_mode=resize_mode
        )
        init_image = init_image.to(dtype=torch.float32)
        

        # 3. Define call parameters
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
        # 4. Encode prompt
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        (
            null_prompt_embeds,
            null_pooled_prompt_embeds,
            null_text_ids,
        ) = self.encode_prompt(
            prompt="",
            prompt_2="",
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = (int(height) // self.vae_scale_factor) * (int(width) // self.vae_scale_factor)
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            mu=mu,
        )
        timesteps, num_inference_steps, sigmas = self.get_timesteps(num_inference_steps, strength, device)

        if num_inference_steps < 1:
            raise ValueError(
                f"After adjusting the num_inference_steps by strength parameter: {strength}, the number of pipeline"
                f"steps is {num_inference_steps} which is < 1 and not appropriate for this pipeline."
            )
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 9. prepare region pts and values/keys
        _height = 2 * (int(height) // self.vae_scale_factor)
        _width = 2 * (int(width) // self.vae_scale_factor)
        
        if src_region is not None:
            # when src_region is provided as mask
            scale_factor=1
            if src_pts is not None and tgt_pts is not None:
                src_region_pts, tgt_region_pts = src_pts, tgt_pts
            else:
                src_region_pts, tgt_region_pts = region_pair_to_pts(src_region, tgt_region,scale_factor)
            src_region_pts=src_region_pts.cpu()
            tgt_region_pts=tgt_region_pts.cpu()
            temp = init_image[:,:,src_region_pts[:,1],src_region_pts[:,0]].clone()
            init_image[:,:,src_region_pts[:,1],src_region_pts[:,0]] = init_image[:,:,tgt_region_pts[:,1],tgt_region_pts[:,0]]
            init_image[:,:,tgt_region_pts[:,1],tgt_region_pts[:,0]] = temp
            tot_pts=np.concatenate([src_region_pts,tgt_region_pts],axis=1)//int(self.vae_scale_factor)
            tot_pts = np.unique(tot_pts, axis=0)
            src_region_pts=tot_pts[:,:2]
            tgt_region_pts=tot_pts[:,2:]
            
            src_region_pts_idx = torch.tensor(src_region_pts[:,1]*_width//2+src_region_pts[:,0],device=device).clamp(0,_height*_width//4-1)
            tgt_region_pts_idx = torch.tensor(tgt_region_pts[:,1]*_width//2+tgt_region_pts[:,0],device=device).clamp(0,_height*_width//4-1)
            src_pts, tgt_pts = process_point_pairs(src_region_pts, tgt_region_pts, height, width)
        else:
            assert src_pts is not None, "src_pts is not provided"
            tot_pts=np.concatenate([src_pts,tgt_pts],axis=1)//self.vae_scale_factor
            tot_pts = np.unique(tot_pts, axis=0)
            src_pts=tot_pts[:,:2]
            tgt_pts=tot_pts[:,2:]
            src_region_pts_idx = torch.tensor(src_pts[:,1]*_width//2+src_pts[:,0],device=device).clamp(0,_height*_width//4-1)
            tgt_region_pts_idx = torch.tensor(tgt_pts[:,1]*_width//2+tgt_pts[:,0],device=device).clamp(0,_height*_width//4-1)        
        
        # 6. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        num_channels_transformer = self.transformer.config.in_channels

        main_latents, main_noise, main_image_latents, main_latent_image_ids = self.prepare_latents(
            image=init_image,
            timestep=latent_timestep,
            batch_size=batch_size * num_images_per_prompt,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=main_latents,
        )
        orig_main_latents=main_image_latents.clone()
        if mask is not None:
            mask=mask.unsqueeze(0).unsqueeze(0)
            mask_id = self._pack_latents(
                mask.repeat(1, num_channels_latents, 1, 1),
                batch_size,
                num_channels_latents,
                _height,
                _width,
            )
            
        # 7. handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(main_latents.shape[0])
        else:
            guidance = None
            
        inv_dic= self.controlled_forward_ode(
            image_latents=main_image_latents, 
            latent_image_ids=main_latent_image_ids, 
            sigmas=sigmas,
            gamma=gamma, 
            null_prompt_embeds=null_prompt_embeds, 
            null_pooled_prompt_embeds=null_pooled_prompt_embeds, 
            null_text_ids=null_text_ids, 
            timesteps=timesteps, 
            guidance=guidance, 
            height=height, 
            width=width,
            joint_attention_kwargs=joint_attention_kwargs,
            skip_T=skip_T,
        )
        inv_latents, inv_latents_trajectory = inv_dic['inv_latents'], inv_dic['inv_latents_trajectory']

        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 8. Compose main_latents 
        _inv_latents=self._unpack_latents(inv_latents, height, width, self.vae_scale_factor)
        gen_latents = blur_source(_inv_latents, alpha, src_pts,tgt_pts,drag_class=drag_class)
        gen_latents = self._pack_latents(gen_latents, batch_size, num_channels_latents, _height, _width)
         
        
        # 11. Denoising loop warming up
        y_0 = main_image_latents.clone()
        current = self.joint_attention_kwargs['current']
        t_prime = int(num_inference_steps * t_prime_ratio)
        sigmas = self.scheduler.sigmas.to(device=orig_main_latents.device, dtype=orig_main_latents.dtype)
        union_idx=torch.cat([src_region_pts_idx,tgt_region_pts_idx],dim=0)
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                current['step'] = i
                t_i = 1 - t / 1000 
                if self.interrupt:
                    continue
                
                timestep = t.expand(gen_latents.shape[0]).to(gen_latents.dtype)
                if num_inference_steps-i-1>t_prime:
                    gen_latents[:,tgt_region_pts_idx,:]=inv_latents_trajectory[-i-1][:,src_region_pts_idx,:]
                    if drag_class == 'cut':
                        gen_latents[:,src_region_pts_idx,:]=inv_latents_trajectory[-i-1][:,tgt_region_pts_idx,:]
                        
                noise_pred = self.transformer(
                    hidden_states=gen_latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=main_latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                if i < len(timesteps):
                    sigma = sigmas[i+1].flatten()
                    init_latents_proper = sigma * main_noise + (1.0 - sigma) * orig_main_latents
                    
                if mask is None:
                    init_latents_proper[:,union_idx,:]=gen_latents[:,union_idx,:]
                else:
                    init_latents_proper=init_latents_proper*(1-mask_id)+gen_latents*(mask_id)
                gen_latents=init_latents_proper
                
                #  rf-inversion applied
                v_t_cond = (y_0 - gen_latents) / (1 - t_i)
                eta_t = eta if start_timestep <= i < stop_timestep else 0.0
                v_hat_t = -noise_pred + eta_t * (v_t_cond + noise_pred)
                
                # compute the previous noisy sample x_t -> x_t-1
                gen_latents = self.scheduler.step(-v_hat_t, t, gen_latents, return_dict=False)[0]
                
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
        
        if output_type == "latent":
            image = gen_latents
        else:       
            gen_latents = self._unpack_latents(gen_latents, height, width, self.vae_scale_factor)
            gen_latents = (gen_latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(gen_latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)