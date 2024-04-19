# Copyright 2024 Stanford University Team and The HuggingFace Team. All rights reserved.
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

# DISCLAIMER: This code is strongly influenced by https://github.com/pesser/pytorch_diffusion
# and https://github.com/hojonathanho/diffusion

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import PIL.Image
from PIL import Image
import torch
from dataclasses import dataclass
import numpy as np
from tqdm.auto import tqdm

from ..utils.output import BaseOutput, randn_tensor, BaseModel
from dataclasses import dataclass
from .unet_2d import UNet2DModel
from .vq_model import VQModel, VAEDecoder
from .scheduling_lcm import LCMScheduler

@dataclass
class ImagePipelineOutput(BaseOutput):
    """
    Output class for image pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def preprocess(image):
    w, h = image.size
    #w, h = (x - x % 32 for x in (w, h))  # resize to integer multiple of 32
    #image = image.resize((w, h), resample=PIL_INTERPOLATION["lanczos"])
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0



# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    """
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
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
                must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
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
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from diffusers import AutoPipelineForImage2Image
        >>> import torch
        >>> import PIL

        >>> pipe = AutoPipelineForImage2Image.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")
        >>> # To save GPU memory, torch.float16 can be used, but it may compromise image quality.
        >>> pipe.to(torch_device="cuda", torch_dtype=torch.float32)

        >>> prompt = "High altitude snowy mountains"
        >>> image = PIL.Image.open("./snowy_mountains.png")

        >>> # Can be set to 1~50 steps. LCM support fast inference even <= 4 steps. Recommend: 1~8 steps.
        >>> num_inference_steps = 4
        >>> images = pipe(
        ...     prompt=prompt, image=image, num_inference_steps=num_inference_steps, guidance_scale=8.0
        ... ).images

        >>> images[0].save("image.png")
        ```

"""


class LDMConsistencySRPipeline(BaseModel):
    r"""
    Pipeline for image-to-image generation using a latent consistency model.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).


    Args:
        vae ([`VQModel`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
       
        unet ([`UNet2DModel`]):
            A `UNet2DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Currently only
            supports [`LCMScheduler`].
       
    """

    def __init__(
        self
    ):
        super().__init__()

        #self.vqvae = VQModel(in_channels=3,out_channels=3,down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'], block_out_channels=[128, 256, 512], layers_per_block=2, act_fn='silu', latent_channels=3, sample_size=256,num_vq_embeddings=8192, norm_num_groups=32, vq_embed_dim =3,scaling_factor=0.18215,norm_type='group')
        self.vaedecoder = VAEDecoder(in_channels=3,out_channels=3,up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'], block_out_channels=[128, 256, 512], layers_per_block=2, act_fn='silu', latent_channels=3, sample_size=256,num_vq_embeddings=8192, norm_num_groups=32, vq_embed_dim =3,scaling_factor=0.18215,norm_type='group')
        self.unet = UNet2DModel(sample_size=64, in_channels=6,out_channels=3, center_input_sample=False,time_embedding_type='positional',freq_shift=0,flip_sin_to_cos=True, down_block_types=['DownBlock2D', 'DownBlock2D', 'DownBlock2D', 'AttnDownBlock2D'],up_block_types=['AttnUpBlock2D', 'UpBlock2D', 'UpBlock2D', 'UpBlock2D'], block_out_channels=[160, 320, 320, 640],layers_per_block=2,mid_block_scale_factor=1,downsample_padding=1,downsample_type='conv',upsample_type='conv',dropout=0.0,act_fn='silu', attention_head_dim=32,norm_num_groups=32,attn_norm_num_groups=None,norm_eps=1e-05,resnet_time_scale_shift='default',add_attention=True,class_embed_type=None,num_class_embeds=None,num_train_timesteps=None)
        self.scheduler = LCMScheduler(num_train_timesteps = 1000,
                                      beta_start = 0.0015,
                                      beta_end = 0.0155,
                                      beta_schedule = "scaled_linear",
                                      trained_betas = None,
                                      original_inference_steps = 50,
                                      clip_sample = False,
                                      clip_sample_range = 1.0,
                                      set_alpha_to_one = True,
                                      steps_offset = 1,
                         )
                         
                         
    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        if iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")                     
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.get_timesteps
    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)

        return timesteps, num_inference_steps - t_start

   
       
   
    @property
    def num_timesteps(self):
        return self._num_timesteps

    @torch.no_grad()
    def __call__(
        self,
        image: Union[torch.Tensor, PIL.Image.Image] = None,
        num_inference_steps: int = 4,
        strength: float = 0.8,
        original_inference_steps: int = None,
        timesteps: List[int] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            original_inference_steps (`int`, *optional*):
                The original number of inference steps use to generate a linearly-spaced timestep schedule, from which
                we will draw `num_inference_steps` evenly spaced timesteps from as our final timestep schedule,
                following the Skipping-Step method in the paper (see Section 4.3). If not set this will default to the
                scheduler's `original_inference_steps` attribute.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process. If not defined, equal spaced `num_inference_steps`
                timesteps on the original LCM training/distillation timestep schedule are used. Must be in descending
                order.
            
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
        
        #

        
        device = self.unet.device
        
        if isinstance(image, PIL.Image.Image):
            image = preprocess(image)
        
        
        latents_dtype = next(self.unet.parameters()).dtype
        image = image.to(latents_dtype)
        
        # 6. Prepare latent variables
        original_inference_steps = (
            original_inference_steps
            if original_inference_steps is not None
            else self.scheduler.config.original_inference_steps
        )
        
        # 5. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            original_inference_steps=original_inference_steps,
            strength=strength,
        )

        
        
        latents = randn_tensor(image.shape, generator=generator, device=self.device, dtype=latents_dtype)

         # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma    
        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, None)


        # 8. LCM Multistep Sampling Loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as prog_bar:
            for i, t in enumerate(timesteps):
                
                latents_input = torch.cat([latents, image], dim=1)
                # model prediction (v-prediction, eps, x)
                model_pred = self.unet(latents_input,t).sample

                # compute the previous noisy sample x_t -> x_t-1
                latents, denoised = self.scheduler.step(model_pred, t, latents, **extra_step_kwargs, return_dict=False)
                

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    prog_bar.update()
        

        denoised = denoised.to(latents_dtype)

        # decode the image latents with the VQVAE
        image = self.vaedecoder.decode(denoised).sample
        image = torch.clamp(image, -1.0, 1.0)
        image = image / 2 + 0.5
        image = image.cpu().permute(0, 2, 3, 1).detach().numpy()


        if output_type == "pil":
            image = numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)

      
        
