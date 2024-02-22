import inspect
from typing import List, Optional, Tuple, Union
from PIL import Image
import PIL.Image

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

from ..utils.output import BaseOutput, randn_tensor, BaseModel
from dataclasses import dataclass
from .unet_2d import UNet2DModel
from .vq_model import VQModel, VAEDecoder
from .scheduling_ddim import DDIMScheduler



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

def preprocess(image):
    w, h = image.size
    #w, h = (x - x % 32 for x in (w, h))  # resize to integer multiple of 32
    #image = image.resize((w, h), resample=PIL_INTERPOLATION["lanczos"])
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


class LDMSuperResolutionPipeline(BaseModel):
    r"""
    A pipeline for image super-resolution using latent diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        vaedecoder ([`VAEDecoder`]):
            Vector-quantized (VQ) model to encode and decode images to and from latent representations.
        unet ([`UNet2DModel`]):
            A `UNet2DModel` to denoise the encoded image.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latens. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], [`EulerDiscreteScheduler`],
            [`EulerAncestralDiscreteScheduler`], [`DPMSolverMultistepScheduler`], or [`PNDMScheduler`].
    """

    def __init__(self):
        super().__init__()

        self.unet = UNet2DModel(sample_size=64, in_channels=6,out_channels=3, center_input_sample=False,time_embedding_type='positional',freq_shift=0,flip_sin_to_cos=True, down_block_types=['DownBlock2D', 'DownBlock2D', 'DownBlock2D', 'AttnDownBlock2D'],up_block_types=['AttnUpBlock2D', 'UpBlock2D', 'UpBlock2D', 'UpBlock2D'], block_out_channels=[160, 320, 320, 640],layers_per_block=2,mid_block_scale_factor=1,downsample_padding=1,downsample_type='conv',upsample_type='conv',dropout=0.0,act_fn='silu', attention_head_dim=32,norm_num_groups=32,attn_norm_num_groups=None,norm_eps=1e-05,resnet_time_scale_shift='default',add_attention=True,class_embed_type=None,num_class_embeds=None,num_train_timesteps=None)
        self.scheduler = DDIMScheduler(num_train_timesteps = 1000,
                                  beta_start = 0.0015,
                                  beta_end = 0.015,
                                  beta_schedule ="scaled_linear",
                                  trained_betas = None,
                                  clip_sample = False,
                                  set_alpha_to_one = True,
                                  steps_offset = 1,
                                  prediction_type = "epsilon",
                                  thresholding = False,
                                  dynamic_thresholding_ratio = 0.995,
                                  clip_sample_range = 1.0,
                                  sample_max_value = 1.0,
                                  timestep_spacing = "leading",
                                  rescale_betas_zero_snr = False,
                    )
        self.vaedecoder = VAEDecoder(in_channels=3,out_channels=3,up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'], block_out_channels=[128, 256, 512], layers_per_block=2, act_fn='silu', latent_channels=3, sample_size=256,num_vq_embeddings=8192, norm_num_groups=32, vq_embed_dim =3,scaling_factor=0.18215,norm_type='group')
    
    
    @torch.no_grad()
    def __call__(
        self,
        image: Union[torch.Tensor, PIL.Image.Image] = None,
        batch_size: Optional[int] = 1,
        num_inference_steps: Optional[int] = 100,
        eta: Optional[float] = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[Tuple, ImagePipelineOutput]:
        r"""
        The call function to the pipeline for generation.

        Args:
            image (`torch.Tensor` or `PIL.Image.Image`):
                `Image` or tensor representing an image batch to be used as the starting point for the process.
            batch_size (`int`, *optional*, defaults to 1):
                Number of images to generate.
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`ImagePipelineOutput`] instead of a plain tuple.

        Example:

       

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, torch.Tensor):
            batch_size = image.shape[0]
        else:
            raise ValueError(f"`image` has to be of type `PIL.Image.Image` or `torch.Tensor` but is {type(image)}")

        if isinstance(image, PIL.Image.Image):
            image = preprocess(image)

        height, width = image.shape[-2:]

        # in_channels should be 6: 3 for latents, 3 for low resolution image
        latents_shape = (batch_size, self.unet.config.in_channels // 2, height, width)
        latents_dtype = next(self.unet.parameters()).dtype

        latents = randn_tensor(latents_shape, generator=generator, device=self.device, dtype=latents_dtype)

        image = image.to(device=self.device, dtype=latents_dtype)

        # set timesteps and move to the correct device
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps_tensor = self.scheduler.timesteps

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature.
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_kwargs = {}
        if accepts_eta:
            extra_kwargs["eta"] = eta

        for t in tqdm(timesteps_tensor):
            # concat latents and low resolution image in the channel dimension.
            latents_input = torch.cat([latents, image], dim=1)
            latents_input = self.scheduler.scale_model_input(latents_input, t)
            # predict the noise residual
            noise_pred = self.unet(latents_input, t).sample
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_kwargs).prev_sample

        # decode the image latents with the VQVAE
        image = self.vaedecoder.decode(latents).sample
        image = torch.clamp(image, -1.0, 1.0)
        image = image / 2 + 0.5
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        if output_type == "pil":
            image = numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)

        
   

