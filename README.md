# Super Resolution with Latent Diffusion Model
Repository for Super Resolution Using Latent Diffusion Model
## Introduction
The diffusion models(DMs) achieve state-of-the-art synthesis results by decomposing the image formation process into a sequential application of denoising autoencoders. The formulation of DMs also allows a guiding mechanism to control the image generation process without retraining. By incorporating cross-attention layers into the model architecture, the guiding mechanism can be conditioned on various inputs. When DMs operate directly in pixel space, the optimization of powerful DMs often consumes hundreds of GPU days and inference is expensive due to sequential evaluations. To enable DM training on limited computational resources while retaining their quality and flexibility, we apply them in the latent space of powerful pretrained autoencoders.

In contrast to previous work, training diffusion models on such a representation allows for the first time to reach a near-optimal point between complexity reduction
and detail preservation, greatly boosting visual fidelity. By introducing cross-attention layers into the model architecture, we turn diffusion models into powerful and flexible
generators for general conditioning inputs such as text or low resolution images
