# Super Resolution with Latent Diffusion Model
Repository for Super Resolution Using Latent Diffusion Model
## Introduction
Neural compression is the application of neural networks and other machine learning methods to data compression. Recent advances in statistical machine learning have opened up
new possibilities for data compression, allowing compression algorithms to be learned end-to-end from data using powerful generative models such as variational
autoencoders, diffusion probabilistic models, and generative adversarial networks.
The diffusion models(DMs) achieve state-of-the-art synthesis results by decomposing the image formation process into a sequential application of denoising autoencoders. The formulation of DMs also allows a guiding mechanism to control the image generation process without retraining. By incorporating cross-attention layers into the model architecture, the guiding mechanism can be conditioned on various inputs. When DMs operate directly in pixel space, the optimization of powerful DMs often consumes hundreds of GPU days and inference is expensive due to sequential evaluations. To enable DM training on limited computational resources while retaining their quality and flexibility, the diffusion can be applied in the latent space of powerful pretrained autoencoders.

### Lossy Compressor
It was shown in [ref1] that DMs have an inductive bias that makes them excellent lossy compressors.  The rate-distortion plot below shows the semantic and perceptual compression with DMs. The distortion is measured as RMSE(root mean square error in pixel space) and rate is in bits/dimensions (log likelyhood/image size). The VQ-VAE is used to achieve first stage of compression (perceptual) by encoding from pixel space to the latent space. Then semantic compression is acheived by the the diffusion process on the latent space.  This two step process is euqivalent to learning respresentations using VAE and then learning probability distribution in the second phase.

### Advantages of LDM over GAN 

*  LDMs are trained using Maximum Likelihood cost function (Stable and asymptotically optimal with large dataset)
*  Strong foundation in neurophysics as Theory of Active Inference (Learning is inference in the latent domain and Baysean Brain Hypothesis)
*  Better image quality (IS and FID) than GANs due to inherent inductive bias of learning image represenations using Covnets

LDM also supports other modality input y to condition the denoising process. This can be implemented with a conditional denoising autoencoder θ(zt,t,y)
and paves the way to controlling the synthesis process through inputs y. The low resolution images are used as a condirional input for denoising.

