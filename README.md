# Super Resolution with Latent Diffusion Model
Repository for Super Resolution Using Latent Diffusion Model
## Introduction
Neural compression is the application of neural networks and other machine learning methods to data compression. Recent advances in statistical machine learning have opened up
new possibilities for data compression, allowing compression algorithms to be learned end-to-end from data using powerful generative models such as variational
autoencoders, diffusion probabilistic models, and generative adversarial networks.
The diffusion models(DMs) achieve state-of-the-art synthesis results by decomposing the image formation process into a sequential application of denoising autoencoders. The formulation of DMs also allows a guiding mechanism to control the image generation process without retraining. By incorporating cross-attention layers into the model architecture, the guiding mechanism can be conditioned on various inputs. When DMs operate directly in pixel space, the optimization of powerful DMs often consumes hundreds of GPU days and inference is expensive due to sequential evaluations. To enable DM training on limited computational resources while retaining their quality and flexibility, the diffusion can be applied in the latent space of powerful pretrained autoencoders.

a learned prior model for entropy-coding the latent representations into short bit strings. Using either Gaussian or Laplacian decoders, these models directly optimize for low MSE/MAE distortion performance. Given the increasing focus on perceptual performance over distortion, and the fact that VAEs suffer from mode averaging behavior inducing blurriness (Zhao et al., 2017) suggest expected performance gains when replacing the Gaussian decoder with a more expressive conditional generative model.
This paper proposes to relax the typical requirement of Gaussian (or Laplacian) decoders in compression setups and presents a more expressive generative model instead: a conditional diffusion model. Diffusion models have achieved remarkable results on high-quality image generation tasks (Ho et al., 2020; Song et al., 2021b,a). By hybridizing hierarchical compressive VAEs (Ballé et al., 2018) with conditional diffusion models, we create a novel deep generative model with promising properties for
perceptual image compression. This approach is related to but distinct from the recently proposed Diff-AEs (Preechakul et al., 2022), which are neither variational (as needed for entropy coding) nor tailored to the demands of image compression.

### Lossy Compressor
It was shown in [ref1] that DMs have an inductive bias that makes them excellent lossy compressors.  The rate-distortion plot below shows the semantic and perceptual compression with DMs. The distortion is measured as RMSE(root mean square error in pixel space) and rate is in bits/dimensions (log likelyhood/image size). The VQ-VAE is used to achieve first stage of compression (perceptual) by encoding from pixel space to the latent space. Then semantic compression is acheived by the the diffusion process on the latent space.  This two step process is euqivalent to learning respresentations using VAE and then learning probability distribution in the second phase.

Data (in our context, image) compression and generative modeling are two fundamentally related tasks. Intuitively, the essence of compression is to find all “patterns” in the data and assign fewer bits to more frequent patterns. To know exactly how frequent each pattern occurs, one would
need a good probabilistic model of the data distribution, which coincides with the objective of (likelihood-based) generative modeling. This connection between compression and generative modeling has been well established, both theoretically and experimentally, for the lossless setting. In fact, many modern image generative models are also best-performing lossless image compressors [43, 56]. In particular, a popular class of image generative models, variational autoencoders (VAEs) [19], has been proved to have a rate distortion (R-D) theory interpretation [2,54]. With a distortion metric specified, VAEs learn to “compress” data by minimizing a tight upper bound on their information R-D function [54], showing great potential for application to lossy image compression

### Advantages of LDM over GAN 

*  LDMs are trained using Maximum Likelihood cost function (Stable and asymptotically optimal with large dataset)
*  Strong foundation in neurophysics as Theory of Active Inference (Learning is inference in the latent domain and Baysean Brain Hypothesis)
*  Better image quality (IS and FID) than GANs due to inherent inductive bias of learning image represenations using Covnets

LDM also supports other modality input y to condition the denoising process. This can be implemented with a conditional denoising autoencoder θ(zt,t,y)
and paves the way to controlling the synthesis process through inputs y. The low resolution images are used as a condirional input for denoising.

