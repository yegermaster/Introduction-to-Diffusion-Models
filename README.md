# Introduction to Diffusion Models

## Introduction

This repository explores Latent Diffusion Models (LDMs) for reconstructing high-resolution images, emphasizing the U-Net architecture. We discuss the general mechanism of LDMs, their integration in Code, and specific use cases in visual information processing. This discussion is contextualized within recent research using LDM to reproduce visual perceptions based on brain activity data.

<div style="display: flex; justify-content: space-around;">
    <img src="img/dall-e1.png" alt="Description of image 1" width="150"/>
    <img src="img/dall-e2.png" alt="Description of image 2" width="150"/>
</div>

## Stable diffuision
Stable diffusion operates like a sophisticated mathematical game of 'telephone'. It uses a stochastic process, where each state is solely dependent on the previous one, forming a Markov chain. In this model, information transitions from one distribution to another, akin to processes seen in non-equilibrium thermodynamics. This is similar to how ink disperses in water, starting from a localized concentration and gradually diffusing to form a more uniform distribution.

Just like ink dispersing in water, we begin by embedding the data in noise (the forward process, akin to an encoder) and then systematically recover the original information (the reverse process, similar to a decoder).



<img src="img/ink_in_water.webp" alt="Description of image 1" width="200"/>


So, just like our ink in water, we first immerse the data into noise (the forward process or encoder) and then trace it back to the original information (the reverse process or decoder).

## Forward process
During this process, Gaussian noise is progressively added to the data X over T steps, creating a sequence of increasingly noisy data representations {Xt}. This simulates a forward diffusion process where the data becomes more corrupted at each step. The result for each individual data point X can be seen in the image before where T=10 (normally T~=1000) However, this type of algorithm might struggle with high computational costs and inefficiencies when processing high-dimensional data , as these costs scale significantly with the data's dimensionality. A solution to this can be found in Latent Diffusion Models (LDM).

Check out the `forward_process.py` file for an example of the code in action.

![alt text](img/forward_process.png)

Latent Diffusion Models (LDM) involve encoding an image into a lower-dimensional latent space, before the diffusion process, applying noise to the latent representation and thereby reducing the computational load while maintaining the integrity of data reconstruction.

Check out the `latent_forward_process.py` file for an example of the code in action.

![alt text](img/latent_forward_process.png)

* Step 0: The latent representation of the original image, which retains most of the information from the original image but in a compressed form.
* Step 1 to Step 10: Progressive addition of noise to the latent representation. As you move from Step 1 to Step 10, the latent representation becomes increasingly noisy, leading to a loss of the original information. By Step 10, the representation appears nearly homogeneous, indicating significant information loss.

## Reverse Process
The reverse process, or decoding process, is the core mechanism of the Diffusion Model (LDM). This process involves iteratively removing the noise added during the forward process, effectively reversing the diffusion steps to recover the original data. Each step utilizes a neural network to predict and subtract the noise, gradually converging on the original data distribution.

U-Nets play a crucial role in this process. U-Nets have a U-shaped architecture that is symmetrical along the encoding and decoding paths. This allows the network to capture both local patterns, such as edges and textures, and more abstract features at deeper layers.

This methodology begins with the final noisy representation `Z_T` and employs a neural network model to predict the noise that must be subtracted to achieve `Z_{T-1}`. This process is continued iteratively from `Z_{T-1}` to `Z_{T-2}`, and so forth, until `Z_0` is obtained. Once `Z_0` is reached, it is decoded back into a higher-dimensional space than the latent representation, resulting in `X`, our reconstructed data.


## Implementation of U-Net in LDM

U-Net is central to stable diffusion:

1. Contracting Path: Learns low-level to high-level features (edges, shapes, textures).

2. Bottleneck: Holds a compressed, high-level representation.

3. Expansive Path: Gradually upscales features to reconstruct fine details.

In LDMs, U-Net receives:

1. Noisy latent image at each step.

2. Time embedding indicating the diffusion step (noise level).

3.Optionally, text embeddings (e.g., from CLIP) if conditioning on textual descriptions.

This architecture effectively balances broad context and precise details, mirroring how the brain processes visual information.

