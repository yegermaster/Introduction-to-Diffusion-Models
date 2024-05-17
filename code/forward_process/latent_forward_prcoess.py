"""
Introduction to Diffusion Models: Visual Information Reconstruction in Neural Networks

This module introduces the Forward process of Latent Diffusion Models (LDM).
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from forward_process import image_normalize, forward_diffusion_process, plot_images

def latent_representation(image: np.array, latent_dim: int) -> np.array:
    """
    Convert an image to its latent space representation by reducing dimensionality.

    Parameters:
    image (np.array): The input image as a NumPy array.
    latent_dim (int): The target dimension in the latent space.

    Returns:
    np.array: The latent space representation of the image.
    """
    flat_image = image.flatten()
    u, s, vh = np.linalg.svd(flat_image.reshape(-1, 1), full_matrices=False)
    latent = np.dot(u[:, :latent_dim], np.diag(s[:latent_dim]))
    return latent

if __name__ == "__main__":
    T = 10
    beta = np.linspace(0.01, 0.2, T)
    # Normalize the input image
    x = image_normalize((512, 512), 'img/elephant.jpeg')

    # Convert the image to its latent representation
    latent_dim = 100  # Example latent dimension
    latent_x = latent_representation(x, latent_dim)

    # Run forward diffusion process on the latent representation
    latent_list = forward_diffusion_process(latent_x, T, beta)

    # Plot the results
    plot_images(T, latent_list, latent=True)
