import sys
from pathlib import Path

# Adjusting the path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from forward_process.forward_process import image_normalize, forward_diffusion_process
from unet import unet_model

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def backward_diffusion_process(noised_images: list, scheduler: np.array, model) -> list:
    """
    Perform the backward diffusion process on a list of noised images using a U-Net model.

    Parameters:
    noised_images (list): The list of images at each diffusion step.
    scheduler (np.array): The diffusion scheduler array.
    model: The U-Net model used for denoising.

    Returns:
    list: A list of images at each backward diffusion step.
    """
    steps = len(noised_images) - 1
    reversed_images = [noised_images[-1]]

    for step in range(steps - 1, -1, -1):
        current_image = reversed_images[-1]
        epsilon = np.random.normal(0, 1, current_image.shape)
        de_noised_image = model.predict((current_image - np.sqrt(scheduler[step]) * epsilon) / np.sqrt(1 - scheduler[step])[np.newaxis, ...])[0]
        reversed_images.append(de_noised_image)
    
    reversed_images.reverse()  # To maintain the chronological order
    return reversed_images

def plot_images(steps: int, image_list: list, latent: bool) -> plt:
    """
    Plot images at each step of the diffusion process.

    Parameters:
    T (int): The number of diffusion steps.
    x_list (list): The list of images at each diffusion step.

    Returns:
    plt: The plot object with the displayed images.
    """
    _, axes = plt.subplots(1, steps + 1, figsize=(15, 5))
    for step, noised_image in enumerate(image_list):
        if latent:
            axes[step].imshow(noised_image, aspect='auto', cmap='gray')
        else:
            axes[step].imshow(noised_image, cmap='gray')
        axes[step].set_title(f"Step {step}", fontsize=10)
        axes[step].axis('off')
    plt.show()

if __name__ == "__main__":
    T = 10
    beta = np.linspace(0.01, 0.2, T)
    input_shape = (512, 512, 1)
    unet = unet_model(input_shape)
    
    # Load pre-trained weights if available
    # unet.load_weights('path_to_weights.h5')

    # Normalize the input image
    x = image_normalize((512, 512), 'img/elephant.jpeg')

    # Run forward diffusion process
    x_list = forward_diffusion_process(x, T, beta)

    # Run backward diffusion process using U-Net
    x_reconstructed_list = backward_diffusion_process(x_list, beta, unet)

    # Plot the results of the backward diffusion process
    plot_images(T, x_reconstructed_list, latent=False)
