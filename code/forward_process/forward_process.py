"""
Introduction to Diffuision Models: Visual Information Reconstruction in Neural Networks

This module introduces the Forward procress of Diffusion Models (DM). 
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def image_normalize(size:tuple, image_path:str)->np.array:
    """
    Normalize an image by resizing and scaling pixel values to the [0, 1] range.

    Parameters:
    size (tuple): The target size for the image.
    image_path (str): The path to the image.

    Returns:
    np.array: The normalized image as a NumPy array.
    """
    # Load the input image
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = image.resize(size)
    return np.array(image) / 255.0  # Normalize to [0, 1] 


def forward_diffusion_process(image: np.array, steps: int, scheduler: np.array) -> list:
    """
    Perform the forward diffusion process on an image.

    Parameters:
    x (np.array): The input image as a normalized NumPy array.
    T (int): The number of diffusion steps.
    scheduler (np.array): The diffusion scheduler array.

    Returns:
    list: A list of images at each diffusion step.
    """
    image_list = [image]

    for step in range(1, steps + 1):
        epsilon = np.random.normal(0, 1, image.shape)
        noised_image = np.sqrt(1 - scheduler[step - 1]) * image_list[-1] + np.sqrt(scheduler[step - 1]) * epsilon
        image_list.append(noised_image)
    
    return image_list

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
    # Normalize the input image
    x = image_normalize((512, 512), 'img/elephant.jpeg')

    # Run forward diffusion process
    x_list = forward_diffusion_process(x, T, beta)

    # Plot the results
    plot_images(T, x_list, latent=False)