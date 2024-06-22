"""
Introduction to Diffuision Models: Visual Information Reconstruction in Neural Networks

This module introduces the Forward procress of Diffusion Models (DM). 
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def image_normalize(size:tuple, image_path:str)->np.array:
    """ Normalize an image by resizing and scaling pixel values to the [0, 1] range."""
    # Load the input image
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = image.resize(size)
    return np.array(image) / 255.0  # Normalize to [0, 1] 



def forward_diffusion_process(image: np.array, steps: int, scheduler: np.array, save_list: list) -> list:
    """Perform the forward diffusion process on an image."""
    image_list = [image]

    for step in range(1, steps + 1):
        epsilon = np.random.normal(0, 1, image.shape)
        noised_image = np.sqrt(1 - scheduler[step - 1]) * image_list[-1] + np.sqrt(scheduler[step - 1]) * epsilon
        image_list.append(noised_image)
        if step in save_list:
            Image.fromarray((noised_image * 255).astype(np.uint8)).save(f"code/code_img/noised_{step}.png")

    
    return image_list

def plot_images(image_list: list, latent: bool, save_list: list) -> plt:
    """Plot images at specified steps from save_list."""
    _, axes = plt.subplots(1, len(save_list), figsize=(15, 5))  
    for i, step in enumerate(save_list):
        noised_image = image_list[step]
        if latent:
            axes[i].imshow(noised_image, aspect='auto', cmap='gray')
        else:
            axes[i].imshow(noised_image, cmap='gray')
        axes[i].set_title(f"Step {step}", fontsize=10)
        axes[i].axis('off')
    plt.show()



if __name__ == "__main__":
    T = 1000
    beta = np.linspace(0.00001, 0.000001326, T)
    s = np.linspace(0, T, T)
    beta = 0.002 * (1 - np.cos(s / T * np.pi))

    save_index = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999]
    # Normalize the input image
    x = image_normalize((512, 512), 'img/elephant.jpeg')

    # Run forward diffusion process
    x_list = forward_diffusion_process(x, T, beta, save_index)

    # Plot the results
    plot_images(x_list, latent=False, save_list=save_index)
