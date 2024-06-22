import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def load_image(image_path, size=(512, 512), to_array=True):
    """Load and optionally convert an image to a numpy array."""
    image = Image.open(image_path).convert('L')
    image = image.resize(size)
    if to_array:
        return np.array(image) / 255.0
    else:
        return image

def save_image(image_array, filename):
    """Save a numpy array as an image."""
    if len(image_array.shape) == 3 and image_array.shape[2] == 1:  # if single channel
        image_array = image_array.squeeze()
    plt.imsave(filename, image_array, cmap='gray')

def image_normalize(image, size=(512, 512)):
    """Normalize image loaded from a path."""
    image = load_image(image, size=size, to_array=False)
    return np.array(image) / 255.0
