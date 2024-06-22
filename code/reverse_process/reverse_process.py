import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_image(image_path, size=(512, 512)):
    """
    Load an image, convert it to grayscale, and normalize it.
    """
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = image.resize(size)
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    return image

def save_image(image_array, filename):
    """
    Save an image from a numpy array.
    """
    image_array = np.squeeze(image_array)  # Remove channel dimension if it exists
    plt.imsave(filename, image_array, cmap='gray')

def load_image(image_path, size=(512, 512)):
    """
    Load an image, convert it to grayscale, and normalize it.
    """
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = image.resize(size)
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    return image

def save_image(image_array, filename):
    """
    Save an image from a numpy array.
    """
    image_array = np.squeeze(image_array)  # Remove channel dimension if it exists
    plt.imsave(filename, image_array, cmap='gray')
def unet_model(input_size=(512, 512, 1)):
    inputs = tf.keras.Input(input_size)

    # Contracting Path
    c1 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = layers.Dropout(0.1)(c1)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = layers.Dropout(0.1)(c2)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = layers.Dropout(0.2)(c3)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # Bottleneck
    c5 = layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c5 = layers.Dropout(0.3)(c5)
    c5 = layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    # Expansive Path
    u6 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c3])
    c6 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c2])
    c7 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c1], axis=3)
    c8 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c8)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

def process_images(model, start=1, end=10):
    """
    Process images from 'img/noised_1.png' to 'img/noised_10.png'.
    """
    for i in range(start, end + 1):
        image_path = f"img/noised_{i}.png"
        image = load_image(image_path)
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        predicted_image = model.predict(image)[0]  # Predict and extract single image from batch
        save_image(predicted_image, f"img/reconstructed_{i}.png")

# Create and compile the U-Net model
model = unet_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Assuming the model is already trained, we process the images
process_images(model)