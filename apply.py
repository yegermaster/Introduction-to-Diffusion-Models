import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import normalize
from tensorflow.keras.optimizers import Adam
from unet import unet_model

def load_data(image_dir, mask_dir, target_size=(128, 128)):
    """Loads input images and corresponding masks."""
    images = []
    masks = []
    for image_name in os.listdir(image_dir):
        # Load and preprocess input image
        img_path = os.path.join(image_dir, image_name)
        img = img_to_array(load_img(img_path, target_size=target_size)) / 255.0
        images.append(img)

        # Load and preprocess corresponding mask
        mask_name = f"no_bg_{image_name}"  # Match mask filename
        mask_path = os.path.join(mask_dir, mask_name)
        mask = img_to_array(load_img(mask_path, target_size=target_size, color_mode="grayscale")) / 255.0
        masks.append(mask)

    return np.array(images), np.array(masks)

def load_test_data(image_dir, target_size=(128, 128)):
    """Loads test images with background only."""
    images = []
    for image_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, image_name)
        img = img_to_array(load_img(img_path, target_size=target_size)) / 255.0
        images.append(img)

    return np.array(images)

# Directories
train_image_dir = "dataset/train/with_bg/"
train_mask_dir = "dataset/train/no_bg/"
test_image_dir = "dataset/test/with_bg/"

# Load training data
X_train, y_train = load_data(train_image_dir, train_mask_dir)

# Load test data
X_test = load_test_data(test_image_dir)

model = unet_model(input_size=(128, 128, 3), n_filters=32, n_classes=1)
# Compile U-Net
model.compile(optimizer=Adam(learning_rate=1e-3), loss="binary_crossentropy", metrics=["accuracy"])
# Train the U-Net
history = model.fit(X_train, y_train, epochs=10, batch_size=4, validation_split=0.2)


# Predict masks for test images
y_pred = model.predict(X_test)

# Binarize predictions for visualization (threshold = 0.5)
y_pred_binary = (y_pred > 0.5).astype(np.uint8)


# Save predicted masks to a directory
output_dir = "predicted_masks/"
os.makedirs(output_dir, exist_ok=True)

for i, pred_mask in enumerate(y_pred_binary):
    pred_path = os.path.join(output_dir, f"pred_mask_{i + 1}.png")
    plt.imsave(pred_path, pred_mask[:, :, 0], cmap="gray")



# Visualize predictions
for i in range(5):
    plt.figure(figsize=(12, 4))

    # Input image with background
    plt.subplot(1, 2, 1)
    plt.imshow(X_test[i])
    plt.title("Input Image (With BG)")
    plt.axis("off")

    # Predicted mask
    plt.subplot(1, 2, 2)
    plt.imshow(y_pred_binary[i, :, :, 0], cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()