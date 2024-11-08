import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
import glob


def load_data(image_path, label_path, target_size=(512, 512)):
    # Load and resize images and masks to ensure consistent shape
    images = sorted(glob.glob(os.path.join(image_path, "*.png")))
    labels = sorted(glob.glob(os.path.join(label_path, "*.png")))

    train_images = np.array(
        [img_to_array(load_img(img, color_mode="grayscale", target_size=target_size)) for img in images]
    ) / 255.0
    train_masks = np.array(
        [img_to_array(load_img(mask, color_mode="grayscale", target_size=target_size)) for mask in labels]
    ) / 255.0

    return train_images.astype("float32"), train_masks.astype("float32")


def load_test_data(test_path, target_size=(512, 512)):
    # Load and resize test images
    test_images = sorted(glob.glob(os.path.join(test_path, "*.png")))
    test_images = np.array(
        [img_to_array(load_img(img, color_mode="grayscale", target_size=target_size)) for img in test_images]
    ) / 255.0

    return test_images.astype("float32")


def save_predictions(predictions, save_path):
    # Ensure the save path directory exists
    os.makedirs(save_path, exist_ok=True)

    for i, pred in enumerate(predictions):
        pred_img = array_to_img(pred)
        pred_img.save(os.path.join(save_path, f"{i}_predict.png"))
