import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from model import unet_model
from cc import load_data

# Force TensorFlow to use only the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disables GPU usage

def train():
    # Load data
    train_images, train_masks = load_data('data/membrane/train/image', 'data/membrane/train/label')

    # Initialize U-Net model
    model = unet_model()
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    # Train model
    model.fit(train_images, train_masks, batch_size=2, epochs=50, validation_split=0.1)

    # Save the model
    model.save('unet_model.h5')

if __name__ == '__main__':
    train()
