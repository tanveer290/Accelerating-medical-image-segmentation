import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from model import unet_model
from cc import load_data

def train():
    # Initialize a distributed training strategy
    strategy = tf.distribute.MirroredStrategy()

    # Load data
    train_images, train_masks = load_data('data/membrane/train/image', 'data/membrane/train/label')

    # Use the strategy scope to set up the model for parallel training
    with strategy.scope():
        # Initialize U-Net model
        model = unet_model()
        model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

        # Train model
        model.fit(train_images, train_masks, batch_size=2, epochs=50, validation_split=0.1)

        # Save the model
        model.save('unet_model.h5')

if __name__ == '__main__':
    train()
