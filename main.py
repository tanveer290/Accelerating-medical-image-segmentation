import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import array_to_img
def predict():
    # Load trained model
    model = load_model('unet_model.h5')

    # Load test images
    test_images = load_test_data('data/membrane/test')

    # Predict segmentation masks for test images
    predictions = model.predict(test_images, batch_size=2)

    # Directory to save segmented images
    segmented_images_dir = 'data/membrane/test/segmentedimages'
    os.makedirs(segmented_images_dir, exist_ok=True)

    # Save predictions in the segmented images directory
    save_predictions(predictions, segmented_images_dir)

def load_test_data(directory):
        test_images = []
        for filename in os.listdir(directory):
            if filename.endswith('.png'):  # Ensure only image files are processed
                img_path = os.path.join(directory, filename)
                img = load_img(img_path, color_mode="grayscale", target_size=(512, 512))
                img_array = img_to_array(img) / 255.0  # Normalize the image
                test_images.append(img_array)
        return np.array(test_images)

def save_predictions(predictions, output_dir):
        """
        Saves each prediction in the specified output directory.
        Assumes predictions are in a 4D numpy array with shape (num_images, height, width, 1).
        """
        os.makedirs(output_dir, exist_ok=True)

        # Iterate through each prediction and save it as an image
        for i, prediction in enumerate(predictions):
            # Convert prediction to an image (assuming grayscale)
            img = array_to_img(prediction)

            # Save the image
            img.save(os.path.join(output_dir, f"{i}_segmented.png"))


if __name__ == '__main__':
    predict()
