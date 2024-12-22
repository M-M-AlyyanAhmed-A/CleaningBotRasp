import cv2
import numpy as np

def preprocess_image(image):
    # Resize the image to the required input size
    image_resized = cv2.resize(image, (224, 224))  # Adjust based on your model
    image_normalized = image_resized / 255.0  # Normalize the image
    input_data = np.expand_dims(image_normalized, axis=0).astype(np.float32)
    return input_data
