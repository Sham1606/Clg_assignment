import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the trained model
model = load_model('classification_model_small.h5')

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array

def predict_image(img_path):
    preprocessed_img = preprocess_image(img_path)
    prediction = model.predict(preprocessed_img)
    return prediction[0][0]

def display_result(img_path, prediction):
    img = image.load_img(img_path)
    plt.imshow(img)
    plt.axis('off')
    result = "Pneumonia" if prediction > 0.5 else "Normal"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    plt.title(f"Prediction: {result}\nConfidence: {confidence:.2f}")
    plt.show()

# Test the model on a single image
def test_single_image(img_path):
    if not os.path.exists(img_path):
        print(f"Error: The file {img_path} does not exist.")
        return

    prediction = predict_image(img_path)
    print(f"Prediction value: {prediction}")
    print(f"Diagnosis: {'Pneumonia' if prediction > 0.5 else 'Normal'}")
    display_result(img_path, prediction)

# Test the model on all images in a directory
def test_directory(directory_path):
    if not os.path.exists(directory_path):
        print(f"Error: The directory {directory_path} does not exist.")
        return

    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(directory_path, filename)
            print(f"\nTesting image: {filename}")
            test_single_image(img_path)

# Example usage
if __name__ == "__main__":
    # Test a single image
    test_single_image("D:/My_Data_Science/Clg_assignment/test3.webp")

    # Test all images in a directory
    # test_directory("path/to/your/test_images_directory")

