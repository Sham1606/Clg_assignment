import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import joblib
import os

# Load the trained classification model (for feature extraction)
classification_model = load_model('classification_model_small.h5')

# Create a feature extractor model
feature_extractor = tf.keras.Model(inputs=classification_model.input,
                                   outputs=classification_model.layers[-2].output)

# Load the trained regression model
regression_model = joblib.load('regression_model.joblib')

def preprocess_image(img_path):
    img = load_img(img_path, target_size=(150, 150))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def extract_features(img_array):
    features = feature_extractor.predict(img_array)
    return features.flatten()

def predict_lung_capacity(img_path):
    img_array = preprocess_image(img_path)
    features = extract_features(img_array)
    prediction = regression_model.predict([features])[0]
    return prediction

def display_result(img_path, prediction):
    img = load_img(img_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Predicted Lung Capacity: {prediction:.2f}")
    plt.show()

def test_single_image(img_path):
    if not os.path.exists(img_path):
        print(f"Error: The file {img_path} does not exist.")
        return

    prediction = predict_lung_capacity(img_path)
    print(f"Predicted Lung Capacity: {prediction:.2f}")
    display_result(img_path, prediction)

def test_directory(directory_path):
    if not os.path.exists(directory_path):
        print(f"Error: The directory {directory_path} does not exist.")
        return

    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(directory_path, filename)
            print(f"\nTesting image: {filename}")
            test_single_image(img_path)

if __name__ == "__main__":
    # Test a single image
    test_single_image("D:/My_Data_Science/Clg_assignment/test2.png")

    # Test all images in a directory
    # test_directory("path/to/your/test_images_directory")

