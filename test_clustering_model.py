import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib
import os

# Load the trained classification model (for feature extraction)
classification_model = load_model('classification_model_updated.h5')

# Create a feature extractor model
feature_extractor = tf.keras.Model(inputs=classification_model.input,
                                   outputs=classification_model.layers[-2].output)

# Load the trained KMeans model
kmeans_model = joblib.load('D:/My_Data_Science/Clg_assignment/kmeans_model.joblib')


def preprocess_image(img_path):
    img = load_img(img_path, target_size=(150, 150))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


def extract_features(img_array):
    features = feature_extractor.predict(img_array)
    return features.flatten()


def predict_cluster(img_path):
    img_array = preprocess_image(img_path)
    features = extract_features(img_array)
    cluster = kmeans_model.predict([features])[0]
    return cluster


def visualize_clusters(features, labels):
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.title('Cluster Visualization')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.show()


def test_single_image(img_path):
    if not os.path.exists(img_path):
        print(f"Error: The file {img_path} does not exist.")
        return

    cluster = predict_cluster(img_path)
    print(f"Image: {img_path}")
    print(f"Assigned Cluster: {cluster}")

    # Display the image
    img = load_img(img_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Assigned Cluster: {cluster}")
    plt.show()


def test_directory(directory_path):
    if not os.path.exists(directory_path):
        print(f"Error: The directory {directory_path} does not exist.")
        return

    features_list = []
    file_paths = []

    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(directory_path, filename)
            img_array = preprocess_image(img_path)
            features = extract_features(img_array)
            features_list.append(features)
            file_paths.append(img_path)

    features_array = np.array(features_list)
    clusters = kmeans_model.predict(features_array)

    for file_path, cluster in zip(file_paths, clusters):
        print(f"Image: {file_path}")
        print(f"Assigned Cluster: {cluster}")
        print()

    visualize_clusters(features_array, clusters)


if __name__ == "__main__":
    # Test a single image
    test_single_image("D:/My_Data_Science/Clg_assignment/test2.png")

    # Test all images in a directory
    # test_directory("path/to/your/test_images_directory")

