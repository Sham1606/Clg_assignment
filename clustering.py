import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

# Set the path to the classification model
model_path = 'D:/My_Data_Science/Clg_assignment/classification_model_small.h5'

# Check if the model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"The model file '{model_path}' does not exist. Please check the file path.")

# Load the trained classification model
classification_model = load_model(model_path)

# Create a new model that outputs the features from the second-to-last layer
feature_extractor = tf.keras.Model(inputs=classification_model.input,
                                   outputs=classification_model.layers[-2].output)

# Set up data path
data_dir = 'D:/My_Data_Science/Clg_assignment/chest_xray/chest_xray'
train_dir = os.path.join(data_dir, 'train')

# Generate features for all images
datagen = ImageDataGenerator(rescale=1./255)
generator = datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary',
    shuffle=True
)

features = feature_extractor.predict(generator)

# Determine optimal number of clusters using the elbow method
wcss = []
silhouette_scores = []
max_clusters = 10

for i in range(2, max_clusters + 1):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(features)
    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(features, kmeans.labels_))

# Plot elbow curve
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(2, max_clusters + 1), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')

# Plot silhouette scores
plt.subplot(1, 2, 2)
plt.plot(range(2, max_clusters + 1), silhouette_scores)
plt.title('Silhouette Score')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.tight_layout()
plt.show()

# Choose the optimal number of clusters (e.g., 3)
optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(features)

# Evaluate
silhouette_avg = silhouette_score(features, cluster_labels)
print(f"Silhouette Score: {silhouette_avg:.4f}")

# Visualize clusters using PCA
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(features)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=cluster_labels, cmap='viridis')
plt.title(f'Clusters Visualization (n_clusters = {optimal_clusters})')
plt.colorbar(scatter)
plt.show()

# Analyze cluster centers
cluster_centers = kmeans.cluster_centers_
for i, center in enumerate(cluster_centers):
    print(f"Cluster {i} center:")
    print(center[:10])  # Print first 10 features of the center
    print()

# Save the classification model
classification_model.save('classification_model_updated.h5')
print("Classification model saved successfully as classification_model_updated.h5")


#Why Use joblib for KMeans?
# Joblib is designed for efficient serialization of large NumPy arrays, which are part of the KMeans model (e.g., cluster centers and labels).
# It’s not suitable for TensorFlow/Keras models because it doesn’t handle the complex graph structure of neural networks.
# Optionally, save the feature extractor model
# Save the KMeans model
joblib.dump(kmeans, 'kmeans_model.joblib')
print("KMeans model saved successfully.")
