import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

# Load the trained classification model
classification_model = load_model('classification_model_small2.h5')

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
    shuffle=True)

features = feature_extractor.predict(generator)

# Apply PCA
pca = PCA(n_components=0.95)  # Retain 95% of variance
reduced_features = pca.fit_transform(features)

# Evaluate
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

print(f"Number of components: {pca.n_components_}")
print(f"Cumulative explained variance ratio: {cumulative_variance_ratio[-1]:.4f}")

# Plot explained variance ratio
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_variance_ratio, 'bo-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('PCA: Cumulative Explained Variance Ratio')
plt.show()

# Visualize first two principal components
plt.figure(figsize=(10, 8))
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], alpha=0.5)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA: First Two Principal Components')
plt.show()

# Analyze feature importance
feature_importance = np.abs(pca.components_[0])
sorted_idx = np.argsort(feature_importance)
sorted_features = feature_importance[sorted_idx]

plt.figure(figsize=(10, 6))
plt.barh(range(10), sorted_features[-10:])
plt.yticks(range(10), [f"Feature {i}" for i in sorted_idx[-10:]])
plt.xlabel('Absolute Weight')
plt.title('Top 10 Features in First Principal Component')
plt.tight_layout()
plt.show()

