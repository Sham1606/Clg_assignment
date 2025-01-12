import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib  # Import joblib for saving the model

# Load the trained classification model
classification_model = tf.keras.models.load_model('D:/My_Data_Science/Clg_assignment/classification_model_small.h5')

# Create a new model that outputs the features from the second-to-last layer
feature_extractor = Model(inputs=classification_model.input,
                          outputs=classification_model.layers[-2].output)

# Generate features for all images
datagen = ImageDataGenerator(rescale=1./255)
generator = datagen.flow_from_directory(
    'D:/My_Data_Science/Clg_assignment/chest_xray/chest_xray/train',
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary',
    shuffle=True
)

features = feature_extractor.predict(generator)

# Generate synthetic target values (e.g., lung capacity)
np.random.seed(42)
target = np.random.normal(loc=70, scale=10, size=len(features))

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Create and train model
model = SVR(kernel='linear')
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared: {r2:.4f}")

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Lung Capacity')
plt.ylabel('Predicted Lung Capacity')
plt.title('Actual vs Predicted Lung Capacity')
plt.show()

# Save the regression model using joblib
joblib.dump(model, 'D:/My_Data_Science/Clg_assignment/regression_model2.joblib')
print("Regression model saved successfully.")
