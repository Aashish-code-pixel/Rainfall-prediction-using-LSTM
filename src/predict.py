import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load Model
model = tf.keras.models.load_model("models/lstm_model.h5")

# Load and scale data
X_test = np.load("data/X.npy")[-30:]  # Use last 30 days
y_test = np.load("data/y.npy")[-30:]

# Make predictions
predictions = model.predict(X_test)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(y_test, label="Actual")
plt.plot(predictions, label="Predicted")
plt.legend()
plt.show()
