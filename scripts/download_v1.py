import tensorflow as tf
import numpy as np
import os

print("Downloading MNIST version 1 (full dataset)...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Save to data/raw
np.save('data/raw/x_train_v1.npy', x_train)
np.save('data/raw/y_train_v1.npy', y_train)
np.save('data/raw/x_test.npy', x_test)
np.save('data/raw/y_test.npy', y_test)

print(f"V1 - Training data: {x_train.shape}")
print(f"V1 - Test data: {x_test.shape}")