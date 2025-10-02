import numpy as np

print("Creating version 2 (small dataset)...")
# Load version 1
x_train_v1 = np.load('data/raw/x_train_v1.npy')
y_train_v1 = np.load('data/raw/y_train_v1.npy')

# Take only 1000 samples
np.random.seed(42)
indices = np.random.choice(len(x_train_v1), 1000, replace=False)
x_train_v2 = x_train_v1[indices]
y_train_v2 = y_train_v1[indices]

np.save('data/raw/x_train_v2.npy', x_train_v2)
np.save('data/raw/y_train_v2.npy', y_train_v2)

print(f"V2 - Training data: {x_train_v2.shape} (1000 samples)")