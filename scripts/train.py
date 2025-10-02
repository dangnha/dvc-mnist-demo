import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# Load data
print("Loading data...")
x_train = np.load('data/raw/x_train.npy')
y_train = np.load('data/raw/y_train.npy')
x_test = np.load('data/raw/x_test.npy')
y_test = np.load('data/raw/y_test.npy')

# Flatten and normalize
x_train_flat = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test_flat = x_test.reshape(x_test.shape[0], -1) / 255.0

print(f"Training on {len(x_train)} samples...")

# Simple model
model = RandomForestClassifier(n_estimators=10, max_depth=10, random_state=42)
model.fit(x_train_flat, y_train)

# Evaluate
y_pred = model.predict(x_test_flat)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")

# Save model and metrics
os.makedirs('models', exist_ok=True)
np.save('models/model.npy', model)

metrics = {
    'accuracy': float(accuracy),
    'dataset_size': len(x_train),
    'dataset_version': 'unknown'
}
with open('models/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("Model and metrics saved!")