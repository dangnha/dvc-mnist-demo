import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
import argparse

def train_model(version):
    print(f"Training model for version: {version}")
    
    # Load specific version data
    x_train = np.load(f'data/raw/x_train_{version}.npy')
    y_train = np.load(f'data/raw/y_train_{version}.npy')
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

    # Save model and metrics with version
    os.makedirs('models', exist_ok=True)
    np.save(f'models/model_{version}.npy', model)

    metrics = {
        'accuracy': float(accuracy),
        'dataset_size': len(x_train),
        'dataset_version': version,
        'model': f'model_{version}'
    }
    
    with open(f'models/metrics_{version}.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Model and metrics saved for {version}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', required=True, choices=['v1', 'v2', 'v3'])
    args = parser.parse_args()
    
    train_model(args.version)