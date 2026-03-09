import os
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
CSV_FILE = 'keypoint_data.csv'  # Input: collected gesture landmarks
LABELS_FILE = 'gesture_labels.csv'  # Contains gesture names
MODEL_SAVE_DIR = 'models'
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'gesture_model_custom.h5')

def load_data_from_csv():
    """Load landmark data from CSV file."""
    if not os.path.exists(CSV_FILE):
        print(f"ERROR: {CSV_FILE} not found. Please run collect_data.py first.")
        return None, None
    
    data = []
    labels = []
    
    with open(CSV_FILE, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        
        for row in reader:
            if len(row) < 43:  # Need at least class_id + 42 features
                continue
            
            class_id = int(row[0])
            features = np.array([float(x) for x in row[1:]], dtype=np.float32)
            
            data.append(features)
            labels.append(class_id)
    
    if not data:
        print("ERROR: No valid data found in CSV file.")
        return None, None
    
    print(f"Loaded {len(data)} samples from {CSV_FILE}")
    return np.array(data), np.array(labels)

def load_gesture_labels():
    """Load gesture label names from CSV."""
    labels = []
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, 'r') as f:
            labels = [line.strip() for line in f.readlines()]
    return labels

# Load data
print("Loading data from CSV...")
X, y = load_data_from_csv()

if X is None or y is None:
    exit(1)

# Normalize features (important for neural networks)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Number of features: {X.shape[1]}")
print(f"Number of gestures: {len(np.unique(y))}")

# Build Neural Network Model
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(len(np.unique(y)), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n--- Training Model ---")
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")

# Save the model
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

model.save(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")

# Also save the scaler for inference
import pickle
scaler_path = os.path.join(MODEL_SAVE_DIR, 'scaler.pkl')
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"Scaler saved to {scaler_path}")
