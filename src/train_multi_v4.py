
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path
import json

import numpy as np
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Resizing, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------------
# 1. CONFIGURATION
# -----------------------------------

SEED = 42
tf.keras.utils.set_random_seed(SEED)

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results/metrics")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------
# 2. LOAD DATA
# -----------------------------------

X_train = np.load(DATA_DIR / "X_train.npy")
y_train = np.load(DATA_DIR / "y_train.npy")

X_val = np.load(DATA_DIR / "X_val.npy")
y_val = np.load(DATA_DIR / "y_val.npy")

print("Training images:", X_train.shape)
print("Validation images:", X_val.shape)

# -----------------------------------
# 3. BUILD MULTILAYER ANN V4
# -----------------------------------

model = Sequential([
    Input(shape=(64, 64, 3)),

    # Resize from 64x64 to 32x32
    Resizing(32, 32),

    Flatten(),

    Dense(128, activation="relu"),

    Dense(64, activation="relu"),

    Dense(15, activation="softmax")
])

# -----------------------------------
# 4. COMPILE MODEL
# -----------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------------
# 5. TRAIN MODEL
# -----------------------------------

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# -----------------------------------
# 6. SAVE MODEL AND HISTORY
# -----------------------------------

model.save(MODEL_DIR / "multilayer_ann_v4.keras")

history_data = {
    key: [float(value) for value in values]
    for key, values in history.history.items()
}

with open(
    RESULTS_DIR / "multilayer_history_v4.json",
    "w"
) as file:
    json.dump(history_data, file, indent=4)

# -----------------------------------
# 7. VALIDATION EVALUATION
# -----------------------------------

val_loss, val_accuracy = model.evaluate(
    X_val, y_val, verbose=0
)

print("\n================================")
print("MULTILAYER ANN V4 RESULTS")
print("================================")

print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation Loss: {val_loss:.4f}")

print("\nModel saved: models/multilayer_ann_v4.keras")
print("Training history saved successfully!")
