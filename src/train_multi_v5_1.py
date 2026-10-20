
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input, Resizing, Flatten, Dense
)
from tensorflow.keras.callbacks import EarlyStopping

SEED = 42
tf.keras.utils.set_random_seed(SEED)

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results/metrics")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------
# 1. LOAD DATA
# -----------------------------------

X_train = np.load(
    DATA_DIR / "X_train_v5.npy",
    mmap_mode="r"
)
y_train = np.load(DATA_DIR / "y_train_v5.npy")

X_val = np.load(
    DATA_DIR / "X_val.npy",
    mmap_mode="r"
)
y_val = np.load(DATA_DIR / "y_val.npy")

print("Training:", X_train.shape)
print("Validation:", X_val.shape)

# -----------------------------------
# 2. DATA PIPELINE
# -----------------------------------

train_dataset = tf.data.Dataset.from_tensor_slices(
    (X_train, y_train)
)

train_dataset = (
    train_dataset
    .shuffle(2000, seed=SEED)
    .batch(32)
    .map(
        lambda images, labels: (
            tf.cast(images, tf.float32) / 255.0,
            labels
        ),
        num_parallel_calls=2
    )
    .prefetch(1)
)

val_dataset = (
    tf.data.Dataset.from_tensor_slices(
        (X_val, y_val)
    )
    .batch(32)
    .prefetch(1)
)

# -----------------------------------
# 3. BUILD MODEL
# -----------------------------------

model = Sequential([
    Input(shape=(64, 64, 3)),
    Resizing(32, 32),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(15, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------------
# 4. TRAIN
# -----------------------------------

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=30,
    callbacks=[early_stopping],
    verbose=1
)

# -----------------------------------
# 5. SAVE RESULTS
# -----------------------------------

model.save(
    MODEL_DIR / "multilayer_ann_v5_1.keras"
)

history_data = {
    key: [float(value) for value in values]
    for key, values in history.history.items()
}

with open(
    RESULTS_DIR / "multilayer_history_v5_1.json",
    "w"
) as file:
    json.dump(history_data, file, indent=4)

val_loss, val_accuracy = model.evaluate(
    val_dataset,
    verbose=0
)

print("\nMULTILAYER ANN V5.1 RESULTS")
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation Loss: {val_loss:.4f}")
print("Model and history saved!")

