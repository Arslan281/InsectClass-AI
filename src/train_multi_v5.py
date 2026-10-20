
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
from sklearn.utils.class_weight import compute_class_weight

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

# uint8 training images, memory-mapped
X_train = np.load(
    DATA_DIR / "X_train_v5.npy",
    mmap_mode="r"
)

y_train = np.load(
    DATA_DIR / "y_train_v5.npy"
)

# Validation images are already normalized
X_val = np.load(
    DATA_DIR / "X_val.npy",
    mmap_mode="r"
)

y_val = np.load(
    DATA_DIR / "y_val.npy"
)

print("Training images:", X_train.shape)
print("Validation images:", X_val.shape)

# -----------------------------------
# 3. CREATE MEMORY-EFFICIENT DATASET
# -----------------------------------

def training_generator():
    for image, label in zip(X_train, y_train):
        yield image, label

train_dataset = tf.data.Dataset.from_generator(
    training_generator,
    output_signature=(
        tf.TensorSpec(
            shape=(64, 64, 3),
            dtype=tf.uint8
        ),
        tf.TensorSpec(
            shape=(),
            dtype=tf.int32
        )
    )
)

# Normalize each batch to 0-1
train_dataset = (
    train_dataset
    .shuffle(2000, seed=SEED)
    .batch(32)
    .map(
        lambda images, labels: (
            tf.cast(images, tf.float32) / 255.0,
            labels
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    .prefetch(tf.data.AUTOTUNE)
)

# Validation images are already normalized
val_dataset = tf.data.Dataset.from_tensor_slices(
    (X_val, y_val)
).batch(32).prefetch(tf.data.AUTOTUNE)

# -----------------------------------
# 4. CALCULATE CLASS WEIGHTS
# -----------------------------------

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = {
    int(class_id): float(weight)
    for class_id, weight in zip(classes, weights)
}

print("\nClass weights:", class_weights)

# -----------------------------------
# 5. BUILD MULTILAYER ANN V5
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
# 6. TRAIN MODEL
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
    class_weight=class_weights,
    callbacks=[early_stopping],
    verbose=1
)

# -----------------------------------
# 7. SAVE MODEL AND HISTORY
# -----------------------------------

model.save(
    MODEL_DIR / "multilayer_ann_v5.keras"
)

history_data = {
    key: [float(v) for v in values]
    for key, values in history.history.items()
}

with open(
    RESULTS_DIR / "multilayer_history_v5.json",
    "w"
) as file:
    json.dump(history_data, file, indent=4)

# -----------------------------------
# 8. FINAL VALIDATION
# -----------------------------------

val_loss, val_accuracy = model.evaluate(
    val_dataset,
    verbose=0
)

print("\n================================")
print("MULTILAYER ANN V5 RESULTS")
print("================================")

print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation Loss: {val_loss:.4f}")

print("\nModel and training history saved!")
