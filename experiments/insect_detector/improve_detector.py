
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"

OLD_MODEL = BASE / "models" / "insect_detector.keras"
NEW_MODEL = BASE / "models" / "insect_detector_v3.keras"

tf.random.set_seed(42)
tf.config.threading.set_intra_op_parallelism_threads(4)

IMAGE_SIZE = 128
BATCH_SIZE = 16

# 1. Load original training and validation records
def load_records(filename):
    with open(BASE / filename, encoding="utf-8") as f:
        return json.load(f)

train_records = load_records("train_manifest.json")
val_records = load_records("val_manifest.json")

# 2. Add screenshots as hard-negative training examples
screenshots = sorted(
    (DATA / "negative_extra").glob("screenshot_*.png")
)

print("Screenshots found:", len(screenshots))

if not screenshots:
    raise RuntimeError("No screenshots found. Training cancelled.")

# Keep original training data and add screenshot examples.
# Screenshots are labelled 0 = non-insect.
for path in screenshots:
    for _ in range(20):
        train_records.append({
            "path": str(path),
            "label": 0
        })

print("New training images:", len(train_records))
print("Validation images:", len(val_records))

# 3. Same image preprocessing as original detector
def make_dataset(records, shuffle=False):
    paths = [str(r["path"]) for r in records]
    labels = np.array(
        [r["label"] for r in records],
        dtype=np.float32
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if shuffle:
        dataset = dataset.shuffle(
            len(records),
            seed=42,
            reshuffle_each_iteration=True
        )

    def preprocess(path, label):
        raw = tf.io.read_file(path)

        image = tf.io.decode_image(
            raw,
            channels=3,
            expand_animations=False
        )

        image.set_shape([None, None, 3])
        image = tf.image.resize(
            image,
            [IMAGE_SIZE, IMAGE_SIZE]
        )

        image = (
            tf.keras.applications.mobilenet_v2
            .preprocess_input(image)
        )

        return image, tf.reshape(label, (1,))

    return (
        dataset
        .map(preprocess, num_parallel_calls=2)
        .batch(BATCH_SIZE)
        .prefetch(1)
    )

train_ds = make_dataset(train_records, shuffle=True)
val_ds = make_dataset(val_records)

# 4. Load our original detector
print("\nLoading original detector...")

model = tf.keras.models.load_model(
    OLD_MODEL,
    compile=False
)

# Keep pretrained feature extractor frozen.
for layer in model.layers:
    layer.trainable = False

# Train only the final binary classification layer.
model.layers[-1].trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

# Balance insect and non-insect training samples
insect_count = sum(
    r["label"] == 1 for r in train_records
)

negative_count = len(train_records) - insect_count

class_weights = {
    0: len(train_records) / (2 * negative_count),
    1: len(train_records) / (2 * insect_count)
}

# 5. Retrain the classification head
print("\nImproving insect detector...")

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    class_weight=class_weights,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True
        )
    ],
    verbose=2
)

# 6. Save separately — original model untouched
model.save(NEW_MODEL)

print("\nSUCCESS!")
print("Improved detector saved to:", NEW_MODEL)
print("Original detector preserved:", OLD_MODEL)
