
import json
import numpy as np
import tensorflow as tf
from pathlib import Path

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Keep resource usage manageable on our laptop
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.random.set_seed(42)

IMAGE_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 5

def load_manifest(name):
    with open(BASE / f"{name}_manifest.json", encoding="utf-8") as f:
        return json.load(f)

def make_dataset(records, shuffle=False):
    paths = [r["path"] for r in records]
    labels = np.array(
        [r["label"] for r in records],
        dtype=np.float32
    )

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        ds = ds.shuffle(
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
            image, [IMAGE_SIZE, IMAGE_SIZE]
        )

        image = tf.keras.applications.mobilenet_v2.preprocess_input(
            image
        )

        return image, tf.reshape(label, (1,))

    return (
        ds.map(preprocess, num_parallel_calls=2)
          .batch(BATCH_SIZE)
          .prefetch(1)
    )

train_records = load_manifest("train")
val_records = load_manifest("val")

train_ds = make_dataset(train_records, shuffle=True)
val_ds = make_dataset(val_records)

print("Training images:", len(train_records))
print("Validation images:", len(val_records))

# Pretrained image feature extractor
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    alpha=0.35,
    include_top=False,
    weights="imagenet",
    pooling="avg"
)

# Freeze pretrained weights
base_model.trainable = False

inputs = tf.keras.Input(
    shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
)

features = base_model(inputs, training=False)
x = tf.keras.layers.Dropout(0.25)(features)

# 0 = non-insect, 1 = insect
outputs = tf.keras.layers.Dense(
    1, activation="sigmoid"
)(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )
]

# Balance the influence of both classes
num_insect = sum(r["label"] == 1 for r in train_records)
num_negative = len(train_records) - num_insect

class_weight = {
    0: len(train_records) / (2 * num_negative),
    1: len(train_records) / (2 * num_insect)
}

print("\nStarting detector training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=callbacks
)

model_path = MODEL_DIR / "insect_detector.keras"
model.save(model_path)

history_path = BASE / "results" / "training_history.json"
history_path.parent.mkdir(exist_ok=True)

with open(history_path, "w", encoding="utf-8") as f:
    json.dump(
        {k: [float(v) for v in values]
         for k, values in history.history.items()},
        f,
        indent=2
    )

print("\nDetector training complete!")
print("Model saved to:", model_path)
print("Training history saved to:", history_path)
