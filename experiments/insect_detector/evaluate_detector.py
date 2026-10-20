
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

BASE = Path(__file__).resolve().parent

MODEL_PATH = BASE / "models" / "insect_detector_v3.keras"
TEST_PATH = BASE / "test_manifest.json"

print("Loading detector...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

with open(TEST_PATH, encoding="utf-8") as f:
    records = json.load(f)

actual = []
predicted = []
scores = []

# Keep exactly the same preprocessing used in training
def prepare_image(path):
    raw = tf.io.read_file(str(path))

    image = tf.io.decode_image(
        raw,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(image, [128, 128])

    image = tf.keras.applications.mobilenet_v2.preprocess_input(
        image
    )

    return image

BATCH_SIZE = 16

print("Evaluating", len(records), "images...")

for start in range(0, len(records), BATCH_SIZE):
    batch_records = records[start:start + BATCH_SIZE]

    images = tf.stack([
        prepare_image(record["path"])
        for record in batch_records
    ])

    probabilities = model(
        images,
        training=False
    ).numpy().reshape(-1)

    for record, probability in zip(
        batch_records, probabilities
    ):
        actual.append(record["label"])
        scores.append(float(probability))
        predicted.append(int(probability >= 0.5))

print("\nTEST ACCURACY:")
print(f"{accuracy_score(actual, predicted) * 100:.2f}%")

print("\nCLASSIFICATION REPORT:")
print(classification_report(
    actual,
    predicted,
    target_names=["Non-insect", "Insect"],
    digits=4,
    zero_division=0
))

print("\nCONFUSION MATRIX:")
print(confusion_matrix(
    actual,
    predicted,
    labels=[0, 1]
))

# Separate evaluation for screenshots
screenshot_results = [
    (record, score, prediction)
    for record, score, prediction in zip(
        records, scores, predicted
    )
    if Path(record["path"]).name.startswith("screenshot_")
]

print("\nSCREENSHOT REJECTION TEST:")

rejected = sum(
    prediction == 0
    for _, _, prediction in screenshot_results
)

print("Screenshots tested:", len(screenshot_results))
print("Correctly rejected:", rejected)
print("Incorrectly accepted:", len(screenshot_results) - rejected)

for record, score, prediction in screenshot_results:
    print(
        Path(record["path"]).name,
        "| Insect score:",
        f"{score:.4f}",
        "|",
        "REJECTED" if prediction == 0 else "ACCEPTED"
    )
