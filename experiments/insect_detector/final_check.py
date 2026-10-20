
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image

BASE = Path(__file__).resolve().parent

MODEL_PATH = BASE / "models" / "insect_detector_v3.keras"
IMAGE_FOLDER = BASE / "data" / "final_check"

print("Loading Insect Detection Gate V3...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

extensions = {".jpg", ".jpeg", ".png", ".webp"}

images = sorted(
    path for path in IMAGE_FOLDER.iterdir()
    if path.is_file() and path.suffix.lower() in extensions
)

if not images:
    print("No images found in final_check folder.")
    raise SystemExit

print(f"\nTesting {len(images)} unseen images...\n")

for path in images:
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize((128, 128))
        array = np.asarray(img, dtype=np.float32)

    array = tf.keras.applications.mobilenet_v2.preprocess_input(
        array
    )

    batch = np.expand_dims(array, axis=0)

    score = float(
        model(batch, training=False).numpy()[0][0]
    )

    prediction = (
        "INSECT DETECTED"
        if score >= 0.5
        else "NO INSECT DETECTED"
    )

    print(f"Image: {path.name}")
    print(f"Insect score: {score:.4f}")
    print(f"Result: {prediction}")
    print("-" * 45)
