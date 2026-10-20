
from pathlib import Path
import json

import numpy as np
import tensorflow as tf
from PIL import Image

# -----------------------------------
# 1. PROJECT PATHS
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"

DETECTOR_PATH = (
    BASE_DIR
    / "experiments"
    / "insect_detector"
    / "models"
    / "insect_detector_v3.keras"
)

DETECTION_THRESHOLD = 0.5

# -----------------------------------
# 2. LOAD CLASS NAMES
# -----------------------------------

with open(
    DATA_DIR / "class_mapping.json",
    "r",
    encoding="utf-8"
) as file:
    class_mapping = json.load(file)

class_names = []

for i in range(15):
    info = class_mapping[str(i)]

    if isinstance(info, dict):
        class_names.append(info.get("name", str(i)))
    else:
        class_names.append(str(info))

# -----------------------------------
# -----------------------------------
# 3. LOAD IMPROVED ANN MODELS
# -----------------------------------

print("Loading improved InsectClass AI models...")

models = {
    "single_layer": tf.keras.models.load_model(
        MODEL_DIR / "improved_single_layer_s_full_noaug.keras",
        compile=False
    ),

    "multilayer": tf.keras.models.load_model(
        MODEL_DIR / "improved_multilayer_m_final_512.keras",
        compile=False
    )
}

print("Both improved ANN models loaded successfully!")
# -----------------------------------
# 4. LOAD INSECT DETECTION GATE
# -----------------------------------

print("Loading Insect Detection Gate V3...")

detector = tf.keras.models.load_model(
    DETECTOR_PATH,
    compile=False
)

print("Insect Detection Gate V3 loaded successfully!")

# -----------------------------------
# 5. PREPROCESS FOR INSECT DETECTOR
# -----------------------------------

def preprocess_detector_image(image: Image.Image):

    image = image.convert("RGB")
    image = image.resize((128, 128))

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    image_array = (
        tf.keras.applications.mobilenet_v2
        .preprocess_input(image_array)
    )

    return np.expand_dims(image_array, axis=0)

# -----------------------------------
# 6. CHECK WHETHER IMAGE HAS AN INSECT
# -----------------------------------

def detect_insect(image: Image.Image):

    processed_image = preprocess_detector_image(image)

    score = float(
        detector(
            processed_image,
            training=False
        ).numpy().reshape(-1)[0]
    )

    return {
        "is_insect": score >= DETECTION_THRESHOLD,
        "insect_score": round(score, 4),
        "threshold": DETECTION_THRESHOLD
    }

# -----------------------------------
# 7. PREPROCESS FOR ORIGINAL ANNS
# -----------------------------------

def preprocess_image(image: Image.Image):

    image = image.convert("RGB")
    image = image.resize((64, 64))

    image_array = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    return np.expand_dims(image_array, axis=0)

# -----------------------------------
# 8. RUN ORIGINAL ANN CLASSIFIERS
# -----------------------------------

def predict_insect(image: Image.Image):

    processed_image = preprocess_image(image)

    results = {}

    for model_name, model in models.items():

        probabilities = model.predict(
            processed_image,
            verbose=0
        )[0]

        top_indices = np.argsort(
            probabilities
        )[::-1][:3]

        top_predictions = []

        for index in top_indices:

            top_predictions.append({
                "insect": class_names[int(index)],
                "confidence": round(
                    float(probabilities[index]) * 100,
                    2
                )
            })

        results[model_name] = {
            "predicted_insect": class_names[
                int(np.argmax(probabilities))
            ],
            "top_predictions": top_predictions
        }

    return results

# -----------------------------------
# 9. COMPLETE PREDICTION PIPELINE
# -----------------------------------

def analyze_image(image: Image.Image):

    detection = detect_insect(image)

    # Reject images without a detected insect
    if not detection["is_insect"]:

        return {
            "is_insect": False,
            "insect_score": detection["insect_score"],
            "message": (
                "No insect detected. Please upload "
                "a clear photograph containing an insect."
            ),
            "predictions": None
        }

    # Insect detected: run both original ANNs
    predictions = predict_insect(image)

    return {
        "is_insect": True,
        "insect_score": detection["insect_score"],
        "message": "Insect detected successfully.",
        "predictions": predictions
    }
