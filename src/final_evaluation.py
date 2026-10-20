
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path
import json
import gc

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# -----------------------------------
# 1. CONFIGURATION
# -----------------------------------

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
METRICS_DIR = Path("results/metrics")
PLOTS_DIR = Path("results/plots")

METRICS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "Single-Layer ANN": "single_layer_ann.keras",
    "Multilayer ANN v4": "multilayer_ann_v4.keras"
}

# -----------------------------------
# 2. LOAD UNTOUCHED TEST DATA
# -----------------------------------

X_test = np.load(
    DATA_DIR / "X_test.npy",
    mmap_mode="r"
)

y_test = np.load(
    DATA_DIR / "y_test.npy"
)

with open(
    DATA_DIR / "class_mapping.json",
    "r",
    encoding="utf-8"
) as file:
    class_mapping = json.load(file)

# Our class IDs are 0 through 14
class_names = []

for i in range(15):
    info = class_mapping[str(i)]

    if isinstance(info, dict):
        class_names.append(info.get("name", str(i)))
    else:
        class_names.append(str(info))

print("Test images:", X_test.shape)
print("Number of classes:", len(class_names))

# -----------------------------------
# 3. EVALUATE SELECTED MODELS
# -----------------------------------

results = []
class_reports = []

for model_name, filename in MODELS.items():

    print(f"\nEvaluating {model_name}...")

    model = tf.keras.models.load_model(
        MODEL_DIR / filename
    )

    probabilities = model.predict(
        X_test,
        batch_size=32,
        verbose=1
    )

    y_pred = np.argmax(probabilities, axis=1)

    accuracy = accuracy_score(y_test, y_pred)

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_test,
            y_pred,
            labels=list(range(15)),
            average="macro",
            zero_division=0
        )
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Macro Precision": precision,
        "Macro Recall": recall,
        "Macro F1": f1
    })

    report = classification_report(
        y_test,
        y_pred,
        labels=list(range(15)),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    for class_name in class_names:
        row = report[class_name]

        class_reports.append({
            "Model": model_name,
            "Class": class_name,
            "Precision": row["precision"],
            "Recall": row["recall"],
            "F1": row["f1-score"],
            "Support": row["support"]
        })

    # -----------------------------------
    # 4. CONFUSION MATRIX
    # -----------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=list(range(15))
    )

    fig, ax = plt.subplots(figsize=(14, 12))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    display.plot(
        ax=ax,
        xticks_rotation=90,
        cmap="Blues",
        values_format="d",
        colorbar=False
    )

    ax.set_title(
        f"{model_name} - Test Confusion Matrix"
    )

    fig.tight_layout()

    safe_name = filename.replace(".keras", "")

    fig.savefig(
        PLOTS_DIR / f"{safe_name}_test_cm.png",
        dpi=160,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall:    {recall:.4f}")
    print(f"Macro F1:        {f1:.4f}")

    del model, probabilities, y_pred
    tf.keras.backend.clear_session()
    gc.collect()

# -----------------------------------
# 5. SAVE FINAL RESULTS
# -----------------------------------

comparison = pd.DataFrame(results)

comparison.to_csv(
    METRICS_DIR / "final_test_comparison.csv",
    index=False
)

pd.DataFrame(class_reports).to_csv(
    METRICS_DIR / "final_test_class_reports.csv",
    index=False
)

print("\n================================")
print("FINAL TEST COMPARISON")
print("================================")

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nFinal evaluation completed!")
print("Results saved to results/metrics")
print("Confusion matrices saved to results/plots")
