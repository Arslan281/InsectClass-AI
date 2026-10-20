
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path
import json
import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# -----------------------------------
# 1. CONFIGURATION
# -----------------------------------

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results/metrics")
PLOTS_DIR = Path("results/plots")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

models = {
    "Single-Layer ANN": "single_layer_ann.keras",
    "Multilayer v1": "multilayer_ann.keras",
    "Multilayer v2": "multilayer_ann_v2.keras",
    "Multilayer v3": "multilayer_ann_v3.keras",
    "Multilayer v4": "multilayer_ann_v4.keras"
}
# -----------------------------------
# 2. LOAD VALIDATION DATA
# -----------------------------------

X_val = np.load(
    DATA_DIR / "X_val.npy",
    mmap_mode="r"
)
y_val = np.load(DATA_DIR / "y_val.npy")

with open(DATA_DIR / "class_mapping.json", "r") as f:
    class_mapping = json.load(f)

class_names = [
    class_mapping[str(i)]["name"]
    for i in range(15)
]

labels = list(range(15))

print("Validation images:", X_val.shape)
print("Number of classes:", len(class_names))

# -----------------------------------
# 3. EVALUATE EACH MODEL
# -----------------------------------

summary = []
all_class_reports = []

for model_name, filename in models.items():

    print(f"\nEvaluating {model_name}...")

    model = tf.keras.models.load_model(
        MODEL_DIR / filename
    )

    probabilities = model.predict(
        X_val,
        batch_size=32,
        verbose=1
    )

    predictions = np.argmax(
        probabilities, axis=1
    )

    accuracy = accuracy_score(
        y_val, predictions
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_val,
            predictions,
            labels=labels,
            average="macro",
            zero_division=0
        )
    )

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Macro F1:  {f1:.4f}")

    summary.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Macro Precision": precision,
        "Macro Recall": recall,
        "Macro F1": f1
    })

    # Detailed class-level performance
    report = classification_report(
        y_val,
        predictions,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    report_rows = []

    for name in class_names:
        result = report[name]
        report_rows.append({
            "Model": model_name,
            "Insect": name,
            "Precision": result["precision"],
            "Recall": result["recall"],
            "F1": result["f1-score"],
            "Support": result["support"]
        })

    all_class_reports.extend(report_rows)

    # -----------------------------------
    # 4. CONFUSION MATRIX
    # -----------------------------------

    cm = confusion_matrix(
        y_val,
        predictions,
        labels=labels
    )

    plt.figure(figsize=(15, 12))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title(f"{model_name} - Validation Confusion Matrix")
    plt.xlabel("Predicted Insect")
    plt.ylabel("Actual Insect")
    plt.xticks(rotation=70, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    safe_name = (
        model_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    plt.savefig(
        PLOTS_DIR / f"{safe_name}_val_cm.png",
        dpi=150
    )

    plt.close()

    # Examine which classes the model predicts
    pred_counts = np.bincount(
        predictions,
        minlength=15
    )

    print("\nPredictions per class:")

    for i, count in enumerate(pred_counts):
        print(f"{class_names[i]}: {count}")

    # Release model memory before loading next one
    del model, probabilities, predictions
    tf.keras.backend.clear_session()
    gc.collect()

# -----------------------------------
# 5. SAVE COMPARISON RESULTS
# -----------------------------------

summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    RESULTS_DIR / "validation_model_comparison.csv",
    index=False
)

report_df = pd.DataFrame(all_class_reports)

report_df.to_csv(
    RESULTS_DIR / "validation_class_reports.csv",
    index=False
)

print("\n================================")
print("FINAL VALIDATION COMPARISON")
print("================================")

print(
    summary_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nValidation diagnostics completed!")
print("Results saved to results/metrics and results/plots.")
