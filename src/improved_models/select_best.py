r"""
select_best.py - compares all improved runs using VALIDATION metrics only and
records the best single-layer and best multilayer model. Test set is not touched.
"""
import json
import pandas as pd
from common import METRICS_DIR

rows = []
for path in sorted(METRICS_DIR.glob("improved_*_val_metrics.json")):
    with open(path) as f:
        s = json.load(f)
    v = s["validation"]
    rows.append({
        "name": s["name"],
        "type": "single" if "Single" in s["architecture"] else "multilayer",
        "train_images": s["train_images"],
        "best_epoch": s["best_epoch"],
        "train_acc": s["train_accuracy_clean"],
        "val_acc": v["accuracy"],
        "val_macro_p": v["macro_precision"],
        "val_macro_r": v["macro_recall"],
        "val_macro_f1": v["macro_f1"],
        "val_majority_baseline": v["majority_class_baseline"],
    })

if not rows:
    raise SystemExit("No improved_*_val_metrics.json files found - run the training scripts first.")

df = pd.DataFrame(rows).sort_values(["type", "val_acc", "val_macro_f1"], ascending=[True, False, False])
df.to_csv(METRICS_DIR / "validation_experiments.csv", index=False)
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

selected = {}
for kind in ("single", "multilayer"):
    sub = df[df["type"] == kind]
    if len(sub):
        selected[kind] = sub.iloc[0]["name"]

with open(METRICS_DIR / "selected_models.json", "w") as f:
    json.dump(selected, f, indent=2)
print("\nSelected on VALIDATION accuracy (tie-break: macro F1):", selected)
