r"""
evaluate_all.py - compares ORIGINAL vs IMPROVED models on the validation or test split.

  python evaluate_all.py                              -> validation split (safe, repeatable)
  python evaluate_all.py --split test --confirm-final -> held-out test split (run ONCE,
                                                        after select_best.py)

Original models are read from D:\Claude_InsectClass\original_models (your copies).
Improved models are the two chosen by select_best.py (override with --extra).
"""
import argparse
import json
import gc
import numpy as np
import pandas as pd
import tensorflow as tf

from common import (METRICS_DIR, MODEL_DIR, ORIG_DIR, PLOTS_DIR, compute_metrics,
                    load_class_names, load_test, load_val, predict_probs,
                    save_confusion_matrix)

parser = argparse.ArgumentParser()
parser.add_argument("--split", choices=["val", "test"], default="val")
parser.add_argument("--confirm-final", action="store_true",
                    help="required for --split test (guards against tuning on test data)")
parser.add_argument("--extra", nargs="*", default=[],
                    help="extra models as label=filename.keras (looked up in models/)")
args = parser.parse_args()

if args.split == "test" and not args.confirm_final:
    raise SystemExit("The test split is for the FINAL evaluation only. Re-run with --confirm-final "
                     "once model selection is finished.")

class_names = load_class_names()
X, y = load_val() if args.split == "val" else load_test()
print(f"{args.split} images:", X.shape)

models = {}
for label, fname in (("Original Single-Layer ANN", "single_layer_ann.keras"),
                     ("Original Multilayer ANN v4", "multilayer_ann_v4.keras")):
    if (ORIG_DIR / fname).exists():
        models[label] = ORIG_DIR / fname
    else:
        print(f"[skip] {ORIG_DIR / fname} not found")

sel_path = METRICS_DIR / "selected_models.json"
if sel_path.exists():
    with open(sel_path) as f:
        sel = json.load(f)
    if "single" in sel:
        models[f"Improved Single-Layer ({sel['single']})"] = MODEL_DIR / f"{sel['single']}.keras"
    if "multilayer" in sel:
        models[f"Improved Multilayer ({sel['multilayer']})"] = MODEL_DIR / f"{sel['multilayer']}.keras"
else:
    print("[note] selected_models.json missing - run select_best.py to include improved models")

for item in args.extra:
    label, fname = item.split("=", 1)
    models[label] = MODEL_DIR / fname

rows, class_rows = [], []
for label, path in models.items():
    print(f"\nEvaluating {label} ...")
    model = tf.keras.models.load_model(path)
    m = compute_metrics(y, predict_probs(model, X))

    rows.append({"Model": label, "Split": args.split, "Accuracy": m["accuracy"],
                 "Macro Precision": m["macro_precision"], "Macro Recall": m["macro_recall"],
                 "Macro F1": m["macro_f1"], "Loss": m["loss"],
                 "Majority-class baseline": m["majority_class_baseline"]})

    cm = np.array(m["confusion_matrix"])
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        p = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        r = tp / cm[i, :].sum() if cm[i, :].sum() else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        class_rows.append({"Model": label, "Class": name, "Precision": p, "Recall": r,
                           "F1": f, "Support": int(cm[i, :].sum()),
                           "Predicted count": m["predictions_per_class"][i]})

    safe = "".join(c if c.isalnum() else "_" for c in label).strip("_").lower()
    save_confusion_matrix(m["confusion_matrix"], class_names,
                          f"{label} - {args.split} confusion matrix",
                          PLOTS_DIR / f"{safe}_{args.split}_cm.png")
    print(f"Accuracy {m['accuracy']:.4f} | P {m['macro_precision']:.4f} | "
          f"R {m['macro_recall']:.4f} | F1 {m['macro_f1']:.4f} | "
          f"majority baseline {m['majority_class_baseline']:.4f}")
    del model
    tf.keras.backend.clear_session()
    gc.collect()

df = pd.DataFrame(rows)
df.to_csv(METRICS_DIR / f"comparison_{args.split}.csv", index=False)
pd.DataFrame(class_rows).to_csv(METRICS_DIR / f"class_reports_{args.split}.csv", index=False)

print("\n" + "=" * 60)
print(f"ORIGINAL vs IMPROVED - {args.split.upper()} SPLIT")
print("=" * 60)
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"\nSaved to {METRICS_DIR} and {PLOTS_DIR}")
