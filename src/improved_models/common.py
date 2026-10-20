r"""
common.py - shared helpers for the InsectClass improvement experiments.

SAFETY: every path below is inside D:\Claude_InsectClass. Nothing in
D:\InsectClass is ever opened for writing by these scripts.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------
# Paths (override with the INSECT_WORK_DIR environment variable)
# ---------------------------------------------------------------
WORK_DIR = Path(os.environ.get("INSECT_WORK_DIR", r"D:\Claude_InsectClass"))
DATA_DIR = WORK_DIR / "data" / "processed"        # COPIES of your .npy files
ORIG_DIR = WORK_DIR / "original_models"           # COPIES of your old models/histories
MODEL_DIR = WORK_DIR / "models"                   # new models are saved here
METRICS_DIR = WORK_DIR / "results" / "metrics"
PLOTS_DIR = WORK_DIR / "results" / "plots"

for d in (MODEL_DIR, METRICS_DIR, PLOTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 15
IMG_SHAPE = (64, 64, 3)


# ---------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------
def load_class_names():
    with open(DATA_DIR / "class_mapping.json", "r", encoding="utf-8") as f:
        mapping = json.load(f)
    return [mapping[str(i)]["name"] for i in range(NUM_CLASSES)]


def load_train(source):
    """source = 'original' (your 4,500 images) or 'full' (build_full_train.py output).
    Returns (X, y). X may be uint8 (full) or float32 in [0,1] (original)."""
    if source == "original":
        X = np.load(DATA_DIR / "X_train.npy")
        y = np.load(DATA_DIR / "y_train.npy")
    elif source == "full":
        X = np.load(DATA_DIR / "X_train_full.npy")
        y = np.load(DATA_DIR / "y_train_full.npy")
    else:
        raise ValueError("train source must be 'original' or 'full'")
    return X, y.astype("int32")


def load_val():
    X = np.load(DATA_DIR / "X_val.npy")           # float32 in [0,1]
    y = np.load(DATA_DIR / "y_val.npy").astype("int32")
    return X, y


def load_test():
    X = np.load(DATA_DIR / "X_test.npy", mmap_mode="r")
    y = np.load(DATA_DIR / "y_test.npy").astype("int32")
    return X, y


def pixel_scale(X):
    """uint8 arrays hold 0..255, float arrays already hold 0..1."""
    return 1.0 / 255.0 if X.dtype == np.uint8 else 1.0


def to_float01(batch, scale):
    return np.asarray(batch, dtype=np.float32) * scale


def channel_stats(X, chunk=1000):
    """Per-channel mean/variance of the TRAINING images only (streamed, no big copy)."""
    scale = pixel_scale(X)
    total = np.zeros(3, np.float64)
    total_sq = np.zeros(3, np.float64)
    count = 0
    for i in range(0, len(X), chunk):
        b = to_float01(X[i:i + chunk], scale).astype(np.float64)
        total += b.sum(axis=(0, 1, 2))
        total_sq += (b ** 2).sum(axis=(0, 1, 2))
        count += b.shape[0] * b.shape[1] * b.shape[2]
    mean = total / count
    var = total_sq / count - mean ** 2
    return mean.astype("float32"), var.astype("float32")


# ---------------------------------------------------------------
# Training pipeline (augmentation ONLY here, never on val/test)
# ---------------------------------------------------------------
def augment_image(img):
    """img: float32 (64,64,3) in [0,1]."""
    img = tf.image.random_flip_left_right(img)
    pad = 6                                        # random shift of up to 6 px
    img = tf.pad(img, [[pad, pad], [pad, pad], [0, 0]], mode="REFLECT")
    img = tf.image.random_crop(img, IMG_SHAPE)
    img = tf.image.random_brightness(img, 0.10)
    img = tf.image.random_contrast(img, 0.85, 1.15)
    return tf.clip_by_value(img, 0.0, 1.0)


def class_weight_vector(y, mode):
    if mode == "none":
        return None
    counts = np.bincount(y, minlength=NUM_CLASSES).astype("float64")
    weights = len(y) / (NUM_CLASSES * np.maximum(counts, 1))
    return weights.astype("float32")


def make_train_ds(X, y, batch_size, augment, class_weights, seed):
    scale = pixel_scale(X)
    cw = None if class_weights is None else tf.constant(class_weights, tf.float32)

    def prep(img, label):
        img = tf.cast(img, tf.float32) * scale
        if augment:
            img = augment_image(img)
        if cw is None:
            return img, label
        return img, label, tf.gather(cw, label)

    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.shuffle(min(len(X), 4096), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(prep, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------
def predict_probs(model, X, batch_size=256):
    scale = pixel_scale(X)
    out = []
    for i in range(0, len(X), batch_size):
        batch = to_float01(X[i:i + batch_size], scale)
        out.append(model(batch, training=False).numpy())
    return np.concatenate(out, axis=0)


def compute_metrics(y_true, probs):
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                 confusion_matrix)
    y_pred = np.argmax(probs, axis=1)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(NUM_CLASSES)),
        average="macro", zero_division=0)
    eps = 1e-12
    loss = float(-np.mean(np.log(probs[np.arange(len(y_true)), y_true] + eps)))
    counts = np.bincount(y_true, minlength=NUM_CLASSES)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f1),
        "loss": loss,
        "majority_class_baseline": float(counts.max() / counts.sum()),
        "uniform_chance": 1.0 / NUM_CLASSES,
        "predictions_per_class": np.bincount(y_pred, minlength=NUM_CLASSES).tolist(),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(NUM_CLASSES))).tolist(),
    }


def save_confusion_matrix(cm, class_names, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(14, 12))
    ConfusionMatrixDisplay(np.array(cm), display_labels=class_names).plot(
        ax=ax, xticks_rotation=90, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------
# Shared training routine
# ---------------------------------------------------------------
def add_common_args(parser):
    parser.add_argument("--tag", required=True, help="unique run name, e.g. s_full_aug")
    parser.add_argument("--train-source", choices=["original", "full"], default="original")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20,
                        help="early-stopping patience on val_accuracy")
    parser.add_argument("--lr-patience", type=int, default=6,
                        help="epochs without val_loss improvement before LR is halved")
    parser.add_argument("--class-weights", choices=["none", "balanced"], default="none")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def run_experiment(model, X, y, Xv, yv, args, name, architecture, config):
    """Train, restore best-val-accuracy weights, evaluate on TRAIN and VAL only,
    save model/history/metrics under distinct names. The test set is never touched."""
    import time
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    model_path = MODEL_DIR / f"{name}.keras"
    if model_path.exists():
        raise SystemExit(f"{model_path} already exists - choose a new --tag (never overwriting).")

    cw = class_weight_vector(y, args.class_weights)
    train_ds = make_train_ds(X, y, args.batch, not args.no_augment, cw, args.seed)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=args.lr_patience,
                          min_lr=1e-6, verbose=1),
        EarlyStopping(monitor="val_accuracy", mode="max", patience=args.patience,
                      restore_best_weights=True, verbose=1),
    ]

    start = time.time()
    history = model.fit(train_ds, validation_data=(Xv, yv), validation_batch_size=256,
                        epochs=args.epochs, callbacks=callbacks, shuffle=False, verbose=2)
    seconds = time.time() - start

    model.save(model_path)

    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(METRICS_DIR / f"{name}_history.json", "w") as f:
        json.dump(hist, f, indent=2)

    train_probs = predict_probs(model, X)
    train_acc = float(np.mean(np.argmax(train_probs, axis=1) == y))
    val_metrics = compute_metrics(yv, predict_probs(model, Xv))

    summary = {
        "name": name,
        "architecture": architecture,
        "config": config,
        "train_images": int(len(X)),
        "epochs_run": len(hist["loss"]),
        "best_epoch": int(np.argmax(hist["val_accuracy"])) + 1,
        "train_seconds": round(seconds, 1),
        "parameters": int(model.count_params()),
        "train_accuracy_clean": train_acc,
        "train_accuracy_last_epoch_augmented": hist["accuracy"][-1],
        "validation": val_metrics,
    }
    with open(METRICS_DIR / f"{name}_val_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 48)
    print(f"{name}  ({architecture})")
    print("=" * 48)
    print(f"Train images:                {len(X)}")
    print(f"Epochs run / best epoch:     {summary['epochs_run']} / {summary['best_epoch']}")
    print(f"Training accuracy (clean):   {train_acc:.4f}")
    print(f"Validation accuracy:         {val_metrics['accuracy']:.4f}")
    print(f"Validation macro precision:  {val_metrics['macro_precision']:.4f}")
    print(f"Validation macro recall:     {val_metrics['macro_recall']:.4f}")
    print(f"Validation macro F1:         {val_metrics['macro_f1']:.4f}")
    print(f"Majority-class baseline:     {val_metrics['majority_class_baseline']:.4f}")
    print(f"Saved model:   {model_path}")
    print("(test set not used)")
