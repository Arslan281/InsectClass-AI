import numpy as np
from pathlib import Path

DATA_DIR = Path("data/processed")

for split in ["train", "val", "test"]:

    X = np.load(DATA_DIR / f"X_{split}.npy")
    y = np.load(DATA_DIR / f"y_{split}.npy")

    print(f"\n--- {split.upper()} DATASET ---")

    print("Images shape:", X.shape)
    print("Labels shape:", y.shape)

    print("Pixel minimum:", X.min())
    print("Pixel maximum:", X.max())

    print("Missing pixel values:", np.isnan(X).sum())

    print("Unique class labels:", np.unique(y))

    classes, counts = np.unique(y, return_counts=True)

    print("\nImages per class:")

    for class_id, count in zip(classes, counts):
        print(f"Class {class_id}: {count}")