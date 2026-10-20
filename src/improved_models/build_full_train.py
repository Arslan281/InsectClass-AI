r"""
build_full_train.py - OPTIONAL. Builds X_train_full.npy / y_train_full.npy from ALL
official IP102 *training-partition* images of your 15 classes (your original script
kept only 300 per class).

Guarantees
  * Reads only train.txt images. val.txt / test.txt images are never added.
  * Uses the SAME class mapping (class_mapping.json), SAME resize call, SAME 64x64.
  * Drops any training image whose 64x64 pixels exactly equal a val/test image
    (exact-duplicate leakage guard). Val/test arrays are never modified.
  * Reads (never writes) D:\InsectClass annotations and the image folder.
  * Writes only into D:\Claude_InsectClass\data\processed.

Run:  python build_full_train.py
"""
import hashlib
import os
from pathlib import Path

import numpy as np
from PIL import Image

from common import DATA_DIR, IMG_SHAPE, load_class_names  # noqa: F401  (creates folders)
import json

ANNOT_DIR = Path(os.environ.get("INSECT_ANNOT_DIR", r"D:\Claude_InsectClass\data\ip102_v1.1"))
IMAGE_DIR = Path(os.environ.get("INSECT_IMAGE_DIR", r"D:\Datasets for ANN\ip102_v1.1\images"))
SEED = 42


def read_split(name):
    rows = []
    with open(ANNOT_DIR / f"{name}.txt", "r") as f:
        for line in f:
            if line.strip():
                filename, label = line.strip().split()
                rows.append((filename, int(label)))
    return rows


def pixel_hash(uint8_image):
    return hashlib.sha256(np.ascontiguousarray(uint8_image).tobytes()).digest()


def main():
    with open(DATA_DIR / "class_mapping.json", "r", encoding="utf-8") as f:
        info = json.load(f)
    old_to_new = {int(v["original_id"]): int(k) for k, v in info.items()}

    train_rows, val_rows, test_rows = read_split("train"), read_split("val"), read_split("test")
    overlap = ({r[0] for r in train_rows} & {r[0] for r in val_rows}) \
        | ({r[0] for r in train_rows} & {r[0] for r in test_rows}) \
        | ({r[0] for r in val_rows} & {r[0] for r in test_rows})
    print("Filename overlap between official partitions:", len(overlap))

    records = [(fn, old_to_new[lab]) for fn, lab in train_rows if lab in old_to_new]
    print("Official training images for the 15 classes:", len(records))

    X = np.empty((len(records), *IMG_SHAPE), dtype=np.uint8)
    y = np.empty(len(records), dtype=np.int32)
    kept = 0
    for i, (filename, label) in enumerate(records):
        try:
            with Image.open(IMAGE_DIR / filename) as im:
                im = im.convert("RGB").resize(IMG_SHAPE[:2])   # identical call to preprocessing.py
                X[kept] = np.asarray(im, dtype=np.uint8)
            y[kept] = label
            kept += 1
        except Exception as error:
            print(f"Skipping {filename}: {error}")
        if (i + 1) % 1000 == 0:
            print(f"  processed {i + 1}/{len(records)}")
    X, y = X[:kept], y[:kept]

    # ---- exact-duplicate guard against val/test (train side only is filtered) ----
    held_out = set()
    for split in ("val", "test"):
        arr = np.load(DATA_DIR / f"X_{split}.npy", mmap_mode="r")
        for j in range(len(arr)):
            held_out.add(pixel_hash(np.rint(arr[j] * 255).astype(np.uint8)))
    keep_mask = np.array([pixel_hash(X[i]) not in held_out for i in range(len(X))])
    print("Training images identical to a val/test image (removed):", int((~keep_mask).sum()))
    X, y = X[keep_mask], y[keep_mask]

    # ---- consistency check against your original 4,500-image training set ----
    X_old = np.load(DATA_DIR / "X_train.npy", mmap_mode="r")
    y_old = np.load(DATA_DIR / "y_train.npy")
    lookup = {}
    for i in range(len(X)):
        lookup.setdefault(pixel_hash(X[i]), set()).add(int(y[i]))
    found = wrong = 0
    for j in range(len(X_old)):
        h = pixel_hash(np.rint(X_old[j] * 255).astype(np.uint8))
        if h in lookup:
            found += 1
            wrong += int(int(y_old[j]) not in lookup[h])
    print(f"Original train images found in new set: {found}/{len(X_old)}  label mismatches: {wrong}")
    if wrong:
        raise SystemExit("Label mismatch detected - do not use this file.")

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(X))
    X, y = X[order], y[order]

    np.save(DATA_DIR / "X_train_full.npy", X)
    np.save(DATA_DIR / "y_train_full.npy", y)
    print("\nSaved X_train_full.npy", X.shape, " y_train_full.npy", y.shape)
    print("Images per class:", np.bincount(y, minlength=15).tolist())


if __name__ == "__main__":
    main()
