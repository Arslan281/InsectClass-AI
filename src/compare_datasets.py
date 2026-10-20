
from pathlib import Path
import numpy as np
from collections import Counter

DATA_DIR = Path("data/processed")

# Original balanced dataset
X_old = np.load(DATA_DIR / "X_train.npy", mmap_mode="r")
y_old = np.load(DATA_DIR / "y_train.npy")

# Larger dataset
X_new = np.load(DATA_DIR / "X_train_v5.npy", mmap_mode="r")
y_new = np.load(DATA_DIR / "y_train_v5.npy")

print("=== DATASET COMPARISON ===")

print("\nOriginal dataset:")
print("Shape:", X_old.shape)
print("Labels:", Counter(y_old.tolist()))

print("\nLarger dataset:")
print("Shape:", X_new.shape)
print("Labels:", Counter(y_new.tolist()))

# Compare pixel ranges
print("\nPixel ranges:")
print("Original:", X_old.min(), X_old.max())
print("Larger:", X_new.min(), X_new.max())

# Verify label range
assert set(np.unique(y_old)) == set(range(15))
assert set(np.unique(y_new)) == set(range(15))

print("\nAll 15 labels present in both datasets.")

# Convert first new image to normalized float32
sample_new = X_new[0].astype(np.float32) / 255.0

print("\nFirst new sample:")
print("Label:", y_new[0])
print("Mean brightness:", sample_new.mean())

print("\nComparison completed.")
