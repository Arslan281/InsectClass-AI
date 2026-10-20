
from pathlib import Path
import numpy as np

DATA_DIR = Path("data/processed")

X_train = np.load(
    DATA_DIR / "X_train_v5.npy",
    mmap_mode="r"
)

y_train = np.load(
    DATA_DIR / "y_train_v5.npy"
)

X_val = np.load(
    DATA_DIR / "X_val.npy",
    mmap_mode="r"
)

y_val = np.load(
    DATA_DIR / "y_val.npy"
)

print("=== V5 DATA CHECK ===")

print("\nTraining:")
print("Shape:", X_train.shape)
print("Type:", X_train.dtype)
print("Minimum:", X_train.min())
print("Maximum:", X_train.max())

print("\nValidation:")
print("Shape:", X_val.shape)
print("Type:", X_val.dtype)
print("Minimum:", X_val.min())
print("Maximum:", X_val.max())

# Simulate training normalization
sample = X_train[:32].astype(np.float32) / 255.0

print("\nNormalized training batch:")
print("Minimum:", sample.min())
print("Maximum:", sample.max())

print("\nLabels:")
print("Training classes:", np.unique(y_train))
print("Validation classes:", np.unique(y_val))

# Compare image statistics after normalization
train_sample = X_train[:100].astype(np.float32) / 255.0
val_sample = X_val[:100]

print("\nAverage pixel brightness:")
print("Training:", train_sample.mean())
print("Validation:", val_sample.mean())

print("\nData verification complete!")
