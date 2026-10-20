
from pathlib import Path
from collections import defaultdict
import hashlib
import numpy as np

DATA_DIR = Path("data/processed")

X_old = np.load(DATA_DIR / "X_train.npy", mmap_mode="r")
y_old = np.load(DATA_DIR / "y_train.npy")

X_new = np.load(DATA_DIR / "X_train_v5.npy", mmap_mode="r")
y_new = np.load(DATA_DIR / "y_train_v5.npy")

print("Checking image-label consistency...")

# Build lookup from normalized original images.
# Convert original pixels back to uint8 for comparison.
old_lookup = defaultdict(set)

for image, label in zip(X_old, y_old):
    pixels = np.rint(image * 255).astype(np.uint8)
    image_hash = hashlib.sha256(pixels.tobytes()).digest()
    old_lookup[image_hash].add(int(label))

matches = 0
mismatches = 0

for image, label in zip(X_new, y_new):
    image_hash = hashlib.sha256(image.tobytes()).digest()

    if image_hash in old_lookup:
        matches += 1

        if int(label) not in old_lookup[image_hash]:
            mismatches += 1

print("\n=== CONSISTENCY RESULTS ===")
print("Matching images:", matches)
print("Label mismatches:", mismatches)

if matches == 0:
    print("No matching image hashes found. Check preprocessing consistency.")
elif mismatches == 0:
    print("All matched images have consistent labels!")
else:
    print("WARNING: Label inconsistencies detected!")
