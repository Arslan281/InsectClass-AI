
from pathlib import Path
from collections import Counter
import json

import numpy as np
from PIL import Image

# -----------------------------------
# 1. CONFIGURATION
# -----------------------------------

DATA_DIR = Path("data/ip102_v1.1")
IMAGE_DIR = Path(r"D:\Datasets for ANN\ip102_v1.1\images")
OUTPUT_DIR = Path("data/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (64, 64)

# Load the existing 15-class mapping
with open(
    OUTPUT_DIR / "class_mapping.json",
    "r",
    encoding="utf-8"
) as file:
    class_info = json.load(file)

# Original IP102 class ID -> new class ID
label_mapping = {
    info["original_id"]: int(new_id)
    for new_id, info in class_info.items()
}

# -----------------------------------
# 2. LOAD TRAINING ANNOTATIONS
# -----------------------------------

records = []

with open(DATA_DIR / "train.txt", "r") as file:
    for line in file:
        filename, label = line.strip().split()
        label = int(label)

        if label in label_mapping:
            records.append(
                (filename, label_mapping[label])
            )

print("Total selected training images:", len(records))

# -----------------------------------
# 3. CREATE MEMORY-EFFICIENT ARRAYS
# -----------------------------------

X = np.lib.format.open_memmap(
    OUTPUT_DIR / "X_train_v5.npy",
    mode="w+",
    dtype=np.uint8,
    shape=(len(records), 64, 64, 3)
)

y = np.zeros(len(records), dtype=np.int32)

# -----------------------------------
# 4. PROCESS ALL TRAINING IMAGES
# -----------------------------------

for i, (filename, label) in enumerate(records):

    image_path = IMAGE_DIR / filename

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize(IMAGE_SIZE)

        X[i] = np.asarray(image, dtype=np.uint8)

    y[i] = label

    if (i + 1) % 1000 == 0:
        print(f"Processed {i + 1}/{len(records)} images")

X.flush()

np.save(
    OUTPUT_DIR / "y_train_v5.npy",
    y
)

# -----------------------------------
# 5. CLASS DISTRIBUTION
# -----------------------------------

counts = Counter(y.tolist())

print("\nV5 TRAINING DATASET")
print("Images:", X.shape)
print("Labels:", y.shape)
print("Data type:", X.dtype)

print("\nImages per class:")

for class_id in range(15):
    print(f"Class {class_id}: {counts[class_id]}")

print("\nV5 preprocessing completed successfully!")
