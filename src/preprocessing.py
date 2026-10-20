
from pathlib import Path
import random
import json

import numpy as np
from PIL import Image

# ----------------------------------
# 1. CONFIGURATION
# ----------------------------------

DATA_DIR = Path("data/ip102_v1.1")
IMAGE_DIR = Path(r"D:\Datasets for ANN\ip102_v1.1\images")
OUTPUT_DIR = Path("data/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (64, 64)
TRAIN_PER_CLASS = 300
SEED = 42

random.seed(SEED)

# Original zero-based IP102 class IDs
selected_classes = {
    67: "Lycorma delicatula",
    24: "Aphids",
    51: "Blister beetle",
    22: "Corn borer",
    15: "Mole cricket",
    39: "Beet army worm",
    50: "Legume blister beetle",
    58: "Limacodidae",
    48: "Locustoidea",
    69: "Cicadella viridis",
    68: "Xylotrechus",
    14: "Grub",
    10: "Rice water weevil",
    37: "Flea beetle",
    76: "Icerya purchasi Maskell",
}

# Convert original IDs to new labels 0-14
label_mapping = {
    old_id: new_id
    for new_id, old_id in enumerate(selected_classes)
}

# ----------------------------------
# 2. LOAD IMAGE ANNOTATIONS
# ----------------------------------

def load_annotations(split):
    records = []

    with open(DATA_DIR / f"{split}.txt", "r") as file:
        for line in file:
            filename, label = line.strip().split()
            label = int(label)

            if label in selected_classes:
                records.append((filename, label))

    return records


# ----------------------------------
# 3. PREPROCESS IMAGES
# ----------------------------------

def process_image(filename):
    image_path = IMAGE_DIR / filename

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize(IMAGE_SIZE)

        array = np.asarray(image, dtype=np.float32)

    # Normalize pixel values between 0 and 1
    return array / 255.0


# ----------------------------------
# 4. PREPARE DATASET
# ----------------------------------

def prepare_split(split):
    records = load_annotations(split)

    if split == "train":
        selected_records = []

        for class_id in selected_classes:
            class_records = [
                record for record in records
                if record[1] == class_id
            ]

            random.shuffle(class_records)

            selected_records.extend(
                class_records[:TRAIN_PER_CLASS]
            )

        random.shuffle(selected_records)
        records = selected_records

    images = []
    labels = []

    for index, (filename, old_label) in enumerate(records):
        try:
            image = process_image(filename)

            images.append(image)
            labels.append(label_mapping[old_label])

        except Exception as error:
            print(f"Skipping {filename}: {error}")

        if (index + 1) % 500 == 0:
            print(f"{split}: Processed {index + 1}/{len(records)}")

    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    np.save(OUTPUT_DIR / f"X_{split}.npy", X)
    np.save(OUTPUT_DIR / f"y_{split}.npy", y)

    print(f"\n{split.upper()} COMPLETE")
    print("Images shape:", X.shape)
    print("Labels shape:", y.shape)

    return X, y


# ----------------------------------
# 5. SAVE CLASS INFORMATION
# ----------------------------------

class_info = {
    str(label_mapping[old_id]): {
        "original_id": old_id,
        "name": name
    }
    for old_id, name in selected_classes.items()
}

with open(OUTPUT_DIR / "class_mapping.json", "w") as file:
    json.dump(class_info, file, indent=4)


# ----------------------------------
# 6. RUN PREPROCESSING
# ----------------------------------

if __name__ == "__main__":

    print("Starting InsectClass AI preprocessing...")

    prepare_split("train")
    prepare_split("val")
    prepare_split("test")

    print("\nAll datasets prepared successfully!")
