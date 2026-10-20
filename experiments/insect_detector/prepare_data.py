
from pathlib import Path
from shutil import copy2
import random

# Original dataset
SOURCE = Path(r"D:\Datasets for ANN\ip102_v1.1\images")

# Official training annotations
TRAIN_FILE = Path(r"D:\InsectClass\data\ip102_v1.1\train.txt")

# New detector dataset
DESTINATION = Path(
    r"D:\InsectClass\experiments\insect_detector\data\insect"
)

DESTINATION.mkdir(parents=True, exist_ok=True)

random.seed(42)

# Read filenames from the official training split
filenames = []

with open(TRAIN_FILE, "r", encoding="utf-8") as file:
    for line in file:
        parts = line.strip().split()

        if len(parts) >= 2:
            filenames.append(parts[0])

filenames = list(set(filenames))
random.shuffle(filenames)

# Copy up to 2000 training images
copied = 0

for filename in filenames:
    source_file = SOURCE / filename

    if not source_file.is_file():
        continue

    destination_file = DESTINATION / source_file.name

    copy2(source_file, destination_file)
    copied += 1

    if copied >= 2000:
        break

print("Insect images copied:", copied)
print("Saved to:", DESTINATION)
