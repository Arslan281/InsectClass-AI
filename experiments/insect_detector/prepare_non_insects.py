
from pathlib import Path
from PIL import Image
from tensorflow.keras.datasets import cifar10
import numpy as np
import random

DESTINATION = Path(
    r"D:\InsectClass\experiments\insect_detector\data\non_insect"
)

DESTINATION.mkdir(parents=True, exist_ok=True)

print("Loading CIFAR-10 dataset...")

(X_train, y_train), _ = cifar10.load_data()

random.seed(42)

# Pick 200 images from each of the 10 categories.
# Total = 2000 non-insect photographs.
selected_indices = []

for category in range(10):
    indices = np.where(y_train.flatten() == category)[0]
    selected = random.sample(indices.tolist(), 200)
    selected_indices.extend(selected)

random.shuffle(selected_indices)

for number, index in enumerate(selected_indices):
    image = Image.fromarray(X_train[index])
    image.save(DESTINATION / f"non_insect_{number:04d}.png")

print("Non-insect images saved:", len(selected_indices))
print("Saved to:", DESTINATION)
