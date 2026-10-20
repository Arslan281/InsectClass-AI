from pathlib import Path
import random
import matplotlib.pyplot as plt
from PIL import Image

# Dataset paths
DATA_DIR = Path("data/ip102_v1.1")
IMAGE_DIR = Path(r"D:\Datasets for ANN\ip102_v1.1\images")

# Candidate insect classes (zero-based IP102 IDs)
selected_classes = {
    101: "Cicadellidae",
    70: "Miridae",
    50: "Legume blister beetle",
    23: "Army worm",
    54: "Thrips",
}

# Read training annotations
class_images = {class_id: [] for class_id in selected_classes}

with open(DATA_DIR / "train.txt", "r") as file:
    for line in file:
        filename, label = line.strip().split()
        label = int(label)

        if label in class_images:
            class_images[label].append(filename)

# Reproducible random selection
random.seed(42)

# Create comparison figure
fig, axes = plt.subplots(
    len(selected_classes),
    5,
    figsize=(15, 36)
)

for row, (class_id, insect_name) in enumerate(selected_classes.items()):

    samples = random.sample(class_images[class_id], 5)

    for col, filename in enumerate(samples):
        image_path = IMAGE_DIR / filename

        with Image.open(image_path) as img:
            image = img.convert("RGB")
            axes[row, col].imshow(image)

        axes[row, col].axis("off")

    axes[row, 0].set_ylabel(
        insect_name,
        fontsize=10,
        rotation=0,
        labelpad=110,
        va="center"
    )

plt.subplots_adjust(left=0.23, hspace=0.5, wspace=0.1)

output_dir = Path("results/plots")
output_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(
    output_dir / "insect_samples.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print("Insect sample sheet created successfully!")