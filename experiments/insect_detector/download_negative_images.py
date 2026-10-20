
from pathlib import Path
from io import BytesIO
from PIL import Image
from datasets import load_dataset

OUTPUT_DIR = Path(
    r"D:\InsectClass\experiments\insect_detector\data\negative_extra"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading Oxford-IIIT Pet dataset...")

dataset = load_dataset(
    "timm/oxford-iiit-pet",
    split="train",
    streaming=True
)

saved = 0

for example in dataset:
    if saved >= 100:
        break

    image = example["image"]

    if isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))

    image = image.convert("RGB")

    destination = OUTPUT_DIR / f"pet_{saved:04d}.jpg"

    image.save(destination, quality=90)

    saved += 1

    if saved % 20 == 0:
        print(f"Saved {saved}/100 photographs")

print("\nCompleted!")
print("Additional non-insect photographs:", saved)
