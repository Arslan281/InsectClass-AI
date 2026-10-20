
from pathlib import Path
import json
import random
from collections import defaultdict

from PIL import Image, ImageOps, ImageDraw

# -----------------------------------
# 1. CONFIGURATION
# -----------------------------------

random.seed(42)

BASE_DIR = Path(__file__).resolve().parent.parent

IMAGE_DIR = Path(
    r"D:\Datasets for ANN\ip102_v1.1\images"
)

TRAIN_FILE = BASE_DIR / "data" / "ip102_v1.1" / "train.txt"

MAPPING_FILE = (
    BASE_DIR / "data" / "processed" / "class_mapping.json"
)

OUTPUT_DIR = (
    BASE_DIR / "frontend" / "public" / "insects"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------
# 2. LOAD OUR 15 CLASS MAPPINGS
# -----------------------------------

with open(MAPPING_FILE, "r", encoding="utf-8") as f:
    class_mapping = json.load(f)

original_to_new = {
    int(info["original_id"]): int(new_id)
    for new_id, info in class_mapping.items()
}

# -----------------------------------
# 3. GROUP TRAINING IMAGES
# -----------------------------------

images_by_class = defaultdict(list)

with open(TRAIN_FILE, "r", encoding="utf-8") as f:
    for line in f:
        filename, original_id = line.strip().split()
        original_id = int(original_id)

        if original_id in original_to_new:
            new_id = original_to_new[original_id]
            images_by_class[new_id].append(filename)

# -----------------------------------
# 4. SELECT CANDIDATE PHOTOGRAPHS
# -----------------------------------

# Generate 9 candidate images for every category.
# We'll visually review them before choosing one.

THUMB_SIZE = (220, 165)
CANDIDATES = 9

for class_id in range(15):

    available = images_by_class[class_id]

    selected = random.sample(
        available,
        min(CANDIDATES, len(available))
    )

    contact_sheet = Image.new(
        "RGB",
        (THUMB_SIZE[0] * 3, (THUMB_SIZE[1] + 35) * 3),
        "#15251b"
    )

    draw = ImageDraw.Draw(contact_sheet)

    candidate_names = []

    for index, filename in enumerate(selected):

        image_path = IMAGE_DIR / filename

        with Image.open(image_path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            thumbnail = ImageOps.fit(
                img,
                THUMB_SIZE,
                method=Image.Resampling.LANCZOS
            )

        row = index // 3
        col = index % 3

        x = col * THUMB_SIZE[0]
        y = row * (THUMB_SIZE[1] + 35)

        contact_sheet.paste(thumbnail, (x, y))

        draw.text(
            (x + 10, y + THUMB_SIZE[1] + 9),
            f"Option {index + 1}",
            fill="#a8f0bf"
        )

        candidate_names.append(filename)

    contact_sheet.save(
        OUTPUT_DIR / f"class_{class_id:02d}_candidates.jpg",
        quality=90
    )

    with open(
        OUTPUT_DIR / f"class_{class_id:02d}_candidates.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(candidate_names, f, indent=2)

    print(
        f"Class {class_id:02d}: "
        f"prepared {len(selected)} candidate photographs"
    )

print("\nAll 15 contact sheets generated successfully!")
print("Output:", OUTPUT_DIR)
