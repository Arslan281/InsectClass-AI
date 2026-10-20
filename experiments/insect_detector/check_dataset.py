
from pathlib import Path
from PIL import Image

data_folder = Path("experiments/insect_detector/data")

for category in ["insect", "non_insect"]:
    folder = data_folder / category

    valid = 0
    corrupted = []

    for path in folder.iterdir():
        if not path.is_file():
            continue

        try:
            with Image.open(path) as img:
                img.verify()
            valid += 1
        except Exception:
            corrupted.append(path.name)

    print(f"\nCategory: {category}")
    print("Valid images:", valid)
    print("Corrupted files:", len(corrupted))

    if corrupted:
        print("Problem files:", corrupted[:10])
