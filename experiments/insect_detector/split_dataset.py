
from pathlib import Path
import random
import json

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"

random.seed(42)

groups = {
    "insect": list((DATA / "insect").glob("*")),
    "cifar": list((DATA / "non_insect").glob("*")),
    "extra": list((DATA / "negative_extra").glob("pet_*")),
    "screenshots": list((DATA / "negative_extra").glob("screenshot_*")),
}

def split_files(files):
    files = sorted(files)
    random.shuffle(files)

    n = len(files)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    return {
        "train": files[:train_end],
        "val": files[train_end:val_end],
        "test": files[val_end:]
    }

splits = {
    "train": [],
    "val": [],
    "test": []
}

for group, files in groups.items():
    if group == "screenshots":
        # Reserve screenshots for evaluation.
        splits["test"].extend(
            {"path": str(p), "label": 0}
            for p in sorted(files)
        )
        continue

    group_splits = split_files(files)

    label = 1 if group == "insect" else 0

    for split_name, paths in group_splits.items():
        splits[split_name].extend(
            {"path": str(p), "label": label}
            for p in paths
        )

for split_name, records in splits.items():
    random.shuffle(records)

    output = BASE / f"{split_name}_manifest.json"

    with open(output, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    insects = sum(x["label"] == 1 for x in records)
    negatives = sum(x["label"] == 0 for x in records)

    print(
        f"{split_name}: "
        f"{insects} insects, "
        f"{negatives} non-insects, "
        f"{len(records)} total"
    )

print("\nDataset manifests created!")
