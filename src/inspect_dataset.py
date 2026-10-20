from pathlib import Path
from collections import Counter
import pandas as pd

# Dataset files
DATA_DIR = Path("data/ip102_v1.1")
CLASSES_FILE = Path("data/Classification/classes.txt")

# Read insect class names
with open(CLASSES_FILE, "r", encoding="utf-8") as file:
    class_names = [
        line.strip().split(maxsplit=1)[1]
        for line in file
        if line.strip()
    ]

# Count images belonging to each class
def count_classes(filename):
    counts = Counter()

    with open(DATA_DIR / filename, "r") as file:
        for line in file:
            filename, label = line.strip().split()
            counts[int(label)] += 1

    return counts


train_counts = count_classes("train.txt")
val_counts = count_classes("val.txt")
test_counts = count_classes("test.txt")

# Create dataset summary
results = []

for class_id, name in enumerate(class_names):
    results.append({
        "Class ID": class_id,
        "Insect Name": name,
        "Train": train_counts[class_id],
        "Validation": val_counts[class_id],
        "Test": test_counts[class_id],
        "Total": (
            train_counts[class_id]
            + val_counts[class_id]
            + test_counts[class_id]
        )
    })

df = pd.DataFrame(results)

# Sort by number of images
df = df.sort_values("Total", ascending=False)

print("\nIP102 DATASET CLASS DISTRIBUTION\n")
print(df.to_string(index=False))

# Save results
output = Path("results/metrics")
output.mkdir(parents=True, exist_ok=True)

df.to_csv(output / "class_distribution.csv", index=False)

print("\nClass distribution saved successfully!")