
from pathlib import Path
from PIL import Image, ImageOps
import json

# -----------------------------------
# 1. PROJECT PATHS
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

SOURCE_DIR = Path(
    r"D:\Datasets for ANN\ip102_v1.1\images"
)

OUTPUT_DIR = BASE_DIR / "frontend" / "public" / "insects"
MANIFEST_FILE = OUTPUT_DIR / "galleries.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------
# 2. INSECT CATEGORIES
# -----------------------------------

INSECTS = [
    "Spotted Lanternfly",
    "Aphids",
    "Blister Beetle",
    "Corn Borer",
    "Mole Cricket",
    "Beet Armyworm",
    "Legume Blister Beetle",
    "Limacodidae",
    "Locustoidea",
    "Cicadella viridis",
    "Xylotrechus",
    "Grub",
    "Rice Water Weevil",
    "Flea Beetle",
    "Cottony Cushion Scale"
]

# -----------------------------------
# 3. LOAD EXISTING GALLERIES
# -----------------------------------

if MANIFEST_FILE.exists():
    with open(MANIFEST_FILE, "r", encoding="utf-8") as file:
        galleries = json.load(file)
else:
    galleries = {}

# -----------------------------------
# 4. DISPLAY MAIN MENU
# -----------------------------------

def show_menu():
    print("\n" + "=" * 48)
    print("      INSECTCLASS AI - PHOTO MANAGER")
    print("=" * 48)

    for class_id, name in enumerate(INSECTS):
        count = len(galleries.get(str(class_id), []))

        print(
            f"{class_id:02d} | {name:<25} | {count} photos"
        )

    print("\nType an insect ID (0-14)")
    print("Type Q to quit")


# -----------------------------------
# 5. MANAGE ONE INSECT GALLERY
# -----------------------------------

def manage_gallery(class_id):
    name = INSECTS[class_id]

    candidates_file = (
        OUTPUT_DIR / f"class_{class_id:02d}_candidates.json"
    )

    if not candidates_file.exists():
        print("Candidate file not found!")
        return

    with open(candidates_file, "r", encoding="utf-8") as file:
        candidates = json.load(file)

    existing = galleries.get(str(class_id), [])

    print(f"\nManaging: {name}")

    if existing:
        print("Current selected options:")
        print([photo["option"] for photo in existing])
    else:
        print("No photographs selected yet.")

    print("\nOpen this contact sheet to review options:")
    print(
        OUTPUT_DIR / f"class_{class_id:02d}_candidates.jpg"
    )

    selection = input(
        "\nEnter photo options, e.g. 2,5,8 (Enter to cancel): "
    ).strip()

    if not selection:
        print("No changes made.")
        return

    try:
        options = [
            int(value.strip())
            for value in selection.split(",")
        ]
    except ValueError:
        print("Invalid input! Use numbers separated by commas.")
        return

    if (
        any(option < 1 or option > len(candidates) for option in options)
        or len(set(options)) != len(options)
    ):
        print("Invalid or duplicate options. Please try again.")
        return

    # Remember captions for previously selected options.
    old_captions = {
        photo["option"]: photo.get("caption", "")
        for photo in existing
    }

    selected_photos = []

    for position, option in enumerate(options):
        filename = candidates[option - 1]
        source = SOURCE_DIR / filename

        if not source.is_file():
            print(f"Missing source image: {source}")
            print("No changes saved.")
            return

        print(f"\nPhoto {position + 1} - Option {option}")

        previous_caption = old_captions.get(option, "")

        if previous_caption:
            print(f"Current caption: {previous_caption}")

        caption = input(
            "Caption (Enter for existing/default): "
        ).strip()

        if not caption:
            caption = (
                previous_caption
                or f"{name} - gallery photograph {position + 1}"
            )

        selected_photos.append({
            "source": source,
            "option": option,
            "caption": caption
        })

    # -----------------------------------
    # 6. EXPORT PHOTOGRAPHS
    # -----------------------------------

    exported = []

    try:
        for position, photo in enumerate(selected_photos):
            destination_name = (
                f"insect_{class_id:02d}_{position + 1}.jpg"
            )

            destination = OUTPUT_DIR / destination_name

            with Image.open(photo["source"]) as image:
                image = ImageOps.exif_transpose(image)
                image = image.convert("RGB")

                image.thumbnail(
                    (1200, 900),
                    Image.Resampling.LANCZOS
                )

                image.save(
                    destination,
                    "JPEG",
                    quality=88,
                    optimize=True
                )

            exported.append({
                "src": f"/insects/{destination_name}",
                "option": photo["option"],
                "caption": photo["caption"]
            })

    except (OSError, ValueError) as error:
        print(f"Export failed: {error}")
        print("Gallery manifest was not changed.")
        return

    # -----------------------------------
    # 7. UPDATE MANIFEST
    # -----------------------------------

    updated_galleries = dict(galleries)
    updated_galleries[str(class_id)] = exported

    temporary_manifest = MANIFEST_FILE.with_suffix(".tmp")

    with open(
        temporary_manifest, "w", encoding="utf-8"
    ) as file:
        json.dump(updated_galleries, file, indent=4)

    temporary_manifest.replace(MANIFEST_FILE)

    galleries.clear()
    galleries.update(updated_galleries)

    print(f"\nSUCCESS! {name} gallery updated.")
    print(f"Photographs: {len(exported)}")
    print("Website gallery manifest updated!")

# -----------------------------------
# 8. START PHOTO MANAGER
# -----------------------------------

while True:
    show_menu()

    choice = input("\nYour choice: ").strip()

    if choice.lower() == "q":
        print("\nPhoto Manager closed.")
        break

    try:
        class_id = int(choice)
    except ValueError:
        print("Please enter a valid class ID.")
        continue

    if not 0 <= class_id < len(INSECTS):
        print("Class ID must be between 0 and 14.")
        continue

    manage_gallery(class_id)
