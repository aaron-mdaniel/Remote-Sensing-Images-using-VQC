
from pathlib import Path
import json
import random

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "2750"
OUTPUT_DIR = PROJECT_DIR / "data" / "binary_dataset"

CLASSES = ["Forest", "Highway"]
CLASS_TO_LABEL = {
    "Forest": 0,
    "Highway": 1,
}

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# --------------------------------------------------
# REPRODUCIBILITY
# --------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)

# --------------------------------------------------
# CHECK DATASET
# --------------------------------------------------

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"Dataset directory not found: {DATASET_DIR}"
    )

for class_name in CLASSES:
    class_dir = DATASET_DIR / class_name

    if not class_dir.exists():
        raise FileNotFoundError(
            f"Class directory not found: {class_dir}"
        )

# --------------------------------------------------
# COLLECT IMAGES AND LABELS
# --------------------------------------------------

records = []

for class_name in CLASSES:
    class_dir = DATASET_DIR / class_name
    label = CLASS_TO_LABEL[class_name]

    image_paths = sorted(
        path for path in class_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    print(f"{class_name}: {len(image_paths)} images")

    for image_path in image_paths:
        # Verify that the image can be opened.
        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception as error:
            print(f"Skipping invalid image: {image_path}")
            print(f"Reason: {error}")
            continue

        records.append({
            "path": str(image_path.resolve()),
            "class_name": class_name,
            "label": label,
        })

if not records:
    raise RuntimeError("No valid images were found.")

paths = np.array([item["path"] for item in records])
labels = np.array([item["label"] for item in records])
class_names = np.array([item["class_name"] for item in records])

print("\nTotal valid images:", len(records))

# --------------------------------------------------
# TRAIN / VALIDATION / TEST SPLIT
# --------------------------------------------------

# First: 70% train, 30% temporary.
train_paths, temp_paths, train_labels, temp_labels, train_names, temp_names = (
    train_test_split(
        paths,
        labels,
        class_names,
        test_size=TRAIN_RATIO * 0 + (VAL_RATIO + TEST_RATIO),
        random_state=SEED,
        stratify=labels,
    )
)

# Split the temporary 30% equally into validation and test.
relative_test_ratio = TEST_RATIO / (VAL_RATIO + TEST_RATIO)

val_paths, test_paths, val_labels, test_labels, val_names, test_names = (
    train_test_split(
        temp_paths,
        temp_labels,
        temp_names,
        test_size=relative_test_ratio,
        random_state=SEED,
        stratify=temp_labels,
    )
)

# --------------------------------------------------
# SAVE SPLIT MANIFESTS
# --------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

splits = {
    "train": (train_paths, train_labels, train_names),
    "validation": (val_paths, val_labels, val_names),
    "test": (test_paths, test_labels, test_names),
}

for split_name, (split_paths, split_labels, split_names) in splits.items():
    output_file = OUTPUT_DIR / f"{split_name}.csv"

    with output_file.open("w", encoding="utf-8", newline="") as file:
        file.write("path,class_name,label\n")

        for path, label, class_name in zip(
            split_paths, split_labels, split_names
        ):
            # Quote paths safely for CSV.
            import csv
            writer = csv.writer(file) if False else None
            file.write(
                f'"{path.replace(chr(34), chr(34) * 2)}",'
                f"{class_name},{label}\n"
            )

    print(f"Saved {split_name}: {len(split_paths)} images")

# Save configuration for reproducibility.
metadata = {
    "dataset": "EuroSAT",
    "classes": CLASS_TO_LABEL,
    "random_seed": SEED,
    "train_ratio": TRAIN_RATIO,
    "validation_ratio": VAL_RATIO,
    "test_ratio": TEST_RATIO,
    "total_valid_images": len(records),
}

with (OUTPUT_DIR / "metadata.json").open(
    "w", encoding="utf-8"
) as file:
    json.dump(metadata, file, indent=4)

# --------------------------------------------------
# REPORT CLASS DISTRIBUTIONS
# --------------------------------------------------

print("\n--- Split Verification ---")

for split_name, (_, split_labels, _) in splits.items():
    print(f"\n{split_name.upper()}: {len(split_labels)} images")

    for class_name, label in CLASS_TO_LABEL.items():
        count = int(np.sum(split_labels == label))
        print(f"  {class_name}: {count}")

print("\nDataset preparation completed.")
print("Output directory:", OUTPUT_DIR.resolve())
