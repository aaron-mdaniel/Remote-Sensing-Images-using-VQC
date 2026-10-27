
from pathlib import Path

# CHANGE THIS to the actual folder containing your EuroSAT dataset.
DATASET_DIR = Path(r"C:\Users\AARON\Downloads\M. Tech\AML\New_folder_20thOct\remote_sensing_qml\2750")

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"Dataset folder not found: {DATASET_DIR}\n"
        "Update DATASET_DIR with your actual dataset path."
    )

print("=" * 60)
print("EUROSAT DATASET INSPECTION")
print("=" * 60)
print("Dataset path:", DATASET_DIR)

# Show the immediate contents of the chosen directory
print("\nTop-level contents:")
for item in sorted(DATASET_DIR.iterdir()):
    kind = "DIR " if item.is_dir() else "FILE"
    print(f"  [{kind}] {item.name}")

# Look for image files in common formats
image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
image_files = [
    path for path in DATASET_DIR.rglob("*")
    if path.is_file() and path.suffix.lower() in image_extensions
]

print("\nTotal image files found:", len(image_files))

# Summarize image counts by their immediate parent folder
counts = {}
for path in image_files:
    class_name = path.parent.name
    counts[class_name] = counts.get(class_name, 0) + 1

print("\nImage counts by immediate parent folder:")
for class_name, count in sorted(counts.items()):
    print(f"  {class_name}: {count}")

print("\nFirst 10 image paths:")
for path in image_files[:10]:
    print(" ", path)

print("\nInspection completed.")
