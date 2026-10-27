
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image, ImageFile


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "binary_dataset"
OUTPUT_DIR = PROJECT_DIR / "data" / "features"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
NUM_WORKERS = 0  # Safe default for Windows
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# Automatically use NVIDIA GPU if PyTorch can access it.
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("RESNET18 FEATURE EXTRACTION")
print("=" * 60)
print(f"Project directory: {PROJECT_DIR}")
print(f"Device: {DEVICE}")

if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU. Feature extraction may take longer.")


# --------------------------------------------------
# 2. IMAGE PREPROCESSING
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# 3. DATASET CLASS
# --------------------------------------------------

class EuroSATDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)

        required_columns = {"path", "label"}
        if not required_columns.issubset(self.data.columns):
            raise ValueError(
                f"{csv_path} must contain columns: "
                f"{sorted(required_columns)}. "
                f"Found: {list(self.data.columns)}"
            )

        self.image_paths = self.data["path"].astype(str).tolist()
        self.labels = self.data["label"].astype(int).tolist()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = Path(self.image_paths[index])
        label = self.labels[index]

        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
                image = transform(image)
        except Exception as exc:
            raise RuntimeError(
                f"Could not read image: {image_path}"
            ) from exc

        return image, label


# --------------------------------------------------
# 4. LOAD PRETRAINED RESNET18
# --------------------------------------------------

print("\nLoading pretrained ResNet18...")

weights = models.ResNet18_Weights.DEFAULT
resnet = models.resnet18(weights=weights)

# Remove the final classification layer.
# ResNet18 now returns 512 features per image.
feature_extractor = torch.nn.Sequential(
    *list(resnet.children())[:-1]
)

feature_extractor = feature_extractor.to(DEVICE)
feature_extractor.eval()

# We are extracting features, not training ResNet18.
for parameter in feature_extractor.parameters():
    parameter.requires_grad = False

print("ResNet18 loaded successfully.")


# --------------------------------------------------
# 5. FEATURE EXTRACTION FUNCTION
# --------------------------------------------------

def extract_split(split_name):
    csv_path = DATA_DIR / f"{split_name}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing dataset file: {csv_path}"
        )

    dataset = EuroSATDataset(csv_path)

    if len(dataset) == 0:
        raise ValueError(f"{split_name} dataset is empty.")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    all_features = []
    all_labels = []

    print(f"\nExtracting {split_name} features...")
    print(f"Images: {len(dataset)}")

    with torch.inference_mode():
        for batch_index, (images, labels) in enumerate(loader):
            images = images.to(DEVICE)

            # Shape before flattening: (batch, 512, 1, 1)
            features = feature_extractor(images)

            # Shape after flattening: (batch, 512)
            features = torch.flatten(features, start_dim=1)

            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

            processed = min(
                (batch_index + 1) * BATCH_SIZE,
                len(dataset)
            )

            if (
                (batch_index + 1) % 10 == 0
                or processed == len(dataset)
            ):
                print(
                    f"Processed {processed}/{len(dataset)} images"
                )

    X = np.concatenate(all_features, axis=0).astype(np.float32)
    y = np.concatenate(all_labels, axis=0).astype(np.int64)

    # Save features and labels separately.
    np.save(OUTPUT_DIR / f"features_{split_name}.npy", X)
    np.save(OUTPUT_DIR / f"labels_{split_name}.npy", y)

    print(f"{split_name} feature shape: {X.shape}")
    print(f"{split_name} label shape: {y.shape}")
    print(f"Saved features_{split_name}.npy")
    print(f"Saved labels_{split_name}.npy")

    return X, y


# --------------------------------------------------
# 6. PROCESS TRAIN, VALIDATION, AND TEST SETS
# --------------------------------------------------

if __name__ == "__main__":
    for split in ["train", "validation", "test"]:
        extract_split(split)

    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION COMPLETED")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)
