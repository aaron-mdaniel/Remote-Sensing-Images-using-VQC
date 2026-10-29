"""
Create 8-pixel grayscale features from the existing train/validation/test CSV manifests.
This replaces ResNet18 feature extraction + PCA for the downsampling experiment.
It does NOT modify the original data/features or data/pca folders.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST_DIR = PROJECT_DIR / "data" / "binary_dataset"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "downsampled"
IMAGE_SIZE = (4, 2)  # PIL expects (width, height): 8 pixels total

def load_split(split_name: str, manifest_dir: Path, output_dir: Path):
    csv_path = manifest_dir / f"{split_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing split manifest: {csv_path}\n"
            "Run prepare_dataset.py first, or set --manifest-dir to the folder "
            "containing train.csv, validation.csv, and test.csv."
        )

    df = pd.read_csv(csv_path)
    required = {"path", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"{csv_path} must contain columns {sorted(required)}")

    features, labels = [], []
    for row_number, row in df.iterrows():
        image_path = Path(str(row["path"]))
        if not image_path.exists():
            raise FileNotFoundError(
                f"Image path in {csv_path} does not exist: {image_path}\n"
                "The CSV may contain paths from another computer. Regenerate "
                "the manifests by running prepare_dataset.py on this machine."
            )
        try:
            with Image.open(image_path) as img:
                # Grayscale then resize directly to 4x2 pixels.
                img = img.convert("L").resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
                values = np.asarray(img, dtype=np.float32).reshape(-1) / 255.0
        except Exception as exc:
            raise RuntimeError(f"Could not process image {image_path}") from exc

        features.append(values)
        labels.append(int(row["label"]))

        if (row_number + 1) % 500 == 0 or row_number + 1 == len(df):
            print(f"{split_name}: processed {row_number + 1}/{len(df)} images")

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"X_{split_name}.npy", X)
    np.save(output_dir / f"y_{split_name}.npy", y)

    print(f"{split_name}: X={X.shape}, y={y.shape}")
    print(f"Class counts (Forest=0, Highway=1): {np.bincount(y, minlength=2)}")
    return X, y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    print("DOWNSAMPLING PREPROCESSING")
    print(f"Manifest directory: {args.manifest_dir}")
    print(f"Output directory: {args.output_dir}")
    print("Conversion: RGB image -> grayscale -> 4x2 -> 8 values in [0, 1]")

    for split in ("train", "validation", "test"):
        X, y = load_split(split, args.manifest_dir, args.output_dir)
        assert X.ndim == 2 and X.shape[1] == 8, f"Expected 8 features, got {X.shape}"
        assert len(X) == len(y)
        assert np.isfinite(X).all()
        assert np.all((X >= 0) & (X <= 1))

    print("\nDownsampling data saved successfully.")
    print("Original data/features and data/pca have not been changed.")

if __name__ == "__main__":
    main()
