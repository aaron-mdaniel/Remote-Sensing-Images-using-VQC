# Downsampling experiment for Remote-Sensing-Images-using-VQC

These scripts add a separate downsampling-based experimental pipeline. They do not overwrite the existing ResNet18/PCA scripts, models, or result files.

## What changes?

Existing path:
`data/binary_dataset/*.csv` -> `extract_features.py` (ResNet18, 512 features) -> `apply_pca.py` (8 PCA components) -> VQC/SVM.

New path:
`data/binary_dataset/*.csv` -> `prepare_downsampled_data.py` (grayscale + resize to 4x2 = 8 pixels) -> VQC/SVM.

The eight normalized grayscale intensities are mapped to RY angles by `angle = pi * intensity`. This is still angle encoding; downsampling is the method used to reduce the image to eight input values. The VQC remains eight qubits and two trainable layers.

## Run in VS Code terminal

Activate your existing `.venv` first. From the project root, copy these four `.py` files into the root next to `prepare_dataset.py`.

1. Confirm `data/binary_dataset/train.csv`, `validation.csv`, and `test.csv` exist and contain valid image paths. If not, first run `python prepare_dataset.py` after confirming the dataset is at `2750/Forest` and `2750/Highway`.
2. Create downsampled features:
   `python prepare_downsampled_data.py`
3. Train VQC:
   `python train_downsampling_vqc.py`
4. Train the SVM baseline on the same eight pixels:
   `python train_downsampling_svm.py`
5. Evaluate both on the same held-out test set:
   `python evaluate_downsampling_models.py`

Outputs:
- `data/downsampled/X_train.npy`, `X_validation.npy`, `X_test.npy`
- `data/downsampled/y_train.npy`, `y_validation.npy`, `y_test.npy`
- `models/downsampling_vqc/best_downsampling_vqc.pth`
- `models/downsampling_vqc/best_downsampling_svm.joblib`
- `models/downsampling_vqc/training_history.csv`
- `results/downsampling_test_comparison.csv`
- `results/downsampling_test_predictions.csv`

## Important notes

- `PIL.Image.Resampling.LANCZOS` requires a modern Pillow version. If unavailable, upgrade Pillow or replace `Image.Resampling.LANCZOS` with `Image.LANCZOS`.
- Do not run the old `apply_pca.py` as part of the downsampling path; it is not needed.
- Do not feed downsampled pixels into the existing PCA-based checkpoint. It was trained on PCA features and is not compatible with the new inputs.
- SVM is trained on exactly the same eight downsampled pixel values as the VQC for a fair baseline.
- This is a lossy reduction to only eight grayscale values; accuracy may fall because spatial and color information are discarded. Report actual test results rather than assuming the new approach is better.
