
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
FEATURE_DIR = PROJECT_DIR / "data" / "features"
OUTPUT_DIR = PROJECT_DIR / "data" / "pca"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_COMPONENTS = 8
SEED = 42


# --------------------------------------------------
# 2. LOAD EXTRACTED FEATURES AND LABELS
# --------------------------------------------------

print("=" * 60)
print("PCA DIMENSIONALITY REDUCTION")
print("=" * 60)

X_train = np.load(FEATURE_DIR / "features_train.npy")
X_val = np.load(FEATURE_DIR / "features_validation.npy")
X_test = np.load(FEATURE_DIR / "features_test.npy")

y_train = np.load(FEATURE_DIR / "labels_train.npy")
y_val = np.load(FEATURE_DIR / "labels_validation.npy")
y_test = np.load(FEATURE_DIR / "labels_test.npy")

print("\nOriginal feature shapes:")
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

assert X_train.shape == (3850, 512), "Unexpected training feature shape"
assert X_val.shape == (825, 512), "Unexpected validation feature shape"
assert X_test.shape == (825, 512), "Unexpected test feature shape"


# --------------------------------------------------
# 3. STANDARDIZE FEATURES
# --------------------------------------------------
# Fit the scaler ONLY on training data.
# Then use the same scaler on validation and test data.

print("\nStandardizing features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# --------------------------------------------------
# 4. FIT PCA ON TRAINING DATA ONLY
# --------------------------------------------------

print(f"\nReducing 512 features to {N_COMPONENTS} components...")

pca = PCA(
    n_components=N_COMPONENTS,
    random_state=SEED
)

X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)


# --------------------------------------------------
# 5. DISPLAY EXPLAINED VARIANCE
# --------------------------------------------------

explained_variance = pca.explained_variance_ratio_
total_variance = explained_variance.sum()

print("\nExplained variance ratio per component:")

for i, variance in enumerate(explained_variance, start=1):
    print(f"PC{i}: {variance:.4f} ({variance * 100:.2f}%)")

print(
    f"\nTotal variance retained by 8 components: "
    f"{total_variance:.4f} ({total_variance * 100:.2f}%)"
)


# --------------------------------------------------
# 6. SAVE PCA FEATURES AND LABELS
# --------------------------------------------------

np.save(OUTPUT_DIR / "X_train_pca.npy", X_train_pca)
np.save(OUTPUT_DIR / "X_validation_pca.npy", X_val_pca)
np.save(OUTPUT_DIR / "X_test_pca.npy", X_test_pca)

# Save labels too, preserving their original ordering.
np.save(OUTPUT_DIR / "y_train.npy", y_train)
np.save(OUTPUT_DIR / "y_validation.npy", y_val)
np.save(OUTPUT_DIR / "y_test.npy", y_test)

# Save fitted preprocessing objects for later reuse.
joblib.dump(scaler, OUTPUT_DIR / "scaler.joblib")
joblib.dump(pca, OUTPUT_DIR / "pca_model.joblib")


# --------------------------------------------------
# 7. VERIFY RESULTS
# --------------------------------------------------

print("\nReduced feature shapes:")
print("Train:", X_train_pca.shape)
print("Validation:", X_val_pca.shape)
print("Test:", X_test_pca.shape)

assert X_train_pca.shape == (3850, 8)
assert X_val_pca.shape == (825, 8)
assert X_test_pca.shape == (825, 8)

assert np.isfinite(X_train_pca).all()
assert np.isfinite(X_val_pca).all()
assert np.isfinite(X_test_pca).all()

print("\nAll PCA checks passed.")
print("Saved PCA data to:", OUTPUT_DIR)
print("=" * 60)