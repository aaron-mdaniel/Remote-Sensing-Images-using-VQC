
from pathlib import Path

import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "pca"
MODEL_DIR = PROJECT_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
CLASS_NAMES = ["Forest", "Highway"]

# --------------------------------------------------
# 2. Load PCA-transformed data
# --------------------------------------------------

X_train = np.load(DATA_DIR / "X_train_pca.npy")
X_val = np.load(DATA_DIR / "X_validation_pca.npy")

y_train = np.load(DATA_DIR / "y_train.npy")
y_val = np.load(DATA_DIR / "y_validation.npy")

print("Training features:", X_train.shape)
print("Validation features:", X_val.shape)
print("Training labels:", y_train.shape)
print("Validation labels:", y_val.shape)

assert X_train.shape == (3850, 8)
assert X_val.shape == (825, 8)
assert y_train.shape == (3850,)
assert y_val.shape == (825,)

# --------------------------------------------------
# 3. Build the SVM pipeline
# --------------------------------------------------

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(class_weight="balanced"))
])

# --------------------------------------------------
# 4. Define hyperparameter search
# --------------------------------------------------

param_grid = {
    "svm__kernel": ["linear", "rbf"],
    "svm__C": [0.1, 1, 10],
    "svm__gamma": ["scale", 0.1, 1]
}

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=SEED
)

search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    refit=True,
    verbose=1
)

# --------------------------------------------------
# 5. Train and select the best SVM
# --------------------------------------------------

print("\nStarting SVM hyperparameter search...")

search.fit(X_train, y_train)

best_svm = search.best_estimator_

print("\nBest parameters:", search.best_params_)
print(
    "Best cross-validation macro F1:",
    round(search.best_score_, 4)
)

# --------------------------------------------------
# 6. Evaluate on validation data
# --------------------------------------------------

val_predictions = best_svm.predict(X_val)

val_accuracy = accuracy_score(y_val, val_predictions)
val_macro_f1 = f1_score(
    y_val,
    val_predictions,
    average="macro"
)

print("\n========== SVM VALIDATION RESULTS ==========")
print(f"Validation accuracy: {val_accuracy:.4f}")
print(f"Validation macro F1: {val_macro_f1:.4f}")

print("\nClassification report:")
print(
    classification_report(
        y_val,
        val_predictions,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )
)

print("Confusion matrix:")
print(confusion_matrix(y_val, val_predictions, labels=[0, 1]))

# --------------------------------------------------
# 7. Save the trained model
# --------------------------------------------------

model_path = MODEL_DIR / "best_svm_model.joblib"
joblib.dump(best_svm, model_path)

print(f"\nSVM model saved to: {model_path}")
print("\nSVM baseline training completed.")

