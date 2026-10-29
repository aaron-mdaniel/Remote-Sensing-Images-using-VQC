"""Train an SVM baseline on the exact same 8 downsampled pixels as the VQC."""
from pathlib import Path
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "downsampled"
MODEL_DIR = PROJECT_DIR / "models" / "downsampling_vqc"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42
CLASS_NAMES = ["Forest", "Highway"]

X_train = np.load(DATA_DIR / "X_train.npy")
X_val = np.load(DATA_DIR / "X_validation.npy")
y_train = np.load(DATA_DIR / "y_train.npy")
y_val = np.load(DATA_DIR / "y_validation.npy")
print("Train:", X_train.shape, "Validation:", X_val.shape)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(class_weight="balanced"))
])
param_grid = {
    "svm__kernel": ["linear", "rbf"],
    "svm__C": [0.1, 1, 10],
    "svm__gamma": ["scale", 0.1, 1],
}
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
search = GridSearchCV(pipeline, param_grid, scoring="f1_macro", cv=cv, n_jobs=-1, refit=True, verbose=1)
search.fit(X_train, y_train)
model = search.best_estimator_
pred = model.predict(X_val)

print("Best parameters:", search.best_params_)
print("CV macro F1:", round(search.best_score_, 4))
print("Validation accuracy:", round(accuracy_score(y_val, pred), 4))
print("Validation macro F1:", round(f1_score(y_val, pred, average="macro"), 4))
print(classification_report(y_val, pred, labels=[0, 1], target_names=CLASS_NAMES, digits=4, zero_division=0))
print("Confusion matrix:\n", confusion_matrix(y_val, pred, labels=[0, 1]))
path = MODEL_DIR / "best_downsampling_svm.joblib"
joblib.dump(model, path)
print("Saved:", path)
