
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pennylane as qml
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "pca"
MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_QUBITS = 8
N_LAYERS = 2
BATCH_SIZE = 16

DEVICE = torch.device("cpu")
CLASS_NAMES = ["Forest", "Highway"]


# ============================================================
# 2. LOAD THE UNTOUCHED TEST DATA
# ============================================================

X_test = np.load(DATA_DIR / "X_test_pca.npy")
y_test = np.load(DATA_DIR / "y_test.npy")

assert X_test.shape == (825, 8), (
    f"Unexpected test feature shape: {X_test.shape}"
)
assert y_test.shape == (825,), (
    f"Unexpected test label shape: {y_test.shape}"
)
assert set(np.unique(y_test)).issubset({0, 1})

print("=" * 65)
print("FINAL TEST EVALUATION: HYBRID MODEL VS SVM")
print("=" * 65)
print("Device:", DEVICE)
print("Test features:", X_test.shape)
print("Test labels:", y_test.shape)
print("Test class counts:", np.bincount(y_test))


# ============================================================
# 3. RECREATE THE HYBRID MODEL ARCHITECTURE
#    This must match train_hybrid_model.py.
# ============================================================

quantum_device = qml.device(
    "default.qubit",
    wires=N_QUBITS
)


@qml.qnode(
    quantum_device,
    interface="torch",
    diff_method="backprop"
)
def quantum_circuit(inputs, weights):

    # Feature encoding
    for i in range(N_QUBITS):
        qml.RY(inputs[..., i], wires=i)

    # Variational layers
    for layer in range(N_LAYERS):

        for i in range(N_QUBITS):
            qml.RY(weights[layer, i, 0], wires=i)
            qml.RZ(weights[layer, i, 1], wires=i)

        # Linear entanglement
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])

    # Eight measurement outputs
    return [
        qml.expval(qml.PauliZ(i))
        for i in range(N_QUBITS)
    ]


weight_shapes = {
    "weights": (N_LAYERS, N_QUBITS, 2)
}

quantum_layer = qml.qnn.TorchLayer(
    quantum_circuit,
    weight_shapes
)


class HybridQuantumClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.quantum_layer = quantum_layer
        self.classifier = nn.Linear(N_QUBITS, 2)

    def forward(self, x):
        quantum_output = self.quantum_layer(x)
        return self.classifier(quantum_output)


# ============================================================
# 4. LOAD THE SAVED HYBRID CHECKPOINT
# ============================================================

hybrid_path = MODEL_DIR / "best_hybrid_model.pth"

if not hybrid_path.exists():
    raise FileNotFoundError(
        f"Hybrid checkpoint not found: {hybrid_path}"
    )

hybrid_model = HybridQuantumClassifier().to(DEVICE)

state_dict = torch.load(
    hybrid_path,
    map_location=DEVICE,
    weights_only=True
)

hybrid_model.load_state_dict(state_dict)
hybrid_model.eval()

print("\nHybrid checkpoint loaded successfully.")


# ============================================================
# 5. PREDICT USING THE HYBRID MODEL
# ============================================================

# Use exactly the same encoding as during training.
X_test_angles = np.pi * np.tanh(X_test / 3.0)

X_test_tensor = torch.tensor(
    X_test_angles,
    dtype=torch.float32
)

test_dataset = TensorDataset(X_test_tensor)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

hybrid_predictions = []

with torch.no_grad():
    for (batch_x,) in test_loader:
        batch_x = batch_x.to(DEVICE)

        logits = hybrid_model(batch_x)
        predictions = logits.argmax(dim=1)

        hybrid_predictions.extend(
            predictions.cpu().numpy().tolist()
        )

hybrid_predictions = np.asarray(
    hybrid_predictions,
    dtype=np.int64
)

print("Hybrid test predictions completed.")


# ============================================================
# 6. LOAD AND PREDICT USING THE SVM
# ============================================================

svm_path = MODEL_DIR / "best_svm_model.joblib"

if not svm_path.exists():
    raise FileNotFoundError(
        f"SVM model not found: {svm_path}"
    )

svm_model = joblib.load(svm_path)

# The saved pipeline applies its own StandardScaler.
# Pass the original PCA features, not the quantum angles.
svm_predictions = svm_model.predict(X_test)

print("SVM test predictions completed.")


# ============================================================
# 7. CALCULATE METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "precision_weighted": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
    }


results = {
    "Hybrid Quantum-Classical": calculate_metrics(
        y_test, hybrid_predictions
    ),
    "SVM": calculate_metrics(
        y_test, svm_predictions
    )
}


# ============================================================
# 8. DISPLAY CLASSIFICATION REPORTS AND CONFUSION MATRICES
# ============================================================

for model_name, predictions in [
    ("HYBRID QUANTUM-CLASSICAL", hybrid_predictions),
    ("SVM", svm_predictions)
]:

    print("\n" + "=" * 65)
    print(f"{model_name}: TEST RESULTS")
    print("=" * 65)

    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0
        )
    )

    print("Confusion matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1]
        )
    )


# ============================================================
# 9. COMPARE BOTH MODELS
# ============================================================

comparison = pd.DataFrame(results).T

print("\n" + "=" * 65)
print("SIDE-BY-SIDE TEST COMPARISON")
print("=" * 65)

print(comparison.round(4).to_string())


# ============================================================
# 10. SAVE THE RESULTS
# ============================================================

comparison_path = RESULTS_DIR / "test_model_comparison.csv"

comparison.to_csv(comparison_path, index=True)

predictions_path = RESULTS_DIR / "test_predictions.csv"

pd.DataFrame({
    "actual_label": y_test,
    "hybrid_prediction": hybrid_predictions,
    "svm_prediction": svm_predictions
}).to_csv(predictions_path, index=False)

print("\nResults saved:")
print(comparison_path)
print(predictions_path)

print("\nFinal test evaluation completed.")
