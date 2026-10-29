"""Evaluate downsampling VQC and SVM on the same held-out test set."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pennylane as qml
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "downsampled"
MODEL_DIR = PROJECT_DIR / "models" / "downsampling_vqc"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
N_QUBITS, N_LAYERS = 8, 2
DEVICE = torch.device("cpu")
CLASS_NAMES = ["Forest", "Highway"]

X_test = np.load(DATA_DIR / "X_test.npy").astype(np.float32)
y_test = np.load(DATA_DIR / "y_test.npy").astype(np.int64)
if X_test.ndim != 2 or X_test.shape[1] != 8:
    raise ValueError(f"Expected test data shape (N,8), got {X_test.shape}")

dev = qml.device("default.qubit", wires=N_QUBITS)
@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    for i in range(N_QUBITS):
        qml.RY(inputs[..., i], wires=i)
    for layer in range(N_LAYERS):
        for i in range(N_QUBITS):
            qml.RY(weights[layer, i, 0], wires=i)
            qml.RZ(weights[layer, i, 1], wires=i)
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

class HybridQuantumClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.quantum_layer = qml.qnn.TorchLayer(quantum_circuit, {"weights": (N_LAYERS, N_QUBITS, 2)})
        self.classifier = nn.Linear(N_QUBITS, 2)
    def forward(self, x):
        return self.classifier(self.quantum_layer(x))

hybrid_path = MODEL_DIR / "best_downsampling_vqc.pth"
svm_path = MODEL_DIR / "best_downsampling_svm.joblib"
if not hybrid_path.exists():
    raise FileNotFoundError(f"Missing {hybrid_path}; run train_downsampling_vqc.py first.")
if not svm_path.exists():
    raise FileNotFoundError(f"Missing {svm_path}; run train_downsampling_svm.py first.")

hybrid = HybridQuantumClassifier().to(DEVICE)
hybrid.load_state_dict(torch.load(hybrid_path, map_location=DEVICE, weights_only=True))
hybrid.eval()
X_angles = torch.tensor(np.pi * X_test, dtype=torch.float32)
with torch.no_grad():
    chunks = []
    for start in range(0, len(X_angles), 16):
        chunks.extend(hybrid(X_angles[start:start+16]).argmax(1).cpu().numpy().tolist())
vqc_pred = np.asarray(chunks, dtype=np.int64)
svm = joblib.load(svm_path)
svm_pred = svm.predict(X_test)

def metrics(y, p):
    return {
        "accuracy": accuracy_score(y, p),
        "precision_macro": precision_score(y, p, average="macro", zero_division=0),
        "recall_macro": recall_score(y, p, average="macro", zero_division=0),
        "f1_macro": f1_score(y, p, average="macro", zero_division=0),
    }

results = {"Downsampling VQC": metrics(y_test, vqc_pred), "Downsampling SVM": metrics(y_test, svm_pred)}
comparison = pd.DataFrame(results).T
print("TEST COMPARISON\n", comparison.round(4).to_string())
for name, pred in [("Downsampling VQC", vqc_pred), ("Downsampling SVM", svm_pred)]:
    print(f"\n{name}\n", classification_report(y_test, pred, labels=[0,1], target_names=CLASS_NAMES, digits=4, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, pred, labels=[0,1]))

comparison.to_csv(RESULTS_DIR / "downsampling_test_comparison.csv")
pd.DataFrame({"actual_label": y_test, "vqc_prediction": vqc_pred, "svm_prediction": svm_pred}).to_csv(
    RESULTS_DIR / "downsampling_test_predictions.csv", index=False
)
print("\nSaved results to:", RESULTS_DIR)
