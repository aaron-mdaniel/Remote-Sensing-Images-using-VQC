"""
Train the existing 8-qubit / 2-layer hybrid VQC on 8 downsampled pixel values.
This is a separate experimental script; the original train_hybrid_model.py stays intact.
"""
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pennylane as qml

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "downsampled"
MODEL_DIR = PROJECT_DIR / "models" / "downsampling_vqc"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

N_QUBITS, N_LAYERS = 8, 2
BATCH_SIZE, EPOCHS, LEARNING_RATE, SEED = 16, 20, 0.001, 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def load_xy(split):
    X = np.load(DATA_DIR / f"X_{split}.npy").astype(np.float32)
    y = np.load(DATA_DIR / f"y_{split}.npy").astype(np.int64)
    if X.ndim != 2 or X.shape[1] != N_QUBITS:
        raise ValueError(f"{split}: expected (N, 8) inputs, got {X.shape}")
    if len(X) != len(y):
        raise ValueError(f"{split}: feature/label count mismatch")
    if not np.isfinite(X).all():
        raise ValueError(f"{split}: non-finite input values")
    return X, y

X_train, y_train = load_xy("train")
X_val, y_val = load_xy("validation")
# Pixel intensities are already in [0,1], so map them to rotation angles [0, pi].
X_train_angles = np.pi * X_train
X_val_angles = np.pi * X_val

train_ds = TensorDataset(torch.tensor(X_train_angles), torch.tensor(y_train))
val_ds = TensorDataset(torch.tensor(X_val_angles), torch.tensor(y_val))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

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
        self.quantum_layer = qml.qnn.TorchLayer(
            quantum_circuit, {"weights": (N_LAYERS, N_QUBITS, 2)}
        )
        self.classifier = nn.Linear(N_QUBITS, 2)
    def forward(self, x):
        return self.classifier(self.quantum_layer(x))

model = HybridQuantumClassifier().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
best_val_loss = float("inf")
history = []
checkpoint = MODEL_DIR / "best_downsampling_vqc.pth"

print(f"Device: {DEVICE}")
print(f"Train: {X_train.shape}, validation: {X_val.shape}")
print("Input: eight grayscale pixel values mapped to RY angles in [0, pi]")
print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

for epoch in range(EPOCHS):
    model.train()
    train_loss_sum = train_correct = train_count = 0
    for bx, by in train_loader:
        bx, by = bx.to(DEVICE), by.to(DEVICE)
        optimizer.zero_grad()
        logits = model(bx)
        loss = criterion(logits, by)
        loss.backward()
        optimizer.step()
        n = len(by)
        train_loss_sum += loss.item() * n
        train_correct += (logits.argmax(1) == by).sum().item()
        train_count += n

    model.eval()
    val_loss_sum = val_correct = val_count = 0
    with torch.no_grad():
        for bx, by in val_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            logits = model(bx)
            loss = criterion(logits, by)
            n = len(by)
            val_loss_sum += loss.item() * n
            val_correct += (logits.argmax(1) == by).sum().item()
            val_count += n

    row = {
        "epoch": epoch + 1,
        "train_loss": train_loss_sum / train_count,
        "train_accuracy": train_correct / train_count,
        "val_loss": val_loss_sum / val_count,
        "val_accuracy": val_correct / val_count,
    }
    history.append(row)
    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"| Train Loss: {row['train_loss']:.4f} "
        f"| Train Acc: {row['train_accuracy']:.4f} "
        f"| Val Loss: {row['val_loss']:.4f} "
        f"| Val Acc: {row['val_accuracy']:.4f}"
    )
    if row["val_loss"] < best_val_loss:
        best_val_loss = row["val_loss"]
        torch.save(model.state_dict(), checkpoint)
        print(f"  Saved best checkpoint: {checkpoint}")

pd.DataFrame(history).to_csv(MODEL_DIR / "training_history.csv", index=False)
print("\nTraining complete.")
print("Best checkpoint:", checkpoint)
print("History:", MODEL_DIR / "training_history.csv")
