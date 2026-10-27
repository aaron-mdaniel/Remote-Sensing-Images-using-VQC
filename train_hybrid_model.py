
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pennylane as qml


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "pca"
MODEL_DIR = PROJECT_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

N_QUBITS = 8
N_LAYERS = 2
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.001
SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 65)
print("HYBRID QUANTUM-CLASSICAL MODEL TRAINING")
print("=" * 65)
print("Device:", DEVICE)
print("Qubits:", N_QUBITS)
print("Variational layers:", N_LAYERS)
print("Epochs:", EPOCHS)
print("Batch size:", BATCH_SIZE)


# ============================================================
# 2. LOAD PCA DATA
# ============================================================

X_train = np.load(DATA_DIR / "X_train_pca.npy")
X_val = np.load(DATA_DIR / "X_validation_pca.npy")

y_train = np.load(DATA_DIR / "y_train.npy")
y_val = np.load(DATA_DIR / "y_validation.npy")

assert X_train.shape == (3850, 8)
assert X_val.shape == (825, 8)
assert y_train.shape == (3850,)
assert y_val.shape == (825,)

assert set(np.unique(y_train)).issubset({0, 1})
assert set(np.unique(y_val)).issubset({0, 1})

print("\nTraining data:", X_train.shape)
print("Validation data:", X_val.shape)
print("Training labels:", np.bincount(y_train))
print("Validation labels:", np.bincount(y_val))


# ============================================================
# 3. ENCODE FEATURES AS ROTATION ANGLES
# ============================================================
# PCA features can have different numerical ranges.
# Map them smoothly into the range (-pi, pi).

def encode_angles(features):
    return np.pi * np.tanh(features / 3.0)


X_train_angles = encode_angles(X_train)
X_val_angles = encode_angles(X_val)

X_train_tensor = torch.tensor(
    X_train_angles, dtype=torch.float32
)
X_val_tensor = torch.tensor(
    X_val_angles, dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train, dtype=torch.long
)
y_val_tensor = torch.tensor(
    y_val, dtype=torch.long
)

train_dataset = TensorDataset(
    X_train_tensor, y_train_tensor
)
val_dataset = TensorDataset(
    X_val_tensor, y_val_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# 4. DEFINE THE QUANTUM CIRCUIT
# ============================================================

quantum_device = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(
    quantum_device,
    interface="torch",
    diff_method="backprop"
)
def quantum_circuit(inputs, weights):

    # Encode the 8 input features.
    # The ellipsis supports both a single sample and a batch.
    for i in range(N_QUBITS):
        qml.RY(inputs[..., i], wires=i)

    # Trainable variational circuit.
    for layer in range(N_LAYERS):

        for i in range(N_QUBITS):
            qml.RY(weights[layer, i, 0], wires=i)
            qml.RZ(weights[layer, i, 1], wires=i)

        # Linear entanglement between neighbouring qubits.
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])

    # Return one expectation value per qubit.
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


# ============================================================
# 5. DEFINE THE HYBRID CLASSIFIER
# ============================================================

class HybridQuantumClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.quantum_layer = quantum_layer

        # Eight quantum measurements -> two classes
        self.classifier = nn.Linear(N_QUBITS, 2)

    def forward(self, x):
        quantum_output = self.quantum_layer(x)
        logits = self.classifier(quantum_output)
        return logits


model = HybridQuantumClassifier().to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

print("\nTrainable parameters:", sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
), flush=True)



# ============================================================
# 6. TRAINING AND VALIDATION
# ============================================================

best_val_loss = float("inf")
history = []

checkpoint_path = MODEL_DIR / "best_hybrid_model.pth"

print("\nStarting training...", flush=True)

for epoch in range(EPOCHS):

    # --------------------------
    # Training
    # --------------------------
    model.train()

    train_loss_total = 0.0
    train_correct = 0
    train_samples = 0

    for batch_x, batch_y in train_loader:

        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        loss.backward()
        optimizer.step()

        batch_size = batch_y.size(0)

        train_loss_total += loss.item() * batch_size
        predictions = outputs.argmax(dim=1)

        train_correct += (
            predictions == batch_y
        ).sum().item()

        train_samples += batch_size

    train_loss = train_loss_total / train_samples
    train_accuracy = train_correct / train_samples

    # --------------------------
    # Validation
    # --------------------------
    model.eval()

    val_loss_total = 0.0
    val_correct = 0
    val_samples = 0

    with torch.no_grad():
        for batch_x, batch_y in val_loader:

            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            batch_size = batch_y.size(0)

            val_loss_total += loss.item() * batch_size

            predictions = outputs.argmax(dim=1)

            val_correct += (
                predictions == batch_y
            ).sum().item()

            val_samples += batch_size

    val_loss = val_loss_total / val_samples
    val_accuracy = val_correct / val_samples

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"| Train Loss: {train_loss:.4f} "
        f"| Train Accuracy: {train_accuracy:.4f} "
        f"| Val Loss: {val_loss:.4f} "
        f"| Val Accuracy: {val_accuracy:.4f}",
        flush=True
    )

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy
    })

    # Save the model with the lowest validation loss.
    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            checkpoint_path
        )

        print(
            f"  Best model saved: {checkpoint_path}",
            flush=True
        )


# ============================================================
# 7. SAVE TRAINING HISTORY
# ============================================================

import pandas as pd

history_path = MODEL_DIR / "training_history.csv"

pd.DataFrame(history).to_csv(
    history_path,
    index=False
)

print("\nTraining completed.", flush=True)
print("Best validation loss:", round(best_val_loss, 4))
print("Model checkpoint:", checkpoint_path)
print("Training history:", history_path)
