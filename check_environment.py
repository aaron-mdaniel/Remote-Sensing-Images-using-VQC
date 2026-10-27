import torch
import torchvision
import pennylane as qml
import sklearn

print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("PennyLane:", qml.__version__)
print("Scikit-learn:", sklearn.__version__)

# Test a small simulated quantum circuit
dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def quantum_test():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0))

print("Quantum circuit output:", quantum_test())
print("Environment test completed!")