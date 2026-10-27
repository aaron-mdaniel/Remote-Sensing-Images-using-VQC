
import pennylane as qml
from pennylane import numpy as np


# --------------------------------------------------
# 1. QUANTUM CIRCUIT CONFIGURATION
# --------------------------------------------------

N_QUBITS = 8
N_LAYERS = 2

dev = qml.device("default.qubit", wires=N_QUBITS)


# --------------------------------------------------
# 2. VARIATIONAL QUANTUM CIRCUIT
# --------------------------------------------------

@qml.qnode(dev, interface="autograd", diff_method="backprop")
def quantum_circuit(inputs, weights):

    # Encode eight classical features into eight qubits.
    for i in range(N_QUBITS):
        qml.RY(inputs[i], wires=i)

    # Trainable variational layers.
    for layer in range(N_LAYERS):

        for i in range(N_QUBITS):
            qml.RY(weights[layer, i, 0], wires=i)
            qml.RZ(weights[layer, i, 1], wires=i)

        # Entangle neighboring qubits.
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])

    # Measure each qubit's Pauli-Z expectation value.
    return tuple(
        qml.expval(qml.PauliZ(i))
        for i in range(N_QUBITS)
    )


# --------------------------------------------------
# 3. TEST THE CIRCUIT
# --------------------------------------------------

if __name__ == "__main__":

    # Example input: eight features must be represented
    # as eight rotation angles.
    sample_input = np.array(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        requires_grad=False
    )

    # Two layers, eight qubits, two parameters per qubit.
    sample_weights = np.zeros(
        (N_LAYERS, N_QUBITS, 2),
        requires_grad=True
    )

    output = quantum_circuit(sample_input, sample_weights)

    output = np.array(output, dtype=float)

    print("=" * 50)
    print("VARIATIONAL QUANTUM CIRCUIT TEST")
    print("=" * 50)
    print("Number of qubits:", N_QUBITS)
    print("Number of variational layers:", N_LAYERS)
    print("Trainable parameters:", sample_weights.size)
    print("Output shape:", output.shape)
    print("Output values:", output)

    assert output.shape == (N_QUBITS,)
    assert np.isfinite(output).all()
    assert np.all(output >= -1.0 - 1e-6)
    assert np.all(output <= 1.0 + 1e-6)

    print("\nAll quantum circuit checks passed.")