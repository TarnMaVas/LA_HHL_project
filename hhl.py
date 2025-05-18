import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer, AerSimulator, QasmSimulator
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector
from qiskit.circuit.library import HamiltonianGate, RYGate, UnitaryGate
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime.fake_provider import FakeVigoV2
from qiskit.quantum_info import Pauli, partial_trace
from qiskit.visualization import plot_bloch_vector
import matplotlib.pyplot as plt
from math import pi, ceil
from qiskit.circuit.library import PauliEvolutionGate
from c_est import estimate_min, estimate_max

import numpy as np
from scipy.sparse import csr_matrix
from qiskit.synthesis.evolution import LieTrotter
from decomposition import sparse_pauli_decomposition


def process_qbit(qc: QuantumCircuit, qbit: int):

    qc.h(qbit)

    for i in range(1, qc.num_qubits - qbit):
        qc.cp(np.pi / (2**i), qbit + i, qbit)

    return qc


def swap_registers(circuit: QuantumCircuit):
    for qubit in range(circuit.num_qubits // 2):
        circuit.swap(qubit, circuit.num_qubits - qubit - 1)
    return circuit


def qft(qc: QuantumCircuit):

    swap_registers(qc)

    for i in range(qc.num_qubits):
        process_qbit(qc, i)

    return qc


def prepare_register(qc: QuantumCircuit, b: np.array, target_reg: QuantumRegister):
    b_normalized = b / np.linalg.norm(b)
    qc.initialize(b_normalized, target_reg)


def eiAt(qc, c_reg, b_reg, A, t, reps=1):
    sparse_op = sparse_pauli_decomposition(csr_matrix(A))

    for k in range(len(c_reg)):

        scaled_time = (2**k) * t
        evo_gate = PauliEvolutionGate(
            sparse_op, time=-scaled_time, synthesis=LieTrotter(reps=reps)
        )
        controlled_evo = evo_gate.control()
        qc.append(controlled_evo, [c_reg[k]] + list(b_reg))

    return qc


def reverse_eiAt(qc, c_reg, b_reg, A, t, reps=1):
    sparse_op = sparse_pauli_decomposition(csr_matrix(A))

    for k in reversed(range(len(c_reg))):

        scaled_time = (2**k) * t
        evo_gate = PauliEvolutionGate(
            sparse_op, time=scaled_time, synthesis=LieTrotter(reps=reps)
        )
        controlled_evo = evo_gate.control()
        qc.append(controlled_evo, [c_reg[k]] + list(b_reg))

    return qc


def add_controlled_rotations(circuit, c_req, a_reg):
    num_c = len(c_req)
    C = 1 / (2**num_c)

    for i in range(2**num_c):

        if not i:
            continue

        binary_repr = format(ceil(i), f"0{num_c}b")

        eigenval = i / (2**num_c)

        control_qubits = [
            c_req[num_c - i - 1] for i, bit in enumerate(binary_repr) if bit == "1"
        ]

        rotation = RYGate(2 * np.arcsin(C / eigenval)).control(len(control_qubits))

        circuit.append(rotation, control_qubits + [a_reg[0]])


def HHL_sim(A, b, draw_qc=False):
    min_eig = estimate_min(A)
    max_eig = estimate_max(A)
    n_c = min(ceil(max_eig / min_eig).bit_length(), 10)

    n_b = ceil(np.log2(len(b)))

    t = pi / max_eig

    c_reg = QuantumRegister(n_c, "c")
    b_reg = QuantumRegister(n_b, "b")
    a_reg = QuantumRegister(1, "a")
    clas = ClassicalRegister(2, "classical")

    qc = QuantumCircuit(a_reg, c_reg, b_reg, clas)

    prepare_register(qc, b, b_reg)

    qc.barrier()

    for qubit in c_reg:
        qc.h(qubit)

    qc.barrier()

    eiAt(qc, c_reg, b_reg, A, t)

    qc.barrier()

    qft_circuit = QuantumCircuit(c_reg)
    qft(qft_circuit)
    inverse_qft = qft_circuit.inverse()
    qc.compose(inverse_qft, qubits=c_reg, inplace=True)

    qc.barrier()

    add_controlled_rotations(qc, c_reg, a_reg)

    qc.barrier()

    qc.compose(qft_circuit, qubits=c_reg, inplace=True)

    qc.barrier()

    reverse_eiAt(qc, c_reg, b_reg, A, t)

    qc.barrier()

    for qubit in c_reg:
        qc.h(qubit)

    qc.barrier()

    curr_state = Statevector(qc)

    probabilities = curr_state.probabilities_dict()

    if draw_qc:
        qc.draw("mpl")

    return curr_state, probabilities


def HHL(A, b, num_shots=1024, draw_qc=False):
    # min_eig = estimate_min(A)
    # max_eig = estimate_max(A)

    evals = np.linalg.eigvals(A)
    min_eig = min(evals)
    max_eig = max(evals)

    n_c = min(ceil(max_eig / min_eig).bit_length(), 10)

    n_b = ceil(np.log2(len(b)))

    t = pi / max_eig

    c_reg = QuantumRegister(n_c, "c")
    b_reg = QuantumRegister(n_b, "b")
    a_reg = QuantumRegister(1, "a")
    clas = ClassicalRegister(n_b + 1, "classical")

    qc = QuantumCircuit(a_reg, c_reg, b_reg, clas)

    prepare_register(qc, b, b_reg)

    qc.barrier()

    for qubit in c_reg:
        qc.h(qubit)

    qc.barrier()

    eiAt(qc, c_reg, b_reg, A, t)

    qc.barrier()

    qft_circuit = QuantumCircuit(c_reg)
    qft(qft_circuit)
    inverse_qft = qft_circuit.inverse()
    qc.compose(inverse_qft, qubits=c_reg, inplace=True)

    qc.barrier()

    add_controlled_rotations(qc, c_reg, a_reg)

    qc.barrier()

    qc.compose(qft_circuit, qubits=c_reg, inplace=True)

    qc.barrier()

    reverse_eiAt(qc, c_reg, b_reg, A, t)

    qc.barrier()

    for qubit in c_reg:
        qc.h(qubit)

    qc.barrier()

    qc.measure(a_reg[0], clas[0])
    for i in range(n_b):
        qc.measure(b_reg[i], clas[i + 1])

    backend = AerSimulator(
        shots=num_shots,
        max_parallel_threads=0,
        blocking_qubits=True,
        max_parallel_experiments=0,
        max_parallel_shots=0,
    )

    transpiled_qc = transpile(qc, backend)
    result = backend.run(transpiled_qc, shots=num_shots).result()
    counts = result.get_counts()

    filtered_counts = {key[:-1]: val for key, val in counts.items() if key[-1] == "1"}

    if draw_qc:
        qc.draw("mpl")

    return counts, filtered_counts


if __name__ == "__main__":

    A = np.array([[1, -1 / 3], [-1 / 3, 1]])
    b = np.array([0, 1])

    # meow, lol = HHL_sim(A, b, True)
    # print(lol)

    _, filtered_counts = HHL(A, b, 1024)
    norm_counts = {
        k: val / min(filtered_counts.values()) for k, val in filtered_counts.items()
    }
    norm_x = [(value) ** 0.5 for _, value in sorted(norm_counts.items())]
    print("Normalized x:", norm_x)
