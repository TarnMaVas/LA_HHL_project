import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector
from qiskit.circuit.library import HamiltonianGate, RYGate, UnitaryGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info import Pauli, partial_trace
from qiskit.visualization import plot_bloch_vector
import matplotlib.pyplot as plt
from math import pi, ceil
import scipy

def process_qbit(qc: QuantumCircuit, qbit: int):
    
    qc.h(qbit)

    for i in range(1, qc.num_qubits - qbit):
        qc.cp(np.pi / (2 ** i), qbit + i, qbit)
    
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

def eiAt(circuit, c_reg, b_reg, A, t):

    num_controls = len(c_reg)

    for k in range(num_controls):

        scaled_time = (2**k) * t

        h_gate = HamiltonianGate(-A, scaled_time)

        controlled_h = h_gate.control(1)
        circuit.append(controlled_h, [c_reg[k]] + list(b_reg))
    
    return circuit

def reverse_eiAt(circuit, c_reg, b_reg, A, t):
    num_controls = len(c_reg)

    for k in reversed(range(num_controls)):
        scaled_time = (2**k) * t

        h_gate = HamiltonianGate(A, scaled_time) 

        controlled_h = h_gate.control(1)

        circuit.append(controlled_h, [c_reg[k]] + list(b_reg))



def estimate_c_reg_size(A):
    eigvals = np.linalg.eigvals(A)
    min_eigh = min(abs(eigvals))
    norm_eigh = eigvals / min_eigh

    n_c = ceil(max(norm_eigh)).bit_length()
    return n_c, norm_eigh

def add_controlled_rotations(circuit, c_req, a_reg, eigvals, C):
    num_c = len(c_req)

    for eigenval in eigvals:

        binary_repr = format(ceil(eigenval), f"0{num_c}b")

        control_qubits = [c_req[num_c - i - 1] for i, bit in enumerate(binary_repr) if bit == "1"]

        rotation = RYGate(2 * np.arcsin(C / eigenval)).control(len(control_qubits))

        circuit.append(rotation, control_qubits + [a_reg[0]])


def HHL(A, b, draw_qc = False):
    n_c, norm_eigh = estimate_c_reg_size(A)
    n_b = ceil(np.log2(len(b)))

    t = 3 * pi / 4

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
    
    add_controlled_rotations(qc, c_reg, a_reg, norm_eigh, 1)

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
    
    if (draw_qc):
        qc.draw("mpl")

    return curr_state, probabilities


if __name__ == "__main__":
    A = np.array([[1, -1/3], [-1/3, 1]])
    b = np.array([0, 1])

    meow, lol = HHL(A, b, True)
    print(lol)
