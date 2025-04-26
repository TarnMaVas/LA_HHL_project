from itertools import product
import numpy as np
from qiskit.quantum_info import SparsePauliOp, Operator, Pauli


def pauli_labels(n):

    return [''.join(p) for p in product('IXYZ', repeat=n)]

def sparse_pauli_decomposition(H_sparse, threshold=1e-6):

    H = H_sparse.toarray()
    n = int(np.log2(H.shape[0]))

    labels = []
    coeffs = []
    for label in pauli_labels(n):
        P = Operator(Pauli(label))
        c = np.trace(P.adjoint().data @ H) / (2**n)
        if np.abs(c) > threshold:
            labels.append(label)
            coeffs.append(c)

    return SparsePauliOp.from_list(list(zip(labels, coeffs)))