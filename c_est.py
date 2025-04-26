import numpy as np
from scipy.sparse import  identity
from scipy.sparse.linalg import minres

def estimate_min(A, num_simulations: int = 100, shift: float = 1e-2):

    n = A.shape[0]
    I = identity(n, format='csr')
    A_shifted = A - shift * I

    b_k = np.random.rand(n)
    b_k = b_k / np.linalg.norm(b_k)

    for _ in range(num_simulations):
        b_k1, info = minres(A_shifted, b_k)
        if info != 0:
            raise RuntimeError(f"MINRES failed to converge: info={info}")
        
        b_k = b_k1 / np.linalg.norm(b_k1)

    eigenvalue_estimate = np.dot(b_k.T, A @ b_k)
    return eigenvalue_estimate

def estimate_max(A, num_simulations: int = 100):
    n = A.shape[0]
    b_k = np.random.rand(n)
    b_k = b_k / np.linalg.norm(b_k)

    for _ in range(num_simulations):
        b_k1 = A @ b_k
        b_k1_norm = np.linalg.norm(b_k1)
        b_k = b_k1 / b_k1_norm

    eigenvalue_estimate = np.dot(b_k.T, A @ b_k)
    return eigenvalue_estimate