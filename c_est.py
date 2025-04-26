import numpy as np
from scipy.sparse import  identity
from scipy.sparse.linalg import minres

def estimate_min(A, num_simulations: int = 100, shift: float = 1e-2):

    n = A.shape[0]
    I = identity(n, format='csr')
    A_shifted = A - shift * I

    b_cur = np.random.rand(n)
    b_cur = b_cur / np.linalg.norm(b_cur)

    for _ in range(num_simulations):
        b_next, info = minres(A_shifted, b_cur)
        if info != 0:
            raise RuntimeError(f"MINRES failed to converge: info={info}")
        
        b_cur = b_cur / np.linalg.norm(b_next)

    eigenvalue_estimate = np.dot(b_cur.T, A @ b_cur)
    return eigenvalue_estimate

def estimate_max(A, num_simulations: int = 100):
    n = A.shape[0]
    b_cur = np.random.rand(n)
    b_cur = b_cur / np.linalg.norm(b_cur)

    for _ in range(num_simulations):
        b_next = A @ b_cur
        b_next_norm = np.linalg.norm(b_next)
        b_cur = b_next / b_next_norm

    eigenvalue_estimate = np.dot(b_cur.T, A @ b_cur)
    return eigenvalue_estimate