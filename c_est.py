import numpy as np
from scipy.sparse import  identity
from scipy.sparse.linalg import minres

def estimate_min(A, num_simulations: int = 100, shift: float = None):
    """Estimate smallest eigenvalue using shifted inverse iteration with MINRES.
    
    Args:
        A: Sparse symmetric matrix
        num_simulations: Number of power iterations
        shift: Optional shift parameter (automatic if None)
        tol: Tolerance for MINRES solver
        
    Returns:
        Estimated smallest eigenvalue
    """
    n = A.shape[0]
    I = identity(n, format='csr')
    
    # Automatic shift selection if not provided
    if shift is None:
        # Get rough estimate of largest eigenvalue
        max_eig = estimate_max(A, min(50, num_simulations//2))
        shift = 0.9 * max_eig  # Conservative shift
    
    A_shifted = A - shift * I
    b_cur = np.random.randn(n)  # Normal distribution better for numerical stability
    b_cur /= np.linalg.norm(b_cur)

    for _ in range(num_simulations):
        # Solve with tighter tolerance and maxiter
        b_next, info = minres(A_shifted, b_cur, maxiter=2*n)
        if info != 0:
            raise RuntimeError(f"MINRES failed to converge: info={info}")
        
        # Normalize the new vector
        b_next /= np.linalg.norm(b_next)
        
        # Ensure consistent orientation (prevent sign flips)
        if np.dot(b_next, b_cur) < 0:
            b_next = -b_next
            
        b_cur = b_next

    # Rayleigh quotient for final estimate
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