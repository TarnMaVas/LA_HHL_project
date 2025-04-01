import numpy as np
import matplotlib.pyplot as plt
from time import time
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Operator
from scipy.linalg import solve as classical_solve
from typing import Tuple, List, Dict
from hhl import HHL
from multiprocessing import Pool, cpu_count
from functools import partial


def generate_hermitian_and_solution(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    size = 2**n
    
    random_matrix = np.random.random((size, size))
    
    A = (random_matrix + random_matrix.T) / 2
    
    x = np.random.random(size)
    
    b = A @ x
    
    return A, b, x

def gradient_descent_solver(A: np.ndarray, b: np.ndarray, max_iter: int = 1000, tol: float = 1e-6) -> Tuple[np.ndarray, float]:
    start_time = time()
    
    n = len(b)
    x = np.zeros(n)
    
    for i in range(max_iter):
        r = b - A @ x
        if np.linalg.norm(r) < tol:
            break
        
        alpha = np.real(np.dot(r, r) / np.dot(r, A @ r))
        x = x + alpha * r
    
    return x, time() - start_time

def classical_solver(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, float]:
    start_time = time()
    x = classical_solve(A, b)
    return x, time() - start_time

def hhl_solver(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, float]:
    start_time = time()
    state, probabilities = HHL(A, b)
    return state, time() - start_time

def run_solver(solver_func, A: np.ndarray, b: np.ndarray, solver_name: str) -> Dict:
    try:
        x, solve_time = solver_func(A, b)
        return {
            'solver': solver_name,
            'time': solve_time,
            'success': True,
            'error': None
        }
    except Exception as e:
        return {
            'solver': solver_name,
            'time': None,
            'success': False,
            'error': str(e)
        }

def solve_for_size(n: int) -> Dict:
    print(f"Testing for n = {n} (matrix size: {2**n} × {2**n})")
    
    A, b, x_true = generate_hermitian_and_solution(n)
    
    solvers = [
        ('Classical', classical_solver),
        ('Gradient Descent', gradient_descent_solver),
        ('HHL', hhl_solver)
    ]
    
    with Pool(min(len(solvers), cpu_count())) as pool:
        results = [
            run_solver(solver_func, A, b, name)
            for name, solver_func in solvers
        ]
    
    return {res['solver']: res for res in results}

def compare_solvers(max_n: int = 10):
    ns = range(1, max_n + 1)
    all_results = {}
    
    for n in ns:
        all_results[n] = solve_for_size(n)
    
    times = {
        solver: [all_results[n][solver]['time'] if all_results[n][solver]['success'] else None 
                for n in ns]
        for solver in ['Classical', 'Gradient Descent', 'HHL']
    }
    
    plt.figure(figsize=(10, 6))
    markers = {'Classical': 'o-', 'Gradient Descent': 's-', 'HHL': '^-'}
    
    for solver, solver_times in times.items():
        valid_points = [(n, t) for n, t in zip(ns, solver_times) if t is not None]
        if valid_points:
            x, y = zip(*valid_points)
            plt.plot(x, y, markers[solver], label=solver)
    
    plt.xlabel('n (matrix size = 2^n)')
    plt.ylabel('Execution time (seconds)')
    plt.title('Comparison of Linear System Solvers')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.savefig('solver_comparison.png')
    plt.close()

    
if __name__ == "__main__":
    compare_solvers(max_n=4) 