import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

import hhl

import numpy as np


def square_ratios(vec):
    return [(x * x) / (y * y) for x in vec for y in vec]


def symmetric_from_eigenvalues(eigenvalues, seed=None):
    lambdas = np.asarray(eigenvalues, dtype=float)
    n = lambdas.size

    if seed is not None:
        np.random.seed(seed)

    A = np.random.randn(n, n)
    Q, R = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]

    D = np.diag(lambdas)

    S = Q @ D @ Q.T

    S = (S + S.T) / 2
    return S


def plot_normed_distance_bar(x_labels, pairs, ord=2, title=None, rotation=45):

    if len(x_labels) != len(pairs):
        raise ValueError(
            f"Need exactly one label per pair: got {len(x_labels)} labels, {len(pairs)} pairs"
        )

    distances = []
    for v1, v2 in pairs:
        v1 = np.asarray(v1)
        v2 = np.asarray(v2)
        if v1.shape != v2.shape:
            raise ValueError(f"Shape mismatch: {v1.shape} vs {v2.shape}")
        distances.append(np.linalg.norm(v1 - v2, ord=ord))

    plt.figure(figsize=(max(6, len(x_labels) * 0.5), 4))
    plt.bar(x_labels, distances, edgecolor="black")
    plt.xlabel("Number of shots")
    plt.ylabel(f"{ord}-norm distance")
    if title:
        plt.title(title)
    plt.xticks(rotation=rotation, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    return distances


if __name__ == "__main__":

    evals = np.array(range(1, 65))
    A = symmetric_from_eigenvalues(evals, 1)

    # for row in A:
    # for el in row:
    # print(f"{el:.2f}", end="&")

    # print("\\\\")

    # b = np.array([1, 1, 1, 1, 0, 0, 2, 3])
    b = evals.copy()

    orig_sol = np.linalg.solve(A, b)
    orig_sol = orig_sol / np.linalg.norm(orig_sol)

    orig_squared_ratios = square_ratios(orig_sol)

    vector_pairs = []

    shots_list = [
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
    ][-1:]

    # for _ in range(3):
    # sol = np.linalg.solve(A, b)

    for num_shots in shots_list:
        _, filtered_counts = hhl.HHL(A, b, num_shots)
    #     norm_counts = {
    #         k: val / min(filtered_counts.values()) for k, val in filtered_counts.items()
    #     }
    #     norm_x = [(value) ** 0.5 for _, value in sorted(norm_counts.items())]
    #     print(norm_x)
