"""Matrix normalization methods."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def row_normalize(
    matrix: NDArray[np.float64],
    handle_zero_rows: str = "uniform",
) -> NDArray[np.float64]:
    """
    Convert matrix to row-stochastic form (rows sum to 1).

    Args:
        matrix: Input matrix with non-negative entries.
        handle_zero_rows: Strategy for rows that sum to zero:
            - "uniform": Replace with uniform distribution (1/n for all entries)
            - "identity": Replace with 1 on diagonal, 0 elsewhere
            - "zero": Keep as zeros (warning: not a valid stochastic matrix)

    Returns:
        Row-stochastic matrix where each row sums to 1.

    Raises:
        ValueError: If handle_zero_rows is not a valid option.
    """
    if handle_zero_rows not in ("uniform", "identity", "zero"):
        raise ValueError(
            f"handle_zero_rows must be 'uniform', 'identity', or 'zero', "
            f"got '{handle_zero_rows}'"
        )

    matrix = matrix.astype(np.float64)
    n = matrix.shape[0]
    row_sums = matrix.sum(axis=1, keepdims=True)

    # Identify zero rows
    zero_rows = (row_sums.ravel() == 0)

    # Avoid division by zero
    row_sums = np.where(row_sums == 0, 1, row_sums)
    normalized = matrix / row_sums

    # Handle zero rows according to strategy
    if np.any(zero_rows):
        if handle_zero_rows == "uniform":
            normalized[zero_rows, :] = 1.0 / n
        elif handle_zero_rows == "identity":
            normalized[zero_rows, :] = 0.0
            normalized[zero_rows, np.where(zero_rows)[0]] = 1.0
        # "zero" case: already zeros, do nothing

    return normalized


def bistochastic_normalize(
    matrix: NDArray[np.float64],
    max_iterations: int = 1000,
    tolerance: float = 1e-10,
    regularization: float = 1e-10,
) -> NDArray[np.float64]:
    """
    Convert matrix to doubly stochastic form using Sinkhorn-Knopp algorithm.

    A doubly stochastic matrix has both rows AND columns summing to 1.
    This normalization removes both accuracy (row) and prediction bias (column)
    effects, isolating pure confusion structure.

    Args:
        matrix: Input matrix with non-negative entries.
        max_iterations: Maximum number of Sinkhorn-Knopp iterations.
        tolerance: Convergence tolerance for row/column sums.
        regularization: Small constant added to avoid division by zero.

    Returns:
        Doubly stochastic matrix where rows and columns sum to 1.

    Raises:
        ValueError: If matrix contains negative values.
        RuntimeWarning: If algorithm doesn't converge within max_iterations.
    """
    if np.any(matrix < 0):
        raise ValueError("Matrix must have non-negative entries")

    # Add small regularization to handle zeros
    A = matrix.astype(np.float64) + regularization

    for iteration in range(max_iterations):
        # Row normalization
        row_sums = A.sum(axis=1, keepdims=True)
        A = A / row_sums

        # Column normalization
        col_sums = A.sum(axis=0, keepdims=True)
        A = A / col_sums

        # Check convergence
        row_error = np.abs(A.sum(axis=1) - 1).max()
        col_error = np.abs(A.sum(axis=0) - 1).max()

        if row_error < tolerance and col_error < tolerance:
            break
    else:
        import warnings
        warnings.warn(
            f"Sinkhorn-Knopp did not converge after {max_iterations} iterations. "
            f"Row error: {row_error:.2e}, Column error: {col_error:.2e}",
            RuntimeWarning,
        )

    return A


def symmetrize(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Symmetrize a matrix: W = (A + A^T) / 2.

    This ensures all eigenvalues are real, which is required for
    proper spectral comparison.

    Args:
        matrix: Input square matrix.

    Returns:
        Symmetric matrix with real eigenvalues.
    """
    return (matrix + matrix.T) / 2


def compute_laplacian(
    matrix: NDArray[np.float64],
    normalized: bool = True,
) -> NDArray[np.float64]:
    """
    Compute the graph Laplacian from a similarity/confusion matrix.

    The Laplacian captures graph connectivity structure, with eigenvalues
    revealing clustering and community structure.

    Args:
        matrix: Symmetric similarity matrix (will be symmetrized if not).
        normalized: If True, compute normalized Laplacian L = I - D^(-1/2) W D^(-1/2).
            If False, compute unnormalized Laplacian L = D - W.

    Returns:
        Laplacian matrix with eigenvalues in [0, 2] (normalized) or [0, ∞) (unnormalized).
    """
    # Ensure symmetry
    W = symmetrize(matrix)

    # Degree matrix
    degrees = W.sum(axis=1)
    D = np.diag(degrees)

    if not normalized:
        return D - W

    # Normalized Laplacian: L = I - D^(-1/2) W D^(-1/2)
    # Handle zero degrees
    degrees_inv_sqrt = np.where(degrees > 0, 1.0 / np.sqrt(degrees), 0.0)
    D_inv_sqrt = np.diag(degrees_inv_sqrt)

    n = matrix.shape[0]
    L = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt

    return L