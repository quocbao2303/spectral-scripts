"""Eigenvalue decomposition utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import linalg


@dataclass(frozen=True)
class EigenResult:
    """
    Result of eigenvalue decomposition.

    Attributes:
        eigenvalues: Raw eigenvalues (may be complex for non-symmetric matrices).
        eigenvectors: Corresponding eigenvectors as columns.
        magnitudes: Eigenvalue magnitudes |λ|, sorted descending.
        source_type: Type of matrix decomposed.
    """

    eigenvalues: NDArray[np.complex128]
    eigenvectors: NDArray[np.complex128]
    magnitudes: NDArray[np.float64]
    source_type: Literal["stochastic", "bistochastic", "laplacian", "symmetric", "general"]

    @property
    def is_real(self) -> bool:
        """Check if all eigenvalues are real."""
        return np.allclose(self.eigenvalues.imag, 0, atol=1e-10)

    @property
    def spectral_gap(self) -> float:
        """
        Spectral gap: 1 - |λ₂|.

        For stochastic matrices, this measures mixing time.
        Larger gap = faster convergence to stationary distribution.
        """
        if len(self.magnitudes) < 2:
            return 0.0
        return float(1.0 - self.magnitudes[1])

    @property
    def fiedler_value(self) -> float:
        """
        Fiedler value: second smallest eigenvalue (for Laplacian).

        Measures algebraic connectivity of the graph.
        Larger value = more connected graph.
        """
        if self.source_type != "laplacian":
            raise ValueError("Fiedler value only defined for Laplacian matrices")
        sorted_real = np.sort(self.eigenvalues.real)
        if len(sorted_real) < 2:
            return 0.0
        return float(sorted_real[1])

    def top_k(self, k: int) -> NDArray[np.float64]:
        """Return top k eigenvalue magnitudes."""
        return self.magnitudes[: min(k, len(self.magnitudes))]

    def normalized_spectrum(self) -> NDArray[np.float64]:
        """
        Return spectrum normalized to sum to 1.

        This treats the spectrum as a probability distribution
        for use with Wasserstein distance.
        """
        total = self.magnitudes.sum()
        if total == 0:
            return np.ones(len(self.magnitudes)) / len(self.magnitudes)
        return self.magnitudes / total

    def cumulative_spectrum(self) -> NDArray[np.float64]:
        """
        Return cumulative distribution of normalized spectrum.

        Used for cumulative Wasserstein distance computation.
        """
        return np.cumsum(self.normalized_spectrum())


def compute_eigen(
    matrix: NDArray[np.float64],
    source_type: Literal["stochastic", "bistochastic", "laplacian", "symmetric", "general"] = "general",
    verify: bool = True,
) -> EigenResult:
    """
    Compute eigenvalue decomposition with verification.

    Uses scipy.linalg.eig for general matrices or scipy.linalg.eigh
    for symmetric matrices (more numerically stable).

    Args:
        matrix: Square matrix to decompose.
        source_type: Type of matrix, used for verification and interpretation.
        verify: If True, verify expected properties based on source_type.

    Returns:
        EigenResult with eigenvalues sorted by magnitude (descending).

    Raises:
        ValueError: If verification fails.
    """
    n = matrix.shape[0]

    # Check if matrix is symmetric
    is_symmetric = np.allclose(matrix, matrix.T, atol=1e-10)

    if is_symmetric or source_type in ("laplacian", "symmetric"):
        # Use eigh for symmetric matrices (more stable, guaranteed real)
        eigenvalues_real, eigenvectors = linalg.eigh(matrix)
        eigenvalues = eigenvalues_real.astype(np.complex128)
    else:
        # Use eig for general matrices
        eigenvalues, eigenvectors = linalg.eig(matrix)

    # Compute magnitudes and sort descending
    magnitudes = np.abs(eigenvalues)
    sort_idx = np.argsort(magnitudes)[::-1]

    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]
    magnitudes = magnitudes[sort_idx]

    # Verification
    if verify:
        _verify_eigen_properties(eigenvalues, magnitudes, source_type)

    return EigenResult(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        magnitudes=magnitudes,
        source_type=source_type,
    )


def _verify_eigen_properties(
    eigenvalues: NDArray[np.complex128],
    magnitudes: NDArray[np.float64],
    source_type: str,
) -> None:
    """Verify eigenvalue properties based on matrix type."""

    if source_type == "stochastic":
        # Largest eigenvalue should be 1
        if not np.isclose(magnitudes[0], 1.0, atol=1e-6):
            raise ValueError(
                f"Stochastic matrix should have λ₁ = 1, got {magnitudes[0]:.6f}"
            )
        # All magnitudes should be ≤ 1
        if np.any(magnitudes > 1.0 + 1e-6):
            raise ValueError(
                f"Stochastic matrix eigenvalues should have |λ| ≤ 1, "
                f"got max {magnitudes.max():.6f}"
            )

    elif source_type == "bistochastic":
        # Same as stochastic
        if not np.isclose(magnitudes[0], 1.0, atol=1e-6):
            raise ValueError(
                f"Bistochastic matrix should have λ₁ = 1, got {magnitudes[0]:.6f}"
            )

    elif source_type == "laplacian":
        # Smallest eigenvalue should be 0
        min_eigenvalue = eigenvalues.real.min()
        if not np.isclose(min_eigenvalue, 0.0, atol=1e-6):
            raise ValueError(
                f"Laplacian should have λ_min = 0, got {min_eigenvalue:.6f}"
            )
        # All eigenvalues should be non-negative
        if np.any(eigenvalues.real < -1e-6):
            raise ValueError(
                f"Laplacian eigenvalues should be non-negative, "
                f"got min {eigenvalues.real.min():.6f}"
            )