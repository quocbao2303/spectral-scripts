"""Spectral feature extraction from confusion matrices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from spectral_scripts.core.confusion_matrix import ConfusionMatrix
from spectral_scripts.core.normalization import (
    bistochastic_normalize,
    symmetrize,
    compute_laplacian,
)
from spectral_scripts.core.eigen import EigenResult, compute_eigen


@dataclass(frozen=True)
class SpectralFeatures:
    """
    Spectral features extracted from a confusion matrix.

    These features capture the structural properties of confusion patterns
    independent of overall accuracy.

    Attributes:
        script: Name of the script.
        matrix_size: Number of characters in the confusion matrix.

        bistochastic_spectrum: Eigenvalue magnitudes of bistochastic matrix.
        bistochastic_gap: Spectral gap (1 - |λ₂|) of bistochastic matrix.
        bistochastic_entropy: Shannon entropy of bistochastic spectrum.
        bistochastic_effective_rank: exp(entropy), effective dimensionality.

        symmetric_spectrum: Eigenvalues of symmetrized bistochastic matrix.
        symmetric_gap: Spectral gap of symmetric matrix.

        laplacian_spectrum: Eigenvalues of normalized Laplacian.
        fiedler_value: Second smallest Laplacian eigenvalue (algebraic connectivity).
        laplacian_entropy: Shannon entropy of Laplacian spectrum.
    """

    script: str
    matrix_size: int

    # Bistochastic spectrum (removes row AND column biases)
    bistochastic_spectrum: NDArray[np.float64]
    bistochastic_gap: float
    bistochastic_entropy: float
    bistochastic_effective_rank: float

    # Symmetric spectrum (guaranteed real eigenvalues)
    symmetric_spectrum: NDArray[np.float64]
    symmetric_gap: float

    # Laplacian spectrum (graph structure)
    laplacian_spectrum: NDArray[np.float64]
    fiedler_value: float
    laplacian_entropy: float

    def to_feature_vector(self) -> NDArray[np.float64]:
        """
        Convert to fixed-length feature vector for distance computation.

        Returns:
            Array of shape (6,) with key scalar features:
            [bistochastic_gap, bistochastic_entropy, bistochastic_effective_rank,
             symmetric_gap, fiedler_value, laplacian_entropy]
        """
        return np.array([
            self.bistochastic_gap,
            self.bistochastic_entropy,
            self.bistochastic_effective_rank,
            self.symmetric_gap,
            self.fiedler_value,
            self.laplacian_entropy,
        ], dtype=np.float64)

    def normalized_bistochastic_spectrum(self) -> NDArray[np.float64]:
        """Return bistochastic spectrum normalized to sum to 1."""
        total = self.bistochastic_spectrum.sum()
        if total == 0:
            return np.ones(len(self.bistochastic_spectrum)) / len(self.bistochastic_spectrum)
        return self.bistochastic_spectrum / total

    def normalized_laplacian_spectrum(self) -> NDArray[np.float64]:
        """Return Laplacian spectrum normalized to sum to 1."""
        # Laplacian eigenvalues are non-negative, use directly
        total = self.laplacian_spectrum.sum()
        if total == 0:
            return np.ones(len(self.laplacian_spectrum)) / len(self.laplacian_spectrum)
        return self.laplacian_spectrum / total

    def cumulative_bistochastic_spectrum(self) -> NDArray[np.float64]:
        """Return cumulative distribution of normalized bistochastic spectrum."""
        return np.cumsum(self.normalized_bistochastic_spectrum())

    def cumulative_laplacian_spectrum(self) -> NDArray[np.float64]:
        """Return cumulative distribution of normalized Laplacian spectrum."""
        return np.cumsum(self.normalized_laplacian_spectrum())

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for serialization."""
        return {
            "script": self.script,
            "matrix_size": self.matrix_size,
            "bistochastic_spectrum": self.bistochastic_spectrum.tolist(),
            "bistochastic_gap": self.bistochastic_gap,
            "bistochastic_entropy": self.bistochastic_entropy,
            "bistochastic_effective_rank": self.bistochastic_effective_rank,
            "symmetric_spectrum": self.symmetric_spectrum.tolist(),
            "symmetric_gap": self.symmetric_gap,
            "laplacian_spectrum": self.laplacian_spectrum.tolist(),
            "fiedler_value": self.fiedler_value,
            "laplacian_entropy": self.laplacian_entropy,
        }


def compute_spectral_entropy(eigenvalues: NDArray[np.float64]) -> float:
    """
    Compute Shannon entropy of eigenvalue distribution.

    H = -Σᵢ pᵢ log(pᵢ)
    where pᵢ = |λᵢ|² / Σⱼ|λⱼ|²

    Args:
        eigenvalues: Array of eigenvalue magnitudes.

    Returns:
        Shannon entropy (non-negative). Higher = more spread out spectrum.
    """
    # Use squared magnitudes as "energy"
    squared = eigenvalues ** 2
    total = squared.sum()

    if total == 0:
        return 0.0

    # Normalize to probabilities
    probs = squared / total

    # Filter out zeros to avoid log(0)
    probs = probs[probs > 0]

    # Shannon entropy
    entropy = -np.sum(probs * np.log(probs))

    return float(entropy)


def compute_effective_rank(entropy: float) -> float:
    """
    Compute effective rank from spectral entropy.

    r_eff = exp(H)

    Interpretation: Approximate number of "significant" eigenvalues.
    A spectrum concentrated on k eigenvalues has effective rank ≈ k.

    Args:
        entropy: Shannon entropy of spectrum.

    Returns:
        Effective rank (≥ 1).
    """
    return float(np.exp(entropy))


def extract_spectral_features(
    confusion: ConfusionMatrix,
    max_spectrum_length: int = 50,
) -> SpectralFeatures:
    """
    Extract complete spectral features from a confusion matrix.

    Pipeline:
    1. Bistochastic normalization (Sinkhorn-Knopp)
    2. Symmetrization for real spectrum
    3. Laplacian for graph structure
    4. Eigendecomposition of each
    5. Feature extraction

    Args:
        confusion: Input confusion matrix.
        max_spectrum_length: Truncate spectra to this length.

    Returns:
        SpectralFeatures with all computed values.
    """
    matrix = confusion.matrix
    n = confusion.size

    # 1. Bistochastic normalization
    bistochastic = bistochastic_normalize(matrix)
    bistochastic_eigen = compute_eigen(bistochastic, source_type="bistochastic", verify=True)

    # Truncate spectrum
    bistochastic_spectrum = bistochastic_eigen.magnitudes[:max_spectrum_length]
    bistochastic_gap = bistochastic_eigen.spectral_gap
    bistochastic_entropy = compute_spectral_entropy(bistochastic_spectrum)
    bistochastic_effective_rank = compute_effective_rank(bistochastic_entropy)

    # 2. Symmetrized bistochastic (guaranteed real eigenvalues)
    symmetric = symmetrize(bistochastic)
    symmetric_eigen = compute_eigen(symmetric, source_type="symmetric", verify=False)

    symmetric_spectrum = symmetric_eigen.magnitudes[:max_spectrum_length]
    symmetric_gap = symmetric_eigen.spectral_gap

    # 3. Laplacian of original confusion matrix
    laplacian = compute_laplacian(matrix, normalized=True)
    laplacian_eigen = compute_eigen(laplacian, source_type="laplacian", verify=True)

    # Laplacian eigenvalues are sorted ascending for Fiedler value
    laplacian_eigenvalues_sorted = np.sort(laplacian_eigen.eigenvalues.real)
    fiedler_value = float(laplacian_eigenvalues_sorted[1]) if n > 1 else 0.0

    # For spectrum comparison, use magnitudes (same as values since all non-negative)
    laplacian_spectrum = laplacian_eigen.magnitudes[:max_spectrum_length]
    laplacian_entropy = compute_spectral_entropy(laplacian_spectrum)

    return SpectralFeatures(
        script=confusion.script,
        matrix_size=n,
        bistochastic_spectrum=bistochastic_spectrum,
        bistochastic_gap=bistochastic_gap,
        bistochastic_entropy=bistochastic_entropy,
        bistochastic_effective_rank=bistochastic_effective_rank,
        symmetric_spectrum=symmetric_spectrum,
        symmetric_gap=symmetric_gap,
        laplacian_spectrum=laplacian_spectrum,
        fiedler_value=fiedler_value,
        laplacian_entropy=laplacian_entropy,
    )