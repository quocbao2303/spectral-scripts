"""Wasserstein distance computations for spectral comparison."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from spectral_scripts.features.spectral import SpectralFeatures


def wasserstein_1d(
    p: NDArray[np.float64],
    q: NDArray[np.float64],
    normalize: bool = True,
) -> float:
    """
    Compute 1D Wasserstein-1 (Earth Mover's) distance.

    For 1D distributions, W₁(p, q) = ∫|F_p(x) - F_q(x)|dx
    where F is the cumulative distribution function.

    For discrete distributions of same length:
    W₁ = Σᵢ |F_p(i) - F_q(i)| / n

    Args:
        p: First distribution (non-negative).
        q: Second distribution (non-negative).
        normalize: If True, normalize both to sum to 1.

    Returns:
        Wasserstein-1 distance (non-negative).

    Raises:
        ValueError: If inputs have different lengths.
    """
    if len(p) != len(q):
        raise ValueError(f"Distributions must have same length: {len(p)} vs {len(q)}")

    if len(p) == 0:
        return 0.0

    p = p.astype(np.float64)
    q = q.astype(np.float64)

    if normalize:
        p_sum, q_sum = p.sum(), q.sum()
        if p_sum > 0:
            p = p / p_sum
        if q_sum > 0:
            q = q / q_sum

    # Compute cumulative distributions
    F_p = np.cumsum(p)
    F_q = np.cumsum(q)

    # Wasserstein-1 = mean absolute difference of CDFs
    return float(np.mean(np.abs(F_p - F_q)))


def cumulative_wasserstein(
    spectrum1: NDArray[np.float64],
    spectrum2: NDArray[np.float64],
    truncate_to: int | None = None,
) -> float:
    """
    Compute Wasserstein distance between two spectra using cumulative approach.

    This method handles spectra of different lengths by:
    1. Optionally truncating to top-k eigenvalues
    2. Padding shorter spectrum with zeros
    3. Computing Wasserstein-1 distance on normalized distributions

    Args:
        spectrum1: First spectrum (eigenvalue magnitudes, sorted descending).
        spectrum2: Second spectrum (eigenvalue magnitudes, sorted descending).
        truncate_to: If provided, truncate both spectra to this length.

    Returns:
        Wasserstein-1 distance between spectra.
    """
    s1 = spectrum1.copy()
    s2 = spectrum2.copy()

    # Truncate if requested
    if truncate_to is not None:
        s1 = s1[:truncate_to]
        s2 = s2[:truncate_to]

    # Pad shorter spectrum with zeros
    max_len = max(len(s1), len(s2))
    if len(s1) < max_len:
        s1 = np.pad(s1, (0, max_len - len(s1)), mode="constant", constant_values=0)
    if len(s2) < max_len:
        s2 = np.pad(s2, (0, max_len - len(s2)), mode="constant", constant_values=0)

    return wasserstein_1d(s1, s2, normalize=True)


def spectral_distance(
    features1: SpectralFeatures,
    features2: SpectralFeatures,
    spectrum_type: Literal["bistochastic", "symmetric", "laplacian"] = "bistochastic",
    truncate_to: int = 30,
    include_scalar_features: bool = True,
    scalar_weight: float = 0.2,
) -> float:
    """
    Compute spectral distance between two feature sets.

    Combines:
    1. Wasserstein distance between spectra (weight: 1 - scalar_weight)
    2. Euclidean distance between scalar features (weight: scalar_weight)

    Args:
        features1: First spectral features.
        features2: Second spectral features.
        spectrum_type: Which spectrum to use for Wasserstein distance.
        truncate_to: Truncate spectra to this many eigenvalues.
        include_scalar_features: If True, include scalar feature distance.
        scalar_weight: Weight for scalar feature distance (0 to 1).

    Returns:
        Combined spectral distance.
    """
    # Select spectrum based on type
    if spectrum_type == "bistochastic":
        s1 = features1.bistochastic_spectrum
        s2 = features2.bistochastic_spectrum
    elif spectrum_type == "symmetric":
        s1 = features1.symmetric_spectrum
        s2 = features2.symmetric_spectrum
    elif spectrum_type == "laplacian":
        s1 = features1.laplacian_spectrum
        s2 = features2.laplacian_spectrum
    else:
        raise ValueError(f"Unknown spectrum_type: {spectrum_type}")

    # Compute Wasserstein distance
    w_dist = cumulative_wasserstein(s1, s2, truncate_to=truncate_to)

    if not include_scalar_features:
        return w_dist

    # Compute scalar feature distance
    v1 = features1.to_feature_vector()
    v2 = features2.to_feature_vector()

    # Normalize scalar features to [0, 1] range approximately
    # Using soft normalization based on typical ranges
    scalar_ranges = np.array([
        1.0,   # bistochastic_gap: [0, 1]
        5.0,   # bistochastic_entropy: [0, ~5]
        50.0,  # bistochastic_effective_rank: [1, ~50]
        1.0,   # symmetric_gap: [0, 1]
        2.0,   # fiedler_value: [0, 2] for normalized Laplacian
        5.0,   # laplacian_entropy: [0, ~5]
    ])

    normalized_diff = (v1 - v2) / scalar_ranges
    scalar_dist = float(np.linalg.norm(normalized_diff) / np.sqrt(len(v1)))

    # Combine distances
    combined = (1 - scalar_weight) * w_dist + scalar_weight * scalar_dist

    return combined


def multi_spectrum_distance(
    features1: SpectralFeatures,
    features2: SpectralFeatures,
    weights: dict[str, float] | None = None,
    truncate_to: int = 30,
) -> float:
    """
    Compute distance using multiple spectra with configurable weights.

    Default weights give equal importance to bistochastic and Laplacian spectra,
    with lower weight on symmetric spectrum (which is derived from bistochastic).

    Args:
        features1: First spectral features.
        features2: Second spectral features.
        weights: Dictionary mapping spectrum names to weights.
            Default: {"bistochastic": 0.4, "symmetric": 0.2, "laplacian": 0.4}
        truncate_to: Truncate spectra to this many eigenvalues.

    Returns:
        Weighted combination of Wasserstein distances.
    """
    if weights is None:
        weights = {
            "bistochastic": 0.4,
            "symmetric": 0.2,
            "laplacian": 0.4,
        }

    total_weight = sum(weights.values())
    if not np.isclose(total_weight, 1.0):
        # Normalize weights
        weights = {k: v / total_weight for k, v in weights.items()}

    distance = 0.0

    if "bistochastic" in weights and weights["bistochastic"] > 0:
        d = cumulative_wasserstein(
            features1.bistochastic_spectrum,
            features2.bistochastic_spectrum,
            truncate_to=truncate_to,
        )
        distance += weights["bistochastic"] * d

    if "symmetric" in weights and weights["symmetric"] > 0:
        d = cumulative_wasserstein(
            features1.symmetric_spectrum,
            features2.symmetric_spectrum,
            truncate_to=truncate_to,
        )
        distance += weights["symmetric"] * d

    if "laplacian" in weights and weights["laplacian"] > 0:
        d = cumulative_wasserstein(
            features1.laplacian_spectrum,
            features2.laplacian_spectrum,
            truncate_to=truncate_to,
        )
        distance += weights["laplacian"] * d

    return distance