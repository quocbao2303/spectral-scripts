"""Synthetic ground truth validation for distance methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from spectral_scripts.core.confusion_matrix import ConfusionMatrix
from spectral_scripts.features.profile import SpectralProfile, extract_profile


@dataclass(frozen=True)
class SyntheticMatrix:
    """
    Synthetic confusion matrix with known ground truth relationships.

    Attributes:
        confusion: The synthetic confusion matrix.
        group: Group label (e.g., "A", "B", "C").
        similarity_to_base: Known similarity to base matrix [0, 1].
    """

    confusion: ConfusionMatrix
    group: str
    similarity_to_base: float


@dataclass(frozen=True)
class SyntheticValidationResult:
    """
    Results of synthetic ground truth validation.

    Attributes:
        spearman_rho: Spearman correlation between true and computed distances.
        spearman_pvalue: P-value for Spearman correlation.
        kendall_tau: Kendall's tau correlation.
        kendall_pvalue: P-value for Kendall's tau.
        rank_preservation: Fraction of pairwise orderings preserved.
        mean_absolute_error: MAE between normalized true and computed distances.
        passed: Whether validation passed (Spearman ρ > threshold).
        threshold: Threshold used for pass/fail.
        details: Additional diagnostic information.
    """

    spearman_rho: float
    spearman_pvalue: float
    kendall_tau: float
    kendall_pvalue: float
    rank_preservation: float
    mean_absolute_error: float
    passed: bool
    threshold: float
    details: dict[str, object]

    def summary(self) -> str:
        """Generate human-readable summary."""
        status = "✓ PASSED" if self.passed else "✗ FAILED"
        lines = [
            f"=== Synthetic Validation {status} ===",
            f"Spearman ρ: {self.spearman_rho:.4f} (p={self.spearman_pvalue:.2e})",
            f"Kendall τ: {self.kendall_tau:.4f} (p={self.kendall_pvalue:.2e})",
            f"Rank preservation: {self.rank_preservation:.1%}",
            f"Mean absolute error: {self.mean_absolute_error:.4f}",
            f"Threshold: ρ > {self.threshold}",
        ]
        return "\n".join(lines)


def generate_base_confusion_matrix(
    n_chars: int = 26,
    accuracy: float = 0.85,
    sparsity: float = 0.7,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """
    Generate a realistic base confusion matrix.

    Creates a matrix with:
    - Diagonal dominance (accuracy)
    - Sparse off-diagonal (most character pairs don't confuse)
    - Clustered confusions (similar characters confuse more)

    Args:
        n_chars: Number of characters.
        accuracy: Target overall accuracy.
        sparsity: Target sparsity of off-diagonal.
        rng: Random number generator.

    Returns:
        Confusion matrix as numpy array.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Start with diagonal matrix (perfect accuracy)
    matrix = np.eye(n_chars) * 1000

    # Add off-diagonal confusions
    n_confusions = int((1 - sparsity) * n_chars * (n_chars - 1))

    # Generate confusion pairs (prefer nearby indices = similar characters)
    for _ in range(n_confusions):
        i = rng.integers(0, n_chars)
        # Prefer nearby characters
        offset = int(rng.exponential(scale=3)) + 1
        if rng.random() < 0.5:
            offset = -offset
        j = (i + offset) % n_chars

        if i != j:
            # Add some confusion counts
            count = rng.integers(10, 100)
            matrix[i, j] += count

    # Scale to achieve target accuracy
    diagonal_sum = np.trace(matrix)
    total = matrix.sum()
    current_accuracy = diagonal_sum / total

    # Adjust diagonal to hit target accuracy
    if current_accuracy < accuracy:
        scale_factor = (accuracy * total - diagonal_sum) / (diagonal_sum * (1 - accuracy))
        np.fill_diagonal(matrix, np.diag(matrix) * (1 + scale_factor))

    return matrix


def perturb_confusion_matrix(
    base: NDArray[np.float64],
    similarity: float,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """
    Create a perturbed version of a confusion matrix with known similarity.

    Higher similarity = more similar to base matrix.

    Args:
        base: Base confusion matrix.
        similarity: Target similarity to base [0, 1].
        rng: Random number generator.

    Returns:
        Perturbed confusion matrix.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = base.shape[0]

    # Generate random perturbation matrix
    noise = rng.exponential(scale=50, size=(n, n))
    noise = noise + noise.T  # Make somewhat symmetric

    # Add diagonal dominance to noise
    np.fill_diagonal(noise, np.diag(noise) * 5)

    # Blend base and noise according to similarity
    # similarity=1.0 → pure base, similarity=0.0 → pure noise
    perturbed = similarity * base + (1 - similarity) * noise

    # Ensure non-negative and reasonable counts
    perturbed = np.maximum(perturbed, 0)

    # Normalize to similar total counts as base
    perturbed = perturbed * (base.sum() / perturbed.sum())

    return perturbed


def generate_synthetic_matrices(
    n_matrices_per_group: int = 5,
    n_chars: int = 26,
    similarity_levels: dict[str, float] | None = None,
    rng: np.random.Generator | None = None,
) -> list[SyntheticMatrix]:
    """
    Generate synthetic confusion matrices with known ground truth relationships.

    Default creates three groups:
    - Group A: High similarity to base (0.9)
    - Group B: Medium similarity to base (0.5)
    - Group C: Low similarity to base (0.1)

    Within each group, matrices are similar to each other.

    Args:
        n_matrices_per_group: Number of matrices per group.
        n_chars: Number of characters in each matrix.
        similarity_levels: Dictionary mapping group names to similarity values.
        rng: Random number generator.

    Returns:
        List of SyntheticMatrix objects.
    """
    if rng is None:
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility

    if similarity_levels is None:
        similarity_levels = {
            "A": 0.9,  # Very similar to base
            "B": 0.5,  # Moderately similar
            "C": 0.1,  # Very different
        }

    # Generate base matrix
    base = generate_base_confusion_matrix(n_chars=n_chars, rng=rng)
    characters = [chr(ord("a") + i) for i in range(n_chars)]

    results = []

    for group, similarity in similarity_levels.items():
        for i in range(n_matrices_per_group):
            # Perturb from base with target similarity
            # Add small within-group variation
            within_group_noise = rng.uniform(-0.05, 0.05)
            actual_similarity = np.clip(similarity + within_group_noise, 0.05, 0.95)

            matrix = perturb_confusion_matrix(base, actual_similarity, rng=rng)

            confusion = ConfusionMatrix(
                matrix=matrix,
                script=f"synthetic_{group}_{i}",
                characters=characters,
                metadata={"group": group, "similarity": actual_similarity},
            )

            results.append(
                SyntheticMatrix(
                    confusion=confusion,
                    group=group,
                    similarity_to_base=actual_similarity,
                )
            )

    return results


def compute_ground_truth_distances(
    matrices: list[SyntheticMatrix],
) -> NDArray[np.float64]:
    """
    Compute ground truth pairwise distances based on known similarities.

    Distance = |similarity_i - similarity_j| for between-group pairs.
    Within-group distance is based on small similarity differences.

    Args:
        matrices: List of synthetic matrices with known similarities.

    Returns:
        Square distance matrix.
    """
    n = len(matrices)
    distances = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            # Ground truth distance based on similarity difference
            d = abs(matrices[i].similarity_to_base - matrices[j].similarity_to_base)
            distances[i, j] = d
            distances[j, i] = d

    return distances


def run_synthetic_validation(
    distance_fn: Callable[[SpectralProfile, SpectralProfile], float],
    n_matrices_per_group: int = 5,
    n_chars: int = 26,
    threshold: float = 0.7,
    rng: np.random.Generator | None = None,
) -> SyntheticValidationResult:
    """
    Run synthetic ground truth validation for a distance function.

    This is the PRIMARY validation criterion. A method passes if it
    can recover the known ordering of synthetic matrices.

    Args:
        distance_fn: Function taking two SpectralProfiles, returning distance.
        n_matrices_per_group: Number of matrices per similarity group.
        n_chars: Number of characters in synthetic matrices.
        threshold: Minimum Spearman ρ to pass validation.
        rng: Random number generator.

    Returns:
        SyntheticValidationResult with all metrics.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Generate synthetic matrices
    synthetic = generate_synthetic_matrices(
        n_matrices_per_group=n_matrices_per_group,
        n_chars=n_chars,
        rng=rng,
    )

    # Extract profiles
    profiles = [extract_profile(s.confusion) for s in synthetic]

    # Compute ground truth distances
    true_distances = compute_ground_truth_distances(synthetic)

    # Compute method distances
    n = len(profiles)
    computed_distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = distance_fn(profiles[i], profiles[j])
            computed_distances[i, j] = d
            computed_distances[j, i] = d

    # Extract upper triangles for correlation
    triu_idx = np.triu_indices(n, k=1)
    true_flat = true_distances[triu_idx]
    computed_flat = computed_distances[triu_idx]

    # Spearman correlation
    spearman_result = stats.spearmanr(true_flat, computed_flat)
    spearman_rho = float(spearman_result.correlation)
    spearman_pvalue = float(spearman_result.pvalue)

    # Kendall's tau
    kendall_result = stats.kendalltau(true_flat, computed_flat)
    kendall_tau = float(kendall_result.correlation)
    kendall_pvalue = float(kendall_result.pvalue)

    # Rank preservation (fraction of pairwise orderings preserved)
    n_pairs = len(true_flat)
    concordant = 0
    total_comparable = 0

    for i in range(n_pairs):
        for j in range(i + 1, n_pairs):
            true_order = np.sign(true_flat[i] - true_flat[j])
            computed_order = np.sign(computed_flat[i] - computed_flat[j])

            if true_order != 0:  # Only count if there's a true ordering
                total_comparable += 1
                if true_order == computed_order:
                    concordant += 1

    rank_preservation = concordant / max(total_comparable, 1)

    # Normalize distances for MAE
    true_norm = true_flat / (true_flat.max() + 1e-10)
    computed_norm = computed_flat / (computed_flat.max() + 1e-10)
    mae = float(np.mean(np.abs(true_norm - computed_norm)))

    # Determine pass/fail
    passed = spearman_rho >= threshold

    # Collect details
    details = {
        "n_matrices": n,
        "n_pairs": n_pairs,
        "groups": list({s.group for s in synthetic}),
        "true_distance_range": (float(true_flat.min()), float(true_flat.max())),
        "computed_distance_range": (
            float(computed_flat.min()),
            float(computed_flat.max()),
        ),
    }

    return SyntheticValidationResult(
        spearman_rho=spearman_rho,
        spearman_pvalue=spearman_pvalue,
        kendall_tau=kendall_tau,
        kendall_pvalue=kendall_pvalue,
        rank_preservation=rank_preservation,
        mean_absolute_error=mae,
        passed=passed,
        threshold=threshold,
        details=details,
    )