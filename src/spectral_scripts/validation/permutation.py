"""Permutation tests for distance significance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from numpy.typing import NDArray

from spectral_scripts.core.confusion_matrix import ConfusionMatrix
from spectral_scripts.features.profile import SpectralProfile, extract_profile


@dataclass(frozen=True)
class PermutationResult:
    """
    Result of permutation test.

    Attributes:
        observed: Observed test statistic.
        null_distribution: Distribution under null hypothesis.
        p_value: P-value (fraction of null ≥ observed, or ≤ for lower-tail).
        alternative: Alternative hypothesis direction.
        n_permutations: Number of permutations performed.
        significant: Whether result is significant at α=0.05.
    """

    observed: float
    null_distribution: NDArray[np.float64]
    p_value: float
    alternative: Literal["greater", "less", "two-sided"]
    n_permutations: int
    significant: bool

    @property
    def null_mean(self) -> float:
        """Mean of null distribution."""
        return float(np.mean(self.null_distribution))

    @property
    def null_std(self) -> float:
        """Std of null distribution."""
        return float(np.std(self.null_distribution))

    @property
    def effect_size(self) -> float:
        """Cohen's d effect size."""
        if self.null_std == 0:
            return float("inf") if self.observed != self.null_mean else 0.0
        return (self.observed - self.null_mean) / self.null_std

    def summary(self) -> str:
        """Generate human-readable summary."""
        sig_str = "✓ significant" if self.significant else "not significant"
        return (
            f"Observed: {self.observed:.4f}, "
            f"Null: {self.null_mean:.4f} ± {self.null_std:.4f}, "
            f"p = {self.p_value:.4f} ({sig_str})"
        )


def permute_confusion_matrix(
    confusion: ConfusionMatrix,
    method: Literal["rows", "full", "block"] = "rows",
    rng: np.random.Generator | None = None,
) -> ConfusionMatrix:
    """
    Generate permuted confusion matrix for null hypothesis testing.

    Args:
        confusion: Original confusion matrix.
        method: Permutation method:
            - "rows": Permute within each row (preserves row sums).
            - "full": Permute all entries (preserves total count only).
            - "block": Permute blocks (preserves some structure).
        rng: Random number generator.

    Returns:
        Permuted confusion matrix.
    """
    if rng is None:
        rng = np.random.default_rng()

    matrix = confusion.matrix.copy()
    n = matrix.shape[0]

    if method == "rows":
        # Permute within each row
        for i in range(n):
            rng.shuffle(matrix[i, :])

    elif method == "full":
        # Permute all entries
        flat = matrix.ravel()
        rng.shuffle(flat)
        matrix = flat.reshape((n, n))

    elif method == "block":
        # Permute 2x2 blocks
        block_size = 2
        n_blocks = n // block_size

        for bi in range(n_blocks):
            for bj in range(n_blocks):
                block = matrix[
                    bi * block_size : (bi + 1) * block_size,
                    bj * block_size : (bj + 1) * block_size,
                ].copy()
                rng.shuffle(block.ravel())
                matrix[
                    bi * block_size : (bi + 1) * block_size,
                    bj * block_size : (bj + 1) * block_size,
                ] = block.reshape((block_size, block_size))

    else:
        raise ValueError(f"Unknown permutation method: {method}")

    return ConfusionMatrix(
        matrix=matrix,
        script=confusion.script + "_permuted",
        characters=confusion.characters,
        metadata={**confusion.metadata, "permuted": method},
    )


def permutation_test(
    profile1: SpectralProfile,
    profile2: SpectralProfile,
    distance_fn: Callable[[SpectralProfile, SpectralProfile], float],
    n_permutations: int = 1000,
    alternative: Literal["greater", "less", "two-sided"] = "less",
    permutation_method: Literal["rows", "full", "block"] = "rows",
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> PermutationResult:
    """
    Test if observed distance is significantly different from null.

    Null hypothesis: The confusion structure is random (no meaningful pattern).
    Alternative (default "less"): Observed distance is smaller than expected
    under random permutation (i.e., scripts are more similar than chance).

    Args:
        profile1: First spectral profile.
        profile2: Second spectral profile.
        distance_fn: Distance function to use.
        n_permutations: Number of permutations.
        alternative: Alternative hypothesis direction.
        permutation_method: How to permute matrices.
        alpha: Significance level.
        rng: Random number generator.

    Returns:
        PermutationResult with test outcome.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Observed distance
    observed = distance_fn(profile1, profile2)

    # Null distribution
    null_distances = np.zeros(n_permutations)

    for i in range(n_permutations):
        # Permute both matrices
        perm1 = permute_confusion_matrix(
            profile1.confusion, method=permutation_method, rng=rng
        )
        perm2 = permute_confusion_matrix(
            profile2.confusion, method=permutation_method, rng=rng
        )

        # Extract profiles and compute distance
        perm_profile1 = extract_profile(perm1)
        perm_profile2 = extract_profile(perm2)
        null_distances[i] = distance_fn(perm_profile1, perm_profile2)

    # Compute p-value
    if alternative == "greater":
        p_value = float(np.mean(null_distances >= observed))
    elif alternative == "less":
        p_value = float(np.mean(null_distances <= observed))
    else:  # two-sided
        p_value = 2 * min(
            float(np.mean(null_distances >= observed)),
            float(np.mean(null_distances <= observed)),
        )
        p_value = min(p_value, 1.0)

    significant = p_value < alpha

    return PermutationResult(
        observed=observed,
        null_distribution=null_distances,
        p_value=p_value,
        alternative=alternative,
        n_permutations=n_permutations,
        significant=significant,
    )


def permutation_test_matrix(
    profiles: list[SpectralProfile],
    distance_fn: Callable[[SpectralProfile, SpectralProfile], float],
    n_permutations: int = 500,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> dict[tuple[str, str], PermutationResult]:
    """
    Run permutation tests for all pairwise distances.

    Args:
        profiles: List of spectral profiles.
        distance_fn: Distance function to use.
        n_permutations: Number of permutations per test.
        alpha: Significance level.
        rng: Random number generator.

    Returns:
        Dictionary mapping (script1, script2) to PermutationResult.
    """
    if rng is None:
        rng = np.random.default_rng()

    results = {}
    n = len(profiles)

    for i in range(n):
        for j in range(i + 1, n):
            result = permutation_test(
                profiles[i],
                profiles[j],
                distance_fn,
                n_permutations=n_permutations,
                alpha=alpha,
                rng=rng,
            )
            key = (profiles[i].script, profiles[j].script)
            results[key] = result

    return results