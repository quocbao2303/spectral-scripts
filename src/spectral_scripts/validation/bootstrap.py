"""Bootstrap confidence intervals for distance estimates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from spectral_scripts.core.confusion_matrix import ConfusionMatrix
from spectral_scripts.features.profile import SpectralProfile, extract_profile


@dataclass(frozen=True)
class BootstrapResult:
    """
    Bootstrap confidence interval result.

    Attributes:
        point_estimate: Original distance estimate.
        mean: Mean of bootstrap distribution.
        std: Standard deviation of bootstrap distribution.
        ci_lower: Lower bound of confidence interval.
        ci_upper: Upper bound of confidence interval.
        ci_level: Confidence level (e.g., 0.95).
        n_bootstrap: Number of bootstrap samples.
        bootstrap_distribution: Full bootstrap distribution.
    """

    point_estimate: float
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    ci_level: float
    n_bootstrap: int
    bootstrap_distribution: NDArray[np.float64]

    @property
    def ci_width(self) -> float:
        """Width of confidence interval."""
        return self.ci_upper - self.ci_lower

    @property
    def relative_uncertainty(self) -> float:
        """Relative uncertainty: CI width / point estimate."""
        if self.point_estimate == 0:
            return float("inf") if self.ci_width > 0 else 0.0
        return self.ci_width / self.point_estimate

    def summary(self) -> str:
        """Generate human-readable summary."""
        return (
            f"Distance: {self.point_estimate:.4f} "
            f"({self.ci_level:.0%} CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}])"
        )


def bootstrap_confusion_matrix(
    confusion: ConfusionMatrix,
    rng: np.random.Generator,
) -> ConfusionMatrix:
    """
    Generate bootstrap sample of a confusion matrix.

    Uses multinomial resampling: treats the confusion matrix as a
    distribution over (true, predicted) pairs and resamples with replacement.

    Args:
        confusion: Original confusion matrix.
        rng: Random number generator.

    Returns:
        Bootstrap resampled confusion matrix.
    """
    matrix = confusion.matrix
    total = int(matrix.sum())

    if total == 0:
        return confusion

    # Flatten to probability distribution
    probs = matrix.ravel() / total

    # Multinomial resampling
    counts = rng.multinomial(total, probs)
    new_matrix = counts.reshape(matrix.shape).astype(np.float64)

    return ConfusionMatrix(
        matrix=new_matrix,
        script=confusion.script,
        characters=confusion.characters,
        metadata={**confusion.metadata, "bootstrap": True},
    )


def bootstrap_distance(
    profile1: SpectralProfile,
    profile2: SpectralProfile,
    distance_fn: Callable[[SpectralProfile, SpectralProfile], float],
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    rng: np.random.Generator | None = None,
) -> BootstrapResult:
    """
    Compute bootstrap confidence interval for distance between two profiles.

    Resamples both confusion matrices and recomputes distance for each
    bootstrap sample.

    Args:
        profile1: First spectral profile.
        profile2: Second spectral profile.
        distance_fn: Distance function to use.
        n_bootstrap: Number of bootstrap iterations.
        ci_level: Confidence level for interval.
        rng: Random number generator.

    Returns:
        BootstrapResult with confidence interval.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Point estimate
    point_estimate = distance_fn(profile1, profile2)

    # Bootstrap distribution
    bootstrap_distances = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        # Resample both confusion matrices
        boot_confusion1 = bootstrap_confusion_matrix(profile1.confusion, rng)
        boot_confusion2 = bootstrap_confusion_matrix(profile2.confusion, rng)

        # Extract profiles from bootstrap samples
        boot_profile1 = extract_profile(boot_confusion1)
        boot_profile2 = extract_profile(boot_confusion2)

        # Compute distance
        bootstrap_distances[i] = distance_fn(boot_profile1, boot_profile2)

    # Compute statistics
    mean = float(np.mean(bootstrap_distances))
    std = float(np.std(bootstrap_distances))

    # Percentile confidence interval
    alpha = 1 - ci_level
    ci_lower = float(np.percentile(bootstrap_distances, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_distances, 100 * (1 - alpha / 2)))

    return BootstrapResult(
        point_estimate=point_estimate,
        mean=mean,
        std=std,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=ci_level,
        n_bootstrap=n_bootstrap,
        bootstrap_distribution=bootstrap_distances,
    )


@dataclass
class BootstrapDistanceMatrix:
    """
    Bootstrap results for all pairwise distances.

    Attributes:
        point_estimates: Matrix of point estimates.
        ci_lower: Matrix of CI lower bounds.
        ci_upper: Matrix of CI upper bounds.
        scripts: List of script names.
        ci_level: Confidence level.
    """

    point_estimates: NDArray[np.float64]
    ci_lower: NDArray[np.float64]
    ci_upper: NDArray[np.float64]
    scripts: list[str]
    ci_level: float

    def get_result(self, script1: str, script2: str) -> dict[str, float]:
        """Get bootstrap result for a specific pair."""
        i = self.scripts.index(script1)
        j = self.scripts.index(script2)
        return {
            "point_estimate": float(self.point_estimates[i, j]),
            "ci_lower": float(self.ci_lower[i, j]),
            "ci_upper": float(self.ci_upper[i, j]),
        }

    def significant_differences(
        self,
        reference_script: str,
    ) -> list[tuple[str, str, float, float]]:
        """
        Find script pairs with non-overlapping CIs relative to reference.

        Returns list of (script, relation, lower, upper) where relation
        is "closer" or "farther" than reference distance.
        """
        ref_idx = self.scripts.index(reference_script)
        results = []

        for i, script in enumerate(self.scripts):
            if i == ref_idx:
                continue

            ref_lower = self.ci_lower[ref_idx, i]
            ref_upper = self.ci_upper[ref_idx, i]

            for j, other in enumerate(self.scripts):
                if j <= i or j == ref_idx:
                    continue

                other_lower = self.ci_lower[ref_idx, j]
                other_upper = self.ci_upper[ref_idx, j]

                # Check for non-overlap
                if ref_upper < other_lower:
                    results.append((script, "closer than", other, ref_upper - other_lower))
                elif other_upper < ref_lower:
                    results.append((other, "closer than", script, other_upper - ref_lower))

        return results


def bootstrap_distance_matrix(
    profiles: list[SpectralProfile],
    distance_fn: Callable[[SpectralProfile, SpectralProfile], float],
    n_bootstrap: int = 500,
    ci_level: float = 0.95,
    rng: np.random.Generator | None = None,
) -> BootstrapDistanceMatrix:
    """
    Compute bootstrap CIs for all pairwise distances.

    Note: This is computationally expensive. Consider reducing n_bootstrap
    for large numbers of profiles.

    Args:
        profiles: List of spectral profiles.
        distance_fn: Distance function to use.
        n_bootstrap: Number of bootstrap iterations per pair.
        ci_level: Confidence level.
        rng: Random number generator.

    Returns:
        BootstrapDistanceMatrix with all results.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(profiles)
    scripts = [p.script for p in profiles]

    point_estimates = np.zeros((n, n))
    ci_lower = np.zeros((n, n))
    ci_upper = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            result = bootstrap_distance(
                profiles[i],
                profiles[j],
                distance_fn,
                n_bootstrap=n_bootstrap,
                ci_level=ci_level,
                rng=rng,
            )

            point_estimates[i, j] = result.point_estimate
            point_estimates[j, i] = result.point_estimate

            ci_lower[i, j] = result.ci_lower
            ci_lower[j, i] = result.ci_lower

            ci_upper[i, j] = result.ci_upper
            ci_upper[j, i] = result.ci_upper

    return BootstrapDistanceMatrix(
        point_estimates=point_estimates,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        scripts=scripts,
        ci_level=ci_level,
    )