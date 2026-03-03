"""Sanity checks for distance metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from spectral_scripts.core.confusion_matrix import ConfusionMatrix
from spectral_scripts.features.profile import SpectralProfile, extract_profile
from spectral_scripts.distance.matrix import DistanceMatrix


@dataclass(frozen=True)
class SanityCheckResult:
    """
    Results of metric sanity checks.

    Attributes:
        non_negative: All distances ≥ 0.
        identity: d(x, x) = 0 for all x.
        symmetry: d(x, y) = d(y, x) for all pairs.
        triangle_inequality: d(x, z) ≤ d(x, y) + d(y, z) for all triples.
        self_minimum: d(x, x) ≤ d(x, y) for all y.
        stability: Small perturbations cause small distance changes.
        all_passed: All checks passed.
        violations: Details of any violations.
    """

    non_negative: bool
    identity: bool
    symmetry: bool
    triangle_inequality: bool
    self_minimum: bool
    stability: bool
    all_passed: bool
    violations: dict[str, list[str]]

    def summary(self) -> str:
        """Generate human-readable summary."""
        status = "✓ ALL PASSED" if self.all_passed else "✗ SOME FAILED"
        lines = [
            f"=== Sanity Checks {status} ===",
            f"  Non-negativity: {'✓' if self.non_negative else '✗'}",
            f"  Identity (d(x,x)=0): {'✓' if self.identity else '✗'}",
            f"  Symmetry: {'✓' if self.symmetry else '✗'}",
            f"  Triangle inequality: {'✓' if self.triangle_inequality else '✗'}",
            f"  Self-minimum: {'✓' if self.self_minimum else '✗'}",
            f"  Stability: {'✓' if self.stability else '✗'}",
        ]

        if self.violations:
            lines.append("\nViolations:")
            for check, issues in self.violations.items():
                for issue in issues[:3]:  # Show first 3 violations
                    lines.append(f"  [{check}] {issue}")
                if len(issues) > 3:
                    lines.append(f"  [{check}] ... and {len(issues) - 3} more")

        return "\n".join(lines)


def check_non_negativity(
    distances: DistanceMatrix,
    tolerance: float = 1e-10,
) -> tuple[bool, list[str]]:
    """Check if all distances are non-negative."""
    violations = []
    D = distances.distances

    negative_mask = D < -tolerance
    if np.any(negative_mask):
        indices = np.argwhere(negative_mask)
        for i, j in indices[:5]:
            violations.append(
                f"d({distances.scripts[i]}, {distances.scripts[j]}) = {D[i, j]:.6f} < 0"
            )

    return len(violations) == 0, violations


def check_identity(
    distances: DistanceMatrix,
    tolerance: float = 1e-10,
) -> tuple[bool, list[str]]:
    """Check if d(x, x) = 0 for all x."""
    violations = []
    D = distances.distances

    diagonal = np.diag(D)
    non_zero_diag = np.abs(diagonal) > tolerance

    if np.any(non_zero_diag):
        indices = np.where(non_zero_diag)[0]
        for i in indices[:5]:
            violations.append(
                f"d({distances.scripts[i]}, {distances.scripts[i]}) = {D[i, i]:.6f} ≠ 0"
            )

    return len(violations) == 0, violations


def check_symmetry(
    distances: DistanceMatrix,
    tolerance: float = 1e-10,
) -> tuple[bool, list[str]]:
    """Check if d(x, y) = d(y, x) for all pairs."""
    violations = []
    D = distances.distances

    asymmetry = np.abs(D - D.T)
    asymmetric_mask = asymmetry > tolerance

    if np.any(asymmetric_mask):
        indices = np.argwhere(asymmetric_mask)
        for i, j in indices[:5]:
            if i < j:  # Only report each pair once
                violations.append(
                    f"d({distances.scripts[i]}, {distances.scripts[j]}) = {D[i, j]:.6f} ≠ "
                    f"d({distances.scripts[j]}, {distances.scripts[i]}) = {D[j, i]:.6f}"
                )

    return len(violations) == 0, violations


def check_triangle_inequality(
    distances: DistanceMatrix,
    tolerance: float = 1e-10,
) -> tuple[bool, list[str]]:
    """Check if d(x, z) ≤ d(x, y) + d(y, z) for all triples."""
    violations = []
    D = distances.distances
    n = distances.n_scripts

    for i in range(n):
        for j in range(n):
            for k in range(n):
                if D[i, k] > D[i, j] + D[j, k] + tolerance:
                    violations.append(
                        f"d({distances.scripts[i]}, {distances.scripts[k]}) = {D[i, k]:.6f} > "
                        f"d({distances.scripts[i]}, {distances.scripts[j]}) + "
                        f"d({distances.scripts[j]}, {distances.scripts[k]}) = "
                        f"{D[i, j]:.6f} + {D[j, k]:.6f} = {D[i, j] + D[j, k]:.6f}"
                    )
                    if len(violations) >= 10:
                        return False, violations

    return len(violations) == 0, violations


def check_self_minimum(
    distances: DistanceMatrix,
    tolerance: float = 1e-10,
) -> tuple[bool, list[str]]:
    """Check if d(x, x) ≤ d(x, y) for all x, y."""
    violations = []
    D = distances.distances
    n = distances.n_scripts

    for i in range(n):
        min_dist = D[i, :].min()
        if D[i, i] > min_dist + tolerance:
            j = np.argmin(D[i, :])
            violations.append(
                f"d({distances.scripts[i]}, {distances.scripts[i]}) = {D[i, i]:.6f} > "
                f"d({distances.scripts[i]}, {distances.scripts[j]}) = {D[i, j]:.6f}"
            )

    return len(violations) == 0, violations


def check_stability(
    profile: SpectralProfile,
    distance_fn: Callable[[SpectralProfile, SpectralProfile], float],
    perturbation_fraction: float = 0.1,
    n_trials: int = 10,
    max_distance_ratio: float = 0.5,
    rng: np.random.Generator | None = None,
) -> tuple[bool, list[str]]:
    """
    Check if small perturbations cause proportionally small distance changes.

    A stable metric should have: d(x, x') ≤ k * perturbation_size
    where x' is a perturbed version of x.

    Args:
        profile: Original profile to perturb.
        distance_fn: Distance function to test.
        perturbation_fraction: Fraction of counts to perturb.
        n_trials: Number of perturbation trials.
        max_distance_ratio: Maximum acceptable d(x, x') / max_distance.
        rng: Random number generator.

    Returns:
        Tuple of (passed, list of violation messages).
    """
    if rng is None:
        rng = np.random.default_rng()

    violations = []
    distances = []

    for trial in range(n_trials):
        # Create perturbed version
        perturbed_confusion = profile.confusion.subsample(
            1.0 - perturbation_fraction, rng=rng
        )
        perturbed_profile = extract_profile(perturbed_confusion)

        # Compute distance
        d = distance_fn(profile, perturbed_profile)
        distances.append(d)

    # Check if distances are reasonably small
    max_observed = max(distances)
    mean_observed = np.mean(distances)

    # Heuristic: perturbation distance should be small relative to
    # typical between-script distances (we use perturbation_fraction as proxy)
    if max_observed > max_distance_ratio:
        violations.append(
            f"Max perturbation distance {max_observed:.4f} > {max_distance_ratio} "
            f"(mean: {mean_observed:.4f})"
        )

    return len(violations) == 0, violations


def run_sanity_checks(
    distances: DistanceMatrix,
    profiles: list[SpectralProfile] | None = None,
    distance_fn: Callable[[SpectralProfile, SpectralProfile], float] | None = None,
    tolerance: float = 1e-10,
) -> SanityCheckResult:
    """
    Run all sanity checks on a distance matrix.

    Args:
        distances: Distance matrix to check.
        profiles: Optional profiles for stability check.
        distance_fn: Optional distance function for stability check.
        tolerance: Numerical tolerance for comparisons.

    Returns:
        SanityCheckResult with all check outcomes.
    """
    violations: dict[str, list[str]] = {}

    # Basic metric properties
    non_neg, non_neg_violations = check_non_negativity(distances, tolerance)
    if non_neg_violations:
        violations["non_negative"] = non_neg_violations

    identity, identity_violations = check_identity(distances, tolerance)
    if identity_violations:
        violations["identity"] = identity_violations

    symmetry, symmetry_violations = check_symmetry(distances, tolerance)
    if symmetry_violations:
        violations["symmetry"] = symmetry_violations

    triangle, triangle_violations = check_triangle_inequality(distances, tolerance)
    if triangle_violations:
        violations["triangle_inequality"] = triangle_violations

    self_min, self_min_violations = check_self_minimum(distances, tolerance)
    if self_min_violations:
        violations["self_minimum"] = self_min_violations

    # Stability check (requires profiles and distance function)
    stability = True
    if profiles is not None and distance_fn is not None and len(profiles) > 0:
        stability, stability_violations = check_stability(
            profiles[0], distance_fn
        )
        if stability_violations:
            violations["stability"] = stability_violations
    else:
        stability = True  # Skip if not enough info

    all_passed = (
        non_neg and identity and symmetry and triangle and self_min and stability
    )

    return SanityCheckResult(
        non_negative=non_neg,
        identity=identity,
        symmetry=symmetry,
        triangle_inequality=triangle,
        self_minimum=self_min,
        stability=stability,
        all_passed=all_passed,
        violations=violations,
    )