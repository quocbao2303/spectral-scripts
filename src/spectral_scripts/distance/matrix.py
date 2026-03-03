"""Pairwise distance matrix computation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Sequence

import numpy as np
from numpy.typing import NDArray

from spectral_scripts.core.confusion_matrix import ConfusionMatrix
from spectral_scripts.features.profile import SpectralProfile, extract_profile
from spectral_scripts.distance.wasserstein import spectral_distance, multi_spectrum_distance
from spectral_scripts.distance.baselines import frobenius_distance  # ADD THIS LINE
from spectral_scripts.distance.baselines import compute_baseline_distances


@dataclass
class DistanceMatrix:
    """
    Container for pairwise distance matrix with metadata.

    Attributes:
        distances: Square distance matrix where distances[i,j] = d(script_i, script_j).
        scripts: List of script names corresponding to matrix indices.
        method: Name of distance method used.
        parameters: Dictionary of method parameters.
    """

    distances: NDArray[np.float64]
    scripts: list[str]
    method: str
    parameters: dict[str, object]

    def __post_init__(self) -> None:
        """Validate matrix properties."""
        n = len(self.scripts)
        if self.distances.shape != (n, n):
            raise ValueError(
                f"Distance matrix shape {self.distances.shape} doesn't match "
                f"number of scripts ({n})"
            )

    @property
    def n_scripts(self) -> int:
        """Number of scripts in the matrix."""
        return len(self.scripts)

    def is_metric(self, tolerance: float = 1e-10) -> dict[str, bool]:
        """
        Check if distance matrix satisfies metric properties.

        Returns:
            Dictionary with boolean values for each property:
            - non_negative: All distances ≥ 0
            - identity: d(x, x) = 0 for all x
            - symmetric: d(x, y) = d(y, x) for all x, y
            - triangle: d(x, z) ≤ d(x, y) + d(y, z) for all x, y, z
        """
        D = self.distances
        n = self.n_scripts

        # Non-negativity
        non_negative = bool(np.all(D >= -tolerance))

        # Identity of indiscernibles (diagonal = 0)
        identity = bool(np.allclose(np.diag(D), 0, atol=tolerance))

        # Symmetry
        symmetric = bool(np.allclose(D, D.T, atol=tolerance))

        # Triangle inequality
        triangle = True
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if D[i, k] > D[i, j] + D[j, k] + tolerance:
                        triangle = False
                        break
                if not triangle:
                    break
            if not triangle:
                break

        return {
            "non_negative": non_negative,
            "identity": identity,
            "symmetric": symmetric,
            "triangle": triangle,
        }

    def get_distance(self, script1: str, script2: str) -> float:
        """Get distance between two scripts by name."""
        if script1 not in self.scripts:
            raise ValueError(f"Script '{script1}' not in matrix")
        if script2 not in self.scripts:
            raise ValueError(f"Script '{script2}' not in matrix")

        i = self.scripts.index(script1)
        j = self.scripts.index(script2)
        return float(self.distances[i, j])

    def rank_by_distance(self, script: str) -> list[tuple[str, float]]:
        """
        Rank all other scripts by distance to given script.

        Args:
            script: Reference script name.

        Returns:
            List of (script_name, distance) tuples, sorted by distance ascending.
        """
        if script not in self.scripts:
            raise ValueError(f"Script '{script}' not in matrix")

        i = self.scripts.index(script)
        distances = self.distances[i, :]

        ranked = [
            (self.scripts[j], float(distances[j]))
            for j in range(self.n_scripts)
            if j != i
        ]
        return sorted(ranked, key=lambda x: x[1])

    def to_condensed(self) -> NDArray[np.float64]:
        """
        Convert to condensed form (upper triangle) for scipy.

        Returns:
            1D array of length n*(n-1)/2 containing upper triangle.
        """
        from scipy.spatial.distance import squareform
        return squareform(self.distances)

    def save(self, path: str | Path) -> None:
        """Save distance matrix to .npz file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            path,
            distances=self.distances,
            scripts=np.array(self.scripts, dtype=object),
            method=self.method,
            parameters=self.parameters,
        )

    @classmethod
    def load(cls, path: str | Path) -> "DistanceMatrix":
        """Load distance matrix from .npz file."""
        data = np.load(path, allow_pickle=True)
        return cls(
            distances=data["distances"],
            scripts=list(data["scripts"]),
            method=str(data["method"]),
            parameters=data["parameters"].item(),
        )

    def summary(self) -> str:
        """Generate human-readable summary."""
        metrics = self.is_metric()
        metric_status = "✓" if all(metrics.values()) else "✗"

        lines = [
            f"=== Distance Matrix ===",
            f"Scripts: {', '.join(self.scripts)}",
            f"Method: {self.method}",
            f"Metric properties: {metric_status}",
            "",
            "Distance matrix:",
        ]

        # Format matrix as table
        header = "        " + "".join(f"{s[:8]:>10}" for s in self.scripts)
        lines.append(header)
        for i, script in enumerate(self.scripts):
            row = f"{script[:8]:<8}" + "".join(
                f"{self.distances[i, j]:>10.4f}" for j in range(self.n_scripts)
            )
            lines.append(row)

        return "\n".join(lines)


def compute_distance_matrix(
    profiles: Sequence[SpectralProfile],
    method: Literal["spectral", "multi_spectrum", "frobenius", "custom"] = "spectral",
    custom_fn: Callable[[SpectralProfile, SpectralProfile], float] | None = None,
    **kwargs,
) -> DistanceMatrix:
    """
    Compute pairwise distance matrix for a collection of spectral profiles.

    Args:
        profiles: Sequence of SpectralProfile objects.
        method: Distance method to use:
            - "spectral": Single spectrum Wasserstein + scalar features
            - "multi_spectrum": Weighted combination of all spectra
            - "frobenius": Frobenius norm baseline (handles different-sized matrices)
            - "custom": User-provided distance function
        custom_fn: Custom distance function (required if method="custom").
        **kwargs: Additional arguments passed to distance function.

    Returns:
        DistanceMatrix containing all pairwise distances.

    Raises:
        ValueError: If method is "custom" but no custom_fn provided.
    """
    n = len(profiles)
    scripts = [p.script for p in profiles]
    distances = np.zeros((n, n), dtype=np.float64)

    # Select distance function
    if method == "spectral":
        def dist_fn(p1: SpectralProfile, p2: SpectralProfile) -> float:
            return spectral_distance(p1.spectral, p2.spectral, **kwargs)
    elif method == "multi_spectrum":
        def dist_fn(p1: SpectralProfile, p2: SpectralProfile) -> float:
            return multi_spectrum_distance(p1.spectral, p2.spectral, **kwargs)
    elif method == "frobenius":
        def dist_fn(p1: SpectralProfile, p2: SpectralProfile) -> float:
            return frobenius_distance(
                p1.confusion.matrix,
                p2.confusion.matrix,
                normalize=True,
                align=True,  # Handle different-sized matrices
            )
    elif method == "custom":
        if custom_fn is None:
            raise ValueError("custom_fn required when method='custom'")
        dist_fn = custom_fn
    else:
        raise ValueError(f"Unknown method: {method}")

    # Compute pairwise distances (only upper triangle, then mirror)
    for i in range(n):
        for j in range(i + 1, n):
            d = dist_fn(profiles[i], profiles[j])
            distances[i, j] = d
            distances[j, i] = d

    return DistanceMatrix(
        distances=distances,
        scripts=scripts,
        method=method,
        parameters=kwargs,
    )


def compute_all_distance_matrices(
    profiles: Sequence[SpectralProfile],
) -> dict[str, DistanceMatrix]:
    """
    Compute distance matrices for all methods.

    Useful for method comparison and baseline benchmarking.

    Args:
        profiles: Sequence of SpectralProfile objects.

    Returns:
        Dictionary mapping method names to DistanceMatrix objects.
    """
    results = {}

    # Spectral distance (default parameters)
    results["spectral"] = compute_distance_matrix(profiles, method="spectral")

    # Multi-spectrum distance
    results["multi_spectrum"] = compute_distance_matrix(
        profiles, method="multi_spectrum"
    )

    # Frobenius baseline
    results["frobenius"] = compute_distance_matrix(profiles, method="frobenius")

    # Accuracy baseline
    def accuracy_dist(p1: SpectralProfile, p2: SpectralProfile) -> float:
        return abs(p1.interpretable.accuracy - p2.interpretable.accuracy)

    results["accuracy"] = compute_distance_matrix(
        profiles, method="custom", custom_fn=accuracy_dist
    )

    return results