"""Distance metrics for comparing confusion matrices and spectral profiles."""

from spectral_scripts.distance.wasserstein import (
    wasserstein_1d,
    cumulative_wasserstein,
    spectral_distance,
)
from spectral_scripts.distance.baselines import (
    frobenius_distance,
    accuracy_distance,
    character_overlap_distance,
    BaselineDistances,
)
from spectral_scripts.distance.matrix import (
    compute_distance_matrix,
    DistanceMatrix,
)

__all__ = [
    "wasserstein_1d",
    "cumulative_wasserstein",
    "spectral_distance",
    "frobenius_distance",
    "accuracy_distance",
    "character_overlap_distance",
    "BaselineDistances",
    "compute_distance_matrix",
    "DistanceMatrix",
]