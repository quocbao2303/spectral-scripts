"""Complete feature profile combining spectral and interpretable features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import NDArray

from spectral_scripts.core.confusion_matrix import ConfusionMatrix
from spectral_scripts.features.spectral import (
    SpectralFeatures,
    extract_spectral_features,
)
from spectral_scripts.features.interpretable import (
    InterpretableFeatures,
    extract_interpretable_features,
)


@dataclass(frozen=True)
class SpectralProfile:
    """
    Complete feature profile for a script's confusion matrix.

    Combines spectral features (structure-focused) with interpretable
    features (human-readable) for comprehensive analysis.

    Attributes:
        confusion: Original confusion matrix.
        spectral: Spectral features from eigendecomposition.
        interpretable: Human-interpretable features.
    """

    confusion: ConfusionMatrix
    spectral: SpectralFeatures
    interpretable: InterpretableFeatures

    @property
    def script(self) -> str:
        """Script name."""
        return self.confusion.script

    @property
    def size(self) -> int:
        """Number of characters."""
        return self.confusion.size

    def spectral_feature_vector(self) -> NDArray[np.float64]:
        """Get spectral features as vector."""
        return self.spectral.to_feature_vector()

    def interpretable_feature_vector(self) -> NDArray[np.float64]:
        """Get interpretable features as vector."""
        return self.interpretable.to_feature_vector()

    def combined_feature_vector(self) -> NDArray[np.float64]:
        """Get all features as single vector."""
        return np.concatenate([
            self.spectral.to_feature_vector(),
            self.interpretable.to_feature_vector(),
        ])

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for serialization."""
        return {
            "script": self.script,
            "size": self.size,
            "total_observations": self.confusion.total_observations,
            "spectral": self.spectral.to_dict(),
            "interpretable": self.interpretable.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict, confusion: ConfusionMatrix) -> Self:
        """Reconstruct from dictionary (requires original confusion matrix)."""
        # This would require reconstructing dataclasses from dicts
        # For now, just re-extract features
        return extract_profile(confusion)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"=== Spectral Profile: {self.script} ===",
            f"Matrix size: {self.size} × {self.size}",
            f"Total observations: {self.confusion.total_observations:,}",
            "",
            "--- Interpretable Features ---",
            f"  Accuracy: {self.interpretable.accuracy:.1%}",
            f"  Sparsity: {self.interpretable.sparsity:.1%}",
            f"  Diagonal dominance: {self.interpretable.diagonal_dominance:.2f}",
            f"  Symmetry score: {self.interpretable.symmetry_score:.3f}",
            f"  Confusion concentration (Gini): {self.interpretable.confusion_concentration:.3f}",
            f"  Top-10 confusion ratio: {self.interpretable.top_confusion_ratio:.1%}",
            "",
            "--- Spectral Features ---",
            f"  Bistochastic gap: {self.spectral.bistochastic_gap:.4f}",
            f"  Bistochastic entropy: {self.spectral.bistochastic_entropy:.3f}",
            f"  Effective rank: {self.spectral.bistochastic_effective_rank:.1f}",
            f"  Symmetric gap: {self.spectral.symmetric_gap:.4f}",
            f"  Fiedler value: {self.spectral.fiedler_value:.4f}",
            f"  Laplacian entropy: {self.spectral.laplacian_entropy:.3f}",
        ]
        return "\n".join(lines)


def extract_profile(
    confusion: ConfusionMatrix,
    max_spectrum_length: int = 50,
) -> SpectralProfile:
    """
    Extract complete feature profile from confusion matrix.

    Args:
        confusion: Input confusion matrix.
        max_spectrum_length: Maximum spectrum length for spectral features.

    Returns:
        SpectralProfile with all features.
    """
    spectral = extract_spectral_features(confusion, max_spectrum_length)
    interpretable = extract_interpretable_features(confusion)

    return SpectralProfile(
        confusion=confusion,
        spectral=spectral,
        interpretable=interpretable,
    )


def compare_profiles(
    profile1: SpectralProfile,
    profile2: SpectralProfile,
) -> dict[str, float]:
    """
    Quick comparison of two profiles.

    Returns dictionary of differences for each interpretable feature.
    Useful for debugging and understanding distance results.

    Args:
        profile1: First profile.
        profile2: Second profile.

    Returns:
        Dictionary mapping feature names to absolute differences.
    """
    i1 = profile1.interpretable
    i2 = profile2.interpretable

    return {
        "accuracy_diff": abs(i1.accuracy - i2.accuracy),
        "sparsity_diff": abs(i1.sparsity - i2.sparsity),
        "concentration_diff": abs(i1.confusion_concentration - i2.confusion_concentration),
        "diagonal_dominance_diff": abs(i1.diagonal_dominance - i2.diagonal_dominance),
        "symmetry_diff": abs(i1.symmetry_score - i2.symmetry_score),
        "entropy_diff": abs(i1.entropy_per_row - i2.entropy_per_row),
        "bistochastic_gap_diff": abs(
            profile1.spectral.bistochastic_gap - profile2.spectral.bistochastic_gap
        ),
        "fiedler_diff": abs(
            profile1.spectral.fiedler_value - profile2.spectral.fiedler_value
        ),
    }