"""Baseline distance methods for comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from spectral_scripts.core.confusion_matrix import ConfusionMatrix
from spectral_scripts.features.profile import SpectralProfile


def frobenius_distance(
    matrix1: NDArray[np.float64],
    matrix2: NDArray[np.float64],
    normalize: bool = True,
    align: bool = True,
) -> float:
    """
    Compute Frobenius norm distance between two matrices.

    d_F(A, B) = ||A - B||_F = sqrt(Σᵢⱼ (Aᵢⱼ - Bᵢⱼ)²)

    Args:
        matrix1: First matrix.
        matrix2: Second matrix.
        normalize: If True, normalize matrices to sum to 1 first.
        align: If True, align matrices to same size before computing distance.
               If False, matrices must have same shape.

    Returns:
        Frobenius distance.

    Raises:
        ValueError: If matrices can't be aligned and shapes don't match.
    """
    m1 = matrix1.astype(np.float64).copy()
    m2 = matrix2.astype(np.float64).copy()

    # Align matrices if needed
    if m1.shape != m2.shape:
        if not align:
            raise ValueError(
                f"Matrices must have same shape: {m1.shape} vs {m2.shape}"
            )
        # Pad to same size
        max_size = max(m1.shape[0], m2.shape[0])
        m1_padded = np.zeros((max_size, max_size))
        m2_padded = np.zeros((max_size, max_size))
        
        m1_padded[:m1.shape[0], :m1.shape[1]] = m1
        m2_padded[:m2.shape[0], :m2.shape[1]] = m2
        
        m1 = m1_padded
        m2 = m2_padded

    if normalize:
        s1, s2 = m1.sum(), m2.sum()
        if s1 > 0:
            m1 = m1 / s1
        if s2 > 0:
            m2 = m2 / s2

    return float(np.linalg.norm(m1 - m2, "fro"))


def accuracy_distance(
    confusion1: ConfusionMatrix,
    confusion2: ConfusionMatrix,
) -> float:
    """
    Compute simple accuracy difference.

    d_acc(A, B) = |accuracy(A) - accuracy(B)|

    This is a naive baseline that ignores confusion structure.

    Args:
        confusion1: First confusion matrix.
        confusion2: Second confusion matrix.

    Returns:
        Absolute accuracy difference in [0, 1].
    """
    return abs(confusion1.accuracy - confusion2.accuracy)


def character_overlap_distance(
    confusion1: ConfusionMatrix,
    confusion2: ConfusionMatrix,
) -> float:
    """
    Compute Jaccard distance based on character set overlap.

    d_J(A, B) = 1 - |chars(A) ∩ chars(B)| / |chars(A) ∪ chars(B)|

    This baseline captures whether scripts share characters
    but ignores actual confusion patterns.

    Args:
        confusion1: First confusion matrix.
        confusion2: Second confusion matrix.

    Returns:
        Jaccard distance in [0, 1].
    """
    chars1 = set(confusion1.characters)
    chars2 = set(confusion2.characters)

    if not chars1 and not chars2:
        return 0.0

    intersection = len(chars1 & chars2)
    union = len(chars1 | chars2)

    jaccard = intersection / union
    return 1.0 - jaccard


def confusion_pattern_distance(
    confusion1: ConfusionMatrix,
    confusion2: ConfusionMatrix,
    top_k: int = 20,
) -> float:
    """
    Compare top confusion patterns between two matrices.

    Extracts top-k confusions from each matrix and computes
    Jaccard distance on the confusion pairs (ignoring counts).

    Args:
        confusion1: First confusion matrix.
        confusion2: Second confusion matrix.
        top_k: Number of top confusions to compare.

    Returns:
        Jaccard distance of confusion pair sets.
    """
    confusions1 = confusion1.top_confusions(top_k)
    confusions2 = confusion2.top_confusions(top_k)

    # Extract (true, pred) pairs
    pairs1 = {(c[0], c[1]) for c in confusions1}
    pairs2 = {(c[0], c[1]) for c in confusions2}

    if not pairs1 and not pairs2:
        return 0.0

    intersection = len(pairs1 & pairs2)
    union = len(pairs1 | pairs2)

    if union == 0:
        return 0.0

    jaccard = intersection / union
    return 1.0 - jaccard


@dataclass(frozen=True)
class BaselineDistances:
    """
    Container for all baseline distance computations.

    Attributes:
        frobenius: Frobenius norm distance (normalized matrices).
        accuracy: Absolute accuracy difference.
        character_overlap: Jaccard distance of character sets.
        confusion_pattern: Jaccard distance of top confusion pairs.
    """

    frobenius: float
    accuracy: float
    character_overlap: float
    confusion_pattern: float

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "frobenius": self.frobenius,
            "accuracy": self.accuracy,
            "character_overlap": self.character_overlap,
            "confusion_pattern": self.confusion_pattern,
        }

    def as_array(self) -> NDArray[np.float64]:
        """Convert to numpy array."""
        return np.array([
            self.frobenius,
            self.accuracy,
            self.character_overlap,
            self.confusion_pattern,
        ])


def compute_baseline_distances(
    confusion1: ConfusionMatrix,
    confusion2: ConfusionMatrix,
) -> BaselineDistances:
    """
    Compute all baseline distances between two confusion matrices.

    Handles matrices of different sizes by padding them.

    Args:
        confusion1: First confusion matrix.
        confusion2: Second confusion matrix.

    Returns:
        BaselineDistances with all computed values.
    """
    # Accuracy distance
    acc_dist = accuracy_distance(confusion1, confusion2)

    # Character overlap distance
    char_dist = character_overlap_distance(confusion1, confusion2)

    # Confusion pattern distance
    pattern_dist = confusion_pattern_distance(confusion1, confusion2)

    # Frobenius distance - now handles different sizes via padding
    frob_dist = frobenius_distance(
        confusion1.matrix,
        confusion2.matrix,
        normalize=True,
        align=True,  # Enable alignment
    )

    return BaselineDistances(
        frobenius=frob_dist,
        accuracy=acc_dist,
        character_overlap=char_dist,
        confusion_pattern=pattern_dist,
    )