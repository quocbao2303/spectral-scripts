"""Human-interpretable features from confusion matrices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from spectral_scripts.core.confusion_matrix import ConfusionMatrix


@dataclass(frozen=True)
class InterpretableFeatures:
    """
    Human-interpretable features from a confusion matrix.

    These features provide intuitive measures that can be understood
    without linear algebra background.

    Attributes:
        script: Name of the script.
        accuracy: Overall OCR accuracy.
        error_rate: Overall OCR error rate (1 - accuracy).
        sparsity: Fraction of zero entries in confusion matrix.

        confusion_concentration: How concentrated errors are (Gini coefficient).
        top_confusion_ratio: Fraction of errors in top 10 confusion pairs.
        case_confusion_rate: Rate of upper/lower case confusions.

        diagonal_dominance: Ratio of diagonal to off-diagonal mass.
        symmetry_score: How symmetric the confusion pattern is.
        entropy_per_row: Average entropy of each row (prediction uncertainty).
    """

    script: str
    accuracy: float
    error_rate: float
    sparsity: float

    confusion_concentration: float
    top_confusion_ratio: float
    case_confusion_rate: float

    diagonal_dominance: float
    symmetry_score: float
    entropy_per_row: float

    def to_feature_vector(self) -> NDArray[np.float64]:
        """Convert to feature vector for analysis."""
        return np.array([
            self.accuracy,
            self.sparsity,
            self.confusion_concentration,
            self.top_confusion_ratio,
            self.diagonal_dominance,
            self.symmetry_score,
            self.entropy_per_row,
        ], dtype=np.float64)

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for serialization."""
        return {
            "script": self.script,
            "accuracy": self.accuracy,
            "error_rate": self.error_rate,
            "sparsity": self.sparsity,
            "confusion_concentration": self.confusion_concentration,
            "top_confusion_ratio": self.top_confusion_ratio,
            "case_confusion_rate": self.case_confusion_rate,
            "diagonal_dominance": self.diagonal_dominance,
            "symmetry_score": self.symmetry_score,
            "entropy_per_row": self.entropy_per_row,
        }


def compute_gini_coefficient(values: NDArray[np.float64]) -> float:
    """
    Compute Gini coefficient of a distribution.

    Gini = 0: Perfect equality (all values equal)
    Gini = 1: Perfect inequality (one value has everything)

    Args:
        values: Array of non-negative values.

    Returns:
        Gini coefficient in [0, 1].
    """
    values = np.sort(values.ravel())
    n = len(values)

    if n == 0 or values.sum() == 0:
        return 0.0

    # Gini formula: G = (2 * Σᵢ i*xᵢ) / (n * Σᵢ xᵢ) - (n+1)/n
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * values)) / (n * values.sum()) - (n + 1) / n)


def compute_row_entropy(row: NDArray[np.float64]) -> float:
    """Compute Shannon entropy of a single row."""
    total = row.sum()
    if total == 0:
        return 0.0

    probs = row / total
    probs = probs[probs > 0]

    return float(-np.sum(probs * np.log(probs)))


def compute_symmetry_score(matrix: NDArray[np.float64]) -> float:
    """
    Compute how symmetric the confusion matrix is.

    Score = 1 - (||A - A^T||_F / ||A||_F)

    Score = 1: Perfectly symmetric
    Score = 0: Completely asymmetric

    Args:
        matrix: Confusion matrix.

    Returns:
        Symmetry score in [0, 1].
    """
    norm_a = np.linalg.norm(matrix, "fro")
    if norm_a == 0:
        return 1.0

    diff_norm = np.linalg.norm(matrix - matrix.T, "fro")
    return float(1.0 - diff_norm / (2 * norm_a))


def extract_interpretable_features(
    confusion: ConfusionMatrix,
) -> InterpretableFeatures:
    """
    Extract human-interpretable features from confusion matrix.

    Args:
        confusion: Input confusion matrix.

    Returns:
        InterpretableFeatures with all computed values.
    """
    matrix = confusion.matrix
    n = confusion.size

    # Basic metrics
    accuracy = confusion.accuracy
    error_rate = confusion.error_rate
    sparsity = confusion.sparsity

    # Off-diagonal analysis
    off_diagonal_mask = ~np.eye(n, dtype=bool)
    off_diagonal = matrix[off_diagonal_mask]
    total_errors = off_diagonal.sum()

    # Confusion concentration (Gini of off-diagonal)
    confusion_concentration = compute_gini_coefficient(off_diagonal)

    # Top confusion ratio
    if total_errors > 0:
        top_10_errors = np.sort(off_diagonal)[-10:].sum()
        top_confusion_ratio = float(top_10_errors / total_errors)
    else:
        top_confusion_ratio = 0.0

    # Case confusion rate (assuming first half lowercase, second half uppercase)
    # This is a heuristic - actual implementation depends on character ordering
    half = n // 2
    if half > 0 and n > half:
        case_confusions = (
            matrix[:half, half:].sum() + matrix[half:, :half].sum()
        )
        case_confusion_rate = float(case_confusions / max(total_errors, 1))
    else:
        case_confusion_rate = 0.0

    # Diagonal dominance
    diagonal_sum = np.trace(matrix)
    off_diagonal_sum = total_errors
    if off_diagonal_sum > 0:
        diagonal_dominance = float(diagonal_sum / off_diagonal_sum)
    else:
        diagonal_dominance = float("inf") if diagonal_sum > 0 else 0.0

    # Symmetry score
    symmetry_score = compute_symmetry_score(matrix)

    # Average row entropy
    row_entropies = [compute_row_entropy(matrix[i, :]) for i in range(n)]
    entropy_per_row = float(np.mean(row_entropies))

    return InterpretableFeatures(
        script=confusion.script,
        accuracy=accuracy,
        error_rate=error_rate,
        sparsity=sparsity,
        confusion_concentration=confusion_concentration,
        top_confusion_ratio=top_confusion_ratio,
        case_confusion_rate=case_confusion_rate,
        diagonal_dominance=diagonal_dominance,
        symmetry_score=symmetry_score,
        entropy_per_row=entropy_per_row,
    )