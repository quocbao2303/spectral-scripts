"""Confusion matrix data container and I/O."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Self
else:
    Self = "ConfusionMatrix"
import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConfusionMatrix:
    """
    Container for an OCR confusion matrix.

    Attributes:
        matrix: Raw count matrix where matrix[i,j] = count of true label i
            predicted as label j.
        script: Name of the script (e.g., "latin", "greek", "cyrillic").
        characters: List of character labels corresponding to matrix indices.
        metadata: Optional dictionary for additional information.
    """

    matrix: NDArray[np.float64]
    script: str
    characters: list[str]
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate matrix properties."""
        if self.matrix.ndim != 2:
            raise ValueError(f"Matrix must be 2D, got {self.matrix.ndim}D")
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError(
                f"Matrix must be square, got shape {self.matrix.shape}"
            )
        if len(self.characters) != self.matrix.shape[0]:
            raise ValueError(
                f"Characters length ({len(self.characters)}) must match "
                f"matrix size ({self.matrix.shape[0]})"
            )
        if np.any(self.matrix < 0):
            raise ValueError("Matrix cannot contain negative values")

    @property
    def size(self) -> int:
        """Number of characters in the confusion matrix."""
        return self.matrix.shape[0]

    @property
    def total_observations(self) -> int:
        """Total number of observations (sum of all counts)."""
        return int(self.matrix.sum())

    @property
    def accuracy(self) -> float:
        """Overall OCR accuracy (diagonal sum / total)."""
        total = self.matrix.sum()
        if total == 0:
            return 0.0
        return float(np.trace(self.matrix) / total)

    @property
    def error_rate(self) -> float:
        """Overall OCR error rate (1 - accuracy)."""
        return 1.0 - self.accuracy

    @property
    def sparsity(self) -> float:
        """Fraction of zero entries in the matrix."""
        return float(np.sum(self.matrix == 0) / self.matrix.size)

    def character_accuracy(self, char: str) -> float:
        """Get accuracy for a specific character."""
        if char not in self.characters:
            raise ValueError(f"Character '{char}' not in matrix")
        idx = self.characters.index(char)
        row_sum = self.matrix[idx, :].sum()
        if row_sum == 0:
            return 0.0
        return float(self.matrix[idx, idx] / row_sum)

    def top_confusions(self, n: int = 10) -> list[tuple[str, str, float]]:
        """
        Get the top n off-diagonal confusions.

        Returns:
            List of (true_char, predicted_char, count) tuples.
        """
        # Create mask for off-diagonal elements
        mask = ~np.eye(self.size, dtype=bool)
        off_diag = self.matrix.copy()
        off_diag[~mask] = 0

        # Get indices of top n values
        flat_indices = np.argsort(off_diag.ravel())[::-1][:n]
        row_indices, col_indices = np.unravel_index(flat_indices, off_diag.shape)

        result = []
        for i, j in zip(row_indices, col_indices):
            if off_diag[i, j] > 0:
                result.append(
                    (self.characters[i], self.characters[j], float(off_diag[i, j]))
                )
        return result

    def prune_unused(self, min_total: int = 1) -> Self:
        """
        Return a new ConfusionMatrix with unused characters removed.

        A character i is kept if:

            row_sum[i] + col_sum[i] >= min_total

        i.e. it appears at least `min_total` times either as ground truth
        or as a prediction. This removes zero rows/columns and can also be
        used as a simple frequency-based pruning.

        Args:
            min_total: Minimum total participation (row+col) required to
                       keep a character. Default 1 = drop only truly dead
                       characters.

        Returns:
            A new ConfusionMatrix instance with a pruned matrix and
            character list. If nothing is pruned, returns self.
        """
        if min_total < 1:
            raise ValueError(f"min_total must be >= 1, got {min_total}")

        if self.matrix.size == 0:
            logger.warning(
                "ConfusionMatrix.prune_unused called on empty matrix for script '%s'",
                self.script,
            )
            return self

        row_sums = self.matrix.sum(axis=1)
        col_sums = self.matrix.sum(axis=0)
        totals = row_sums + col_sums

        mask = totals >= min_total
        removed = int((~mask).sum())

        if removed == 0:
            logger.info(
                "Prune unused: nothing to prune for script '%s' (size=%d)",
                self.script,
                self.size,
            )
            return self

        pruned_matrix = self.matrix[np.ix_(mask, mask)]
        pruned_characters = [c for c, keep in zip(self.characters, mask) if keep]

        logger.info(
            "Pruned %d unused characters for script '%s' (kept=%d, original=%d)",
            removed,
            self.script,
            len(pruned_characters),
            self.size,
        )

        new_metadata = dict(self.metadata)
        new_metadata["pruned_unused_original_size"] = self.size
        new_metadata["pruned_unused_removed"] = removed
        new_metadata["pruned_unused_min_total"] = min_total

        return ConfusionMatrix(
            matrix=pruned_matrix,
            script=self.script,
            characters=pruned_characters,
            metadata=new_metadata,
        )

    @classmethod
    def from_npz(cls, path: str | Path) -> Self:
        """
        Load confusion matrix from .npz file.

        Expected keys in .npz:
            - matrix: The confusion matrix array
            - script: Script name (string)
            - characters: List of character labels (optional, auto-generated if missing)

        Optional keys:
            - Any additional metadata
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        data = np.load(path, allow_pickle=True)

        required_keys = {"matrix", "script"}
        missing = required_keys - set(data.keys())
        if missing:
            raise ValueError(f"Missing required keys in .npz file: {missing}")

        # Extract required fields
        matrix = data["matrix"].astype(np.float64)
        script = str(data["script"])

        # Extract characters or generate them if missing
        if "characters" in data:
            characters = list(data["characters"])
        else:
            # Auto-generate character labels based on matrix size
            n = matrix.shape[0]
            if n <= 26:
                # For small matrices, use a, b, c, ...
                characters = [chr(ord("a") + i) for i in range(n)]
            else:
                # For larger matrices, use char_0, char_1, ...
                characters = [f"char_{i}" for i in range(n)]

        # Extract optional metadata
        metadata: dict[str, object] = {}
        for k in data.keys():
            if k not in {"matrix", "script", "characters"}:
                val = data[k]
                if hasattr(val, "ndim") and val.ndim == 0:
                    metadata[k] = val.item()
                else:
                    metadata[k] = val

        return cls(
            matrix=matrix,
            script=script,
            characters=characters,
            metadata=metadata,
        )

    def to_npz(self, path: str | Path) -> None:
        """Save confusion matrix to .npz file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            path,
            matrix=self.matrix,
            script=self.script,
            characters=np.array(self.characters, dtype=object),
            **self.metadata,
        )

    def subsample(
        self, fraction: float, rng: np.random.Generator | None = None
    ) -> Self:
        """
        Create a subsampled version of the confusion matrix.

        Uses multinomial resampling to create a matrix with
        `fraction * total_observations` total counts.

        Args:
            fraction: Fraction of observations to keep (0 < fraction <= 1).
            rng: Random number generator for reproducibility.

        Returns:
            New ConfusionMatrix with subsampled counts.
        """
        if not 0 < fraction <= 1:
            raise ValueError(f"Fraction must be in (0, 1], got {fraction}")

        if rng is None:
            rng = np.random.default_rng()

        # Flatten matrix to probability distribution
        total = self.matrix.sum()
        if total == 0:
            return self

        probs = self.matrix.ravel() / total
        n_samples = int(total * fraction)

        # Multinomial resampling
        counts = rng.multinomial(n_samples, probs)
        new_matrix = counts.reshape(self.matrix.shape).astype(np.float64)

        return ConfusionMatrix(
            matrix=new_matrix,
            script=self.script,
            characters=self.characters,
            metadata={**self.metadata, "subsampled_fraction": fraction},
        )

    def __repr__(self) -> str:
        return (
            f"ConfusionMatrix(script='{self.script}', size={self.size}, "
            f"accuracy={self.accuracy:.1%}, observations={self.total_observations})"
        )
