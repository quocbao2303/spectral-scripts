"""Build confusion matrices from OCR predictions."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Iterator
import logging

import numpy as np
from numpy.typing import NDArray

from spectral_scripts.core.confusion_matrix import ConfusionMatrix
from spectral_scripts.ocr_pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


@dataclass
class ConfusionMatrixBuilder:
    """Build confusion matrix from character-level predictions."""

    script: str
    charset: str
    counts: dict[tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))
    total_samples: int = 0
    total_characters: int = 0

    def add_sample(self, true_char: str, pred_char: str, count: int = 1) -> None:
        """Add a single confusion sample."""
        self.counts[(true_char, pred_char)] += count
        self.total_samples += count
        self.total_characters = len(set(c for pair in self.counts for c in pair))

    def add_pairs(self, pairs: list[tuple[str, str]]) -> None:
        """Add multiple confusion pairs."""
        for true_char, pred_char in pairs:
            self.add_sample(true_char, pred_char)

    def build(self) -> ConfusionMatrix:
        """
        Build ConfusionMatrix from accumulated counts.

        Returns:
            ConfusionMatrix object ready for spectral analysis.
        """
        # Get all unique characters from the charset
        all_chars = sorted(set(self.charset))

        # Also add any characters seen in predictions (including "surprise" chars)
        seen_chars = set()
        for (true_char, pred_char) in self.counts.keys():
            if true_char:
                seen_chars.add(true_char)
            if pred_char:
                seen_chars.add(pred_char)

        all_chars = sorted(set(all_chars) | seen_chars)
        n = len(all_chars)
        char_to_idx = {c: i for i, c in enumerate(all_chars)}

        # Build raw matrix
        matrix = np.zeros((n, n), dtype=np.float64)

        for (true_char, pred_char), count in self.counts.items():
            if true_char in char_to_idx and pred_char in char_to_idx:
                i = char_to_idx[true_char]
                j = char_to_idx[pred_char]
                matrix[i, j] += count

        logger.info(
            "Built confusion matrix for %s: %dx%d, %d samples (pre-prune)",
            self.script,
            n,
            n,
            self.total_samples,
        )

        # Wrap in ConfusionMatrix and prune characters that never appear
        cm = ConfusionMatrix(
            matrix=matrix,
            script=self.script,
            characters=all_chars,
            metadata={
                "total_samples": self.total_samples,
                "source": "ocr_pipeline",
            },
        )

        pruned = cm.prune_unused(min_total=1)

        if pruned.size != cm.size:
            removed = cm.size - pruned.size
            logger.info(
                "Pruned %d unused characters for %s: size %d → %d",
                removed,
                self.script,
                cm.size,
                pruned.size,
            )
        else:
            logger.info(
                "No unused characters to prune for %s (size=%d)",
                self.script,
                cm.size,
            )

        # Keep builder summary in sync with the pruned matrix
        self.total_characters = pruned.size

        return pruned

    def get_accuracy(self) -> float:
        """Calculate overall accuracy from current counts."""
        correct = sum(
            count
            for (true_char, pred_char), count in self.counts.items()
            if true_char == pred_char
        )
        return correct / self.total_samples if self.total_samples > 0 else 0.0

    def get_top_confusions(self, n: int = 10) -> list[tuple[str, str, int]]:
        """Get top n confusion pairs (excluding correct predictions)."""
        confusions = [
            (true_char, pred_char, count)
            for (true_char, pred_char), count in self.counts.items()
            if true_char != pred_char
        ]
        confusions.sort(key=lambda x: -x[2])
        return confusions[:n]

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            f"Confusion Matrix Builder: {self.script}",
            f"  Total samples: {self.total_samples}",
            f"  Unique characters: {self.total_characters}",
            f"  Accuracy: {self.get_accuracy():.1%}",
            "",
            "Top confusions:",
        ]

        for true_char, pred_char, count in self.get_top_confusions(5):
            lines.append(f"  '{true_char}' → '{pred_char}': {count}")

        return "\n".join(lines)


class MultiScriptMatrixBuilder:
    """Build confusion matrices for multiple scripts."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.builders: dict[str, ConfusionMatrixBuilder] = {}

        for script in config.scripts:
            charset = config.get_charset(script)
            self.builders[script] = ConfusionMatrixBuilder(
                script=script,
                charset=charset,
            )

    def add_pairs(self, script: str, pairs: list[tuple[str, str]]) -> None:
        """Add confusion pairs for a specific script."""
        if script not in self.builders:
            raise ValueError(f"Unknown script: {script}")
        self.builders[script].add_pairs(pairs)

    def add_sample(self, script: str, true_char: str, pred_char: str) -> None:
        """Add a single sample for a script."""
        if script not in self.builders:
            raise ValueError(f"Unknown script: {script}")
        self.builders[script].add_sample(true_char, pred_char)

    def build_all(self) -> dict[str, ConfusionMatrix]:
        """Build confusion matrices for all scripts (with pruning)."""
        return {
            script: builder.build()
            for script, builder in self.builders.items()
        }

    def get_statistics(self) -> dict[str, dict]:
        """Get statistics for all scripts."""
        return {
            script: {
                "total_samples": builder.total_samples,
                "unique_characters": builder.total_characters,
                "accuracy": builder.get_accuracy(),
                "top_confusions": builder.get_top_confusions(5),
            }
            for script, builder in self.builders.items()
        }

    def summary(self) -> str:
        """Generate summary for all scripts."""
        lines = ["=== Multi-Script Confusion Matrix Summary ===", ""]
        for script, builder in self.builders.items():
            lines.append(builder.summary())
            lines.append("")
        return "\n".join(lines)
