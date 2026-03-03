"""Character matching: Align OCR predictions to ground truth."""

from __future__ import annotations

from dataclasses import dataclass
import unicodedata
import logging

from spectral_scripts.ocr_pipeline.config import PipelineConfig, MatchingConfig

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of matching predicted characters to ground truth."""

    true_chars: list[str]
    pred_chars: list[str]
    alignments: list[tuple[str | None, str | None]]
    matches: int
    substitutions: int
    insertions: int
    deletions: int

    @property
    def total_true(self) -> int:
        """Total ground truth characters."""
        return len(self.true_chars)

    @property
    def total_pred(self) -> int:
        """Total predicted characters."""
        return len(self.pred_chars)

    @property
    def accuracy(self) -> float:
        """Character-level accuracy."""
        if self.total_true == 0:
            return 0.0
        return self.matches / self.total_true

    @property
    def error_rate(self) -> float:
        """Character error rate (CER)."""
        if self.total_true == 0:
            return 1.0
        return (self.substitutions + self.insertions + self.deletions) / self.total_true


class CharacterMatcher:
    """Match predicted characters to ground truth using edit distance."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.matching_config = config.matching

    def match(self, ground_truth: str, prediction: str) -> MatchResult:
        """
        Match predicted text to ground truth.

        Uses Levenshtein edit distance with backtracking to align characters.

        Args:
            ground_truth: True text.
            prediction: OCR predicted text.

        Returns:
            MatchResult with alignment details.
        """
        # Normalize if configured
        if self.matching_config.normalize_unicode:
            ground_truth = self._normalize_unicode(ground_truth)
            prediction = self._normalize_unicode(prediction)

        # Handle case sensitivity
        if not self.matching_config.case_sensitive:
            ground_truth = ground_truth.lower()
            prediction = prediction.lower()

        true_chars = list(ground_truth)
        pred_chars = list(prediction)

        # Compute edit distance matrix
        matrix = self._compute_edit_distance_matrix(true_chars, pred_chars)

        # Backtrack to get alignments
        alignments = self._backtrack(matrix, true_chars, pred_chars)

        # Count operations
        matches = 0
        substitutions = 0
        insertions = 0
        deletions = 0

        for true_char, pred_char in alignments:
            if true_char is None:
                insertions += 1
            elif pred_char is None:
                deletions += 1
            elif true_char == pred_char:
                matches += 1
            else:
                substitutions += 1

        return MatchResult(
            true_chars=true_chars,
            pred_chars=pred_chars,
            alignments=alignments,
            matches=matches,
            substitutions=substitutions,
            insertions=insertions,
            deletions=deletions,
        )

    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode to NFC form."""
        return unicodedata.normalize("NFC", text)

    def _compute_edit_distance_matrix(
        self, s1: list[str], s2: list[str]
    ) -> list[list[int]]:
        """Compute Levenshtein edit distance matrix."""
        m, n = len(s1), len(s2)
        
        # Initialize matrix
        matrix = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Base cases
        for i in range(m + 1):
            matrix[i][0] = i
        for j in range(n + 1):
            matrix[0][j] = j

        # Fill matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    cost = 0
                else:
                    cost = 1

                matrix[i][j] = min(
                    matrix[i - 1][j] + 1,      # Deletion
                    matrix[i][j - 1] + 1,      # Insertion
                    matrix[i - 1][j - 1] + cost  # Substitution/Match
                )

        return matrix

    def _backtrack(
        self, matrix: list[list[int]], s1: list[str], s2: list[str]
    ) -> list[tuple[str | None, str | None]]:
        """Backtrack through edit distance matrix to get alignments."""
        alignments = []
        i, j = len(s1), len(s2)

        while i > 0 or j > 0:
            if i > 0 and j > 0 and s1[i - 1] == s2[j - 1]:
                # Match
                alignments.append((s1[i - 1], s2[j - 1]))
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and matrix[i][j] == matrix[i - 1][j - 1] + 1:
                # Substitution
                alignments.append((s1[i - 1], s2[j - 1]))
                i -= 1
                j -= 1
            elif j > 0 and matrix[i][j] == matrix[i][j - 1] + 1:
                # Insertion (extra char in prediction)
                alignments.append((None, s2[j - 1]))
                j -= 1
            elif i > 0:
                # Deletion (missing char in prediction)
                alignments.append((s1[i - 1], None))
                i -= 1

        # Reverse since we backtracked
        alignments.reverse()
        return alignments

    def get_confusion_pairs(
        self, ground_truth: str, prediction: str
    ) -> list[tuple[str, str]]:
        """
        Get list of (true_char, predicted_char) pairs.

        Useful for building confusion matrix.
        Only includes substitutions and matches (not insertions/deletions).

        Returns:
            List of (true_char, predicted_char) tuples.
        """
        result = self.match(ground_truth, prediction)
        
        pairs = []
        for true_char, pred_char in result.alignments:
            if true_char is not None and pred_char is not None:
                pairs.append((true_char, pred_char))
        
        return pairs


class SequenceAligner:
    """Alternative alignment using dynamic programming."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.gap_penalty = -1
        self.match_score = 2
        self.mismatch_score = -1

    def align(self, seq1: str, seq2: str) -> tuple[str, str]:
        """
        Align two sequences using Needleman-Wunsch algorithm.

        Returns aligned sequences with gaps marked as '-'.
        """
        m, n = len(seq1), len(seq2)
        
        # Initialize score matrix
        score = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize first row and column
        for i in range(m + 1):
            score[i][0] = i * self.gap_penalty
        for j in range(n + 1):
            score[0][j] = j * self.gap_penalty

        # Fill score matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    diag = score[i - 1][j - 1] + self.match_score
                else:
                    diag = score[i - 1][j - 1] + self.mismatch_score
                
                up = score[i - 1][j] + self.gap_penalty
                left = score[i][j - 1] + self.gap_penalty
                
                score[i][j] = max(diag, up, left)

        # Traceback
        aligned1 = []
        aligned2 = []
        i, j = m, n

        while i > 0 or j > 0:
            if i > 0 and j > 0:
                if seq1[i - 1] == seq2[j - 1]:
                    current_score = self.match_score
                else:
                    current_score = self.mismatch_score
                
                if score[i][j] == score[i - 1][j - 1] + current_score:
                    aligned1.append(seq1[i - 1])
                    aligned2.append(seq2[j - 1])
                    i -= 1
                    j -= 1
                    continue

            if i > 0 and score[i][j] == score[i - 1][j] + self.gap_penalty:
                aligned1.append(seq1[i - 1])
                aligned2.append("-")
                i -= 1
            elif j > 0:
                aligned1.append("-")
                aligned2.append(seq2[j - 1])
                j -= 1

        return "".join(reversed(aligned1)), "".join(reversed(aligned2))