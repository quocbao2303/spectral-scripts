"""Statistical utilities and multiple testing corrections."""

from spectral_scripts.statistics.corrections import (
    bonferroni_correction,
    fdr_correction,
    holm_bonferroni_correction,
    correct_pvalues,
    CorrectedPValues,
)

__all__ = [
    "bonferroni_correction",
    "fdr_correction",
    "holm_bonferroni_correction",
    "correct_pvalues",
    "CorrectedPValues",
]