"""Multiple testing correction methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CorrectedPValues:
    """
    Container for multiple testing corrected p-values.

    Attributes:
        original: Original uncorrected p-values.
        corrected: Corrected p-values.
        significant: Boolean mask of significant results at alpha.
        method: Correction method used.
        alpha: Significance level.
        n_significant: Number of significant results.
        n_tests: Total number of tests.
    """

    original: NDArray[np.float64]
    corrected: NDArray[np.float64]
    significant: NDArray[np.bool_]
    method: str
    alpha: float

    @property
    def n_significant(self) -> int:
        """Number of significant results."""
        return int(self.significant.sum())

    @property
    def n_tests(self) -> int:
        """Total number of tests."""
        return len(self.original)

    @property
    def proportion_significant(self) -> float:
        """Proportion of significant results."""
        if self.n_tests == 0:
            return 0.0
        return self.n_significant / self.n_tests

    def summary(self) -> str:
        """Generate human-readable summary."""
        return (
            f"Multiple Testing Correction ({self.method})\n"
            f"  Tests: {self.n_tests}\n"
            f"  Alpha: {self.alpha}\n"
            f"  Significant: {self.n_significant} ({self.proportion_significant:.1%})\n"
            f"  Min corrected p: {self.corrected.min():.2e}\n"
            f"  Max corrected p: {self.corrected.max():.2e}"
        )


def bonferroni_correction(
    pvalues: NDArray[np.float64],
    alpha: float = 0.05,
) -> CorrectedPValues:
    """
    Apply Bonferroni correction for multiple testing.

    Most conservative method. Controls family-wise error rate (FWER).
    Adjusted p-value = p * n, capped at 1.0.

    Args:
        pvalues: Array of uncorrected p-values.
        alpha: Significance level.

    Returns:
        CorrectedPValues with Bonferroni-corrected values.
    """
    n = len(pvalues)
    corrected = np.minimum(pvalues * n, 1.0)
    significant = corrected < alpha

    return CorrectedPValues(
        original=pvalues.copy(),
        corrected=corrected,
        significant=significant,
        method="bonferroni",
        alpha=alpha,
    )


def holm_bonferroni_correction(
    pvalues: NDArray[np.float64],
    alpha: float = 0.05,
) -> CorrectedPValues:
    """
    Apply Holm-Bonferroni (step-down) correction.

    Less conservative than Bonferroni while still controlling FWER.
    Sequentially rejects hypotheses starting from smallest p-value.

    Args:
        pvalues: Array of uncorrected p-values.
        alpha: Significance level.

    Returns:
        CorrectedPValues with Holm-corrected values.
    """
    n = len(pvalues)
    if n == 0:
        return CorrectedPValues(
            original=pvalues.copy(),
            corrected=pvalues.copy(),
            significant=np.array([], dtype=bool),
            method="holm-bonferroni",
            alpha=alpha,
        )

    # Sort p-values and track original indices
    sort_idx = np.argsort(pvalues)
    sorted_pvals = pvalues[sort_idx]

    # Compute adjusted p-values
    # p_adj[i] = max(p[j] * (n - j) for j <= i)
    corrected_sorted = np.zeros(n)
    cummax = 0.0

    for i in range(n):
        adjusted = sorted_pvals[i] * (n - i)
        cummax = max(cummax, adjusted)
        corrected_sorted[i] = min(cummax, 1.0)

    # Restore original order
    corrected = np.zeros(n)
    corrected[sort_idx] = corrected_sorted

    significant = corrected < alpha

    return CorrectedPValues(
        original=pvalues.copy(),
        corrected=corrected,
        significant=significant,
        method="holm-bonferroni",
        alpha=alpha,
    )


def fdr_correction(
    pvalues: NDArray[np.float64],
    alpha: float = 0.05,
    method: Literal["bh", "by"] = "bh",
) -> CorrectedPValues:
    """
    Apply False Discovery Rate (FDR) correction.

    Controls the expected proportion of false positives among rejections.
    Less conservative than FWER methods, more powerful for exploratory analysis.

    Args:
        pvalues: Array of uncorrected p-values.
        alpha: Target FDR level.
        method: FDR method:
            - "bh": Benjamini-Hochberg (assumes independence or positive dependence)
            - "by": Benjamini-Yekutieli (valid under any dependence structure)

    Returns:
        CorrectedPValues with FDR-corrected values.
    """
    n = len(pvalues)
    if n == 0:
        return CorrectedPValues(
            original=pvalues.copy(),
            corrected=pvalues.copy(),
            significant=np.array([], dtype=bool),
            method=f"fdr-{method}",
            alpha=alpha,
        )

    # Sort p-values
    sort_idx = np.argsort(pvalues)
    sorted_pvals = pvalues[sort_idx]

    # Compute correction factor for BY method
    if method == "by":
        # c(m) = Σ_{i=1}^{m} 1/i ≈ ln(m) + γ
        correction_factor = np.sum(1.0 / np.arange(1, n + 1))
    else:
        correction_factor = 1.0

    # Compute adjusted p-values using step-up procedure
    # q[i] = min(p[j] * n * c(m) / (j+1) for j >= i)
    corrected_sorted = np.zeros(n)
    cummin = 1.0

    for i in range(n - 1, -1, -1):
        adjusted = sorted_pvals[i] * n * correction_factor / (i + 1)
        cummin = min(cummin, adjusted)
        corrected_sorted[i] = min(cummin, 1.0)

    # Restore original order
    corrected = np.zeros(n)
    corrected[sort_idx] = corrected_sorted

    significant = corrected < alpha

    return CorrectedPValues(
        original=pvalues.copy(),
        corrected=corrected,
        significant=significant,
        method=f"fdr-{method}",
        alpha=alpha,
    )


def correct_pvalues(
    pvalues: NDArray[np.float64],
    method: Literal["bonferroni", "holm", "fdr-bh", "fdr-by"] = "fdr-bh",
    alpha: float = 0.05,
) -> CorrectedPValues:
    """
    Apply multiple testing correction using specified method.

    Recommended methods:
    - "bonferroni": Most conservative, use when false positives are very costly
    - "holm": Less conservative than Bonferroni, controls FWER
    - "fdr-bh": Good default for exploratory analysis
    - "fdr-by": Use when tests may be dependent

    Args:
        pvalues: Array of uncorrected p-values.
        method: Correction method to use.
        alpha: Significance level or FDR target.

    Returns:
        CorrectedPValues with corrected values and significance flags.

    Raises:
        ValueError: If method is not recognized.
    """
    pvalues = np.asarray(pvalues, dtype=np.float64)

    if method == "bonferroni":
        return bonferroni_correction(pvalues, alpha)
    elif method == "holm":
        return holm_bonferroni_correction(pvalues, alpha)
    elif method == "fdr-bh":
        return fdr_correction(pvalues, alpha, method="bh")
    elif method == "fdr-by":
        return fdr_correction(pvalues, alpha, method="by")
    else:
        raise ValueError(
            f"Unknown correction method: {method}. "
            f"Choose from: bonferroni, holm, fdr-bh, fdr-by"
        )


def pairwise_correction(
    n_scripts: int,
    method: Literal["bonferroni", "holm", "fdr-bh", "fdr-by"] = "bonferroni",
) -> float:
    """
    Compute adjusted alpha for pairwise comparisons.

    For n scripts, there are n(n-1)/2 pairwise comparisons.
    This function returns the per-comparison alpha needed to maintain
    the desired family-wise alpha.

    Args:
        n_scripts: Number of scripts being compared.
        method: Correction method.

    Returns:
        Adjusted alpha per comparison.

    Example:
        >>> pairwise_correction(10, "bonferroni")  # 10 scripts = 45 pairs
        0.0011111...  # 0.05 / 45
    """
    n_comparisons = n_scripts * (n_scripts - 1) // 2

    if method == "bonferroni":
        return 0.05 / n_comparisons
    elif method in ("holm", "fdr-bh", "fdr-by"):
        # For step-wise methods, return nominal alpha
        # Actual correction happens during procedure
        return 0.05
    else:
        raise ValueError(f"Unknown correction method: {method}")


def compute_effect_sizes(
    distances: NDArray[np.float64],
    null_mean: float,
    null_std: float,
) -> NDArray[np.float64]:
    """
    Compute Cohen's d effect sizes for distances.

    d = (observed - null_mean) / null_std

    Interpretation:
    - |d| < 0.2: Small effect
    - 0.2 ≤ |d| < 0.8: Medium effect
    - |d| ≥ 0.8: Large effect

    Args:
        distances: Observed distances.
        null_mean: Mean under null hypothesis.
        null_std: Standard deviation under null hypothesis.

    Returns:
        Array of effect sizes.
    """
    if null_std == 0:
        return np.where(distances == null_mean, 0.0, np.inf)
    return (distances - null_mean) / null_std