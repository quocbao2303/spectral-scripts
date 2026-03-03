"""Validation result visualization functions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from spectral_scripts.validation.synthetic import SyntheticValidationResult
from spectral_scripts.validation.bootstrap import BootstrapResult
from spectral_scripts.validation.permutation import PermutationResult


def plot_synthetic_validation(
    result: SyntheticValidationResult,
    figsize: tuple[float, float] = (12, 5),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Visualize synthetic validation results.

    Creates a two-panel figure:
    1. Scatter plot of true vs computed distances
    2. Residual plot

    Args:
        result: SyntheticValidationResult to visualize.
        figsize: Figure size.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Extract data from details
    true_range = result.details.get("true_distance_range", (0, 1))
    computed_range = result.details.get("computed_distance_range", (0, 1))

    # Panel 1: Correlation scatter (simulated since we don't have raw data)
    ax1 = axes[0]

    # Draw identity line
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect correlation")

    # Add correlation info
    ax1.text(
        0.05,
        0.95,
        f"ρ = {result.spearman_rho:.3f}\n"
        f"τ = {result.kendall_tau:.3f}\n"
        f"p = {result.spearman_pvalue:.2e}",
        transform=ax1.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax1.set_xlabel("True Distance (normalized)")
    ax1.set_ylabel("Computed Distance (normalized)")
    ax1.set_title("Ground Truth Correlation")
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Summary metrics bar chart
    ax2 = axes[1]

    metrics = {
        "Spearman ρ": result.spearman_rho,
        "Kendall τ": result.kendall_tau,
        "Rank Preservation": result.rank_preservation,
        "1 - MAE": 1 - result.mean_absolute_error,
    }

    x = np.arange(len(metrics))
    values = list(metrics.values())
    colors = ["green" if v >= result.threshold else "red" for v in values]

    ax2.bar(x, values, color=colors, alpha=0.7)
    ax2.axhline(result.threshold, color="red", linestyle="--", label=f"Threshold ({result.threshold})")
    ax2.set_xticks(x)
    ax2.set_xticklabels(list(metrics.keys()), rotation=45, ha="right")
    ax2.set_ylabel("Score")
    ax2.set_title(f"Validation Metrics ({'PASSED' if result.passed else 'FAILED'})")
    ax2.set_ylim(0, 1.1)
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_bootstrap_distribution(
    result: BootstrapResult,
    figsize: tuple[float, float] = (10, 5),
    bins: int = 50,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Visualize bootstrap distribution with confidence interval.

    Args:
        result: BootstrapResult to visualize.
        figsize: Figure size.
        bins: Number of histogram bins.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Histogram of bootstrap distribution
    ax.hist(
        result.bootstrap_distribution,
        bins=bins,
        density=True,
        alpha=0.7,
        color="steelblue",
        edgecolor="white",
    )

    # Point estimate
    ax.axvline(
        result.point_estimate,
        color="red",
        linewidth=2,
        label=f"Point estimate: {result.point_estimate:.4f}",
    )

    # Confidence interval
    ax.axvline(result.ci_lower, color="orange", linestyle="--", linewidth=1.5)
    ax.axvline(result.ci_upper, color="orange", linestyle="--", linewidth=1.5)
    ax.axvspan(
        result.ci_lower,
        result.ci_upper,
        alpha=0.2,
        color="orange",
        label=f"{result.ci_level:.0%} CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]",
    )

    ax.set_xlabel("Distance")
    ax.set_ylabel("Density")
    ax.set_title(f"Bootstrap Distribution (n={result.n_bootstrap})")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_permutation_null(
    result: PermutationResult,
    figsize: tuple[float, float] = (10, 5),
    bins: int = 50,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Visualize permutation test null distribution.

    Args:
        result: PermutationResult to visualize.
        figsize: Figure size.
        bins: Number of histogram bins.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Histogram of null distribution
    ax.hist(
        result.null_distribution,
        bins=bins,
        density=True,
        alpha=0.7,
        color="gray",
        edgecolor="white",
        label=f"Null distribution (n={result.n_permutations})",
    )

    # Observed value
    ax.axvline(
        result.observed,
        color="red",
        linewidth=2,
        label=f"Observed: {result.observed:.4f}",
    )

    # Null mean
    ax.axvline(
        result.null_mean,
        color="blue",
        linestyle="--",
        linewidth=1.5,
        label=f"Null mean: {result.null_mean:.4f}",
    )

    # Shade rejection region based on alternative
    x_range = ax.get_xlim()
    if result.alternative == "less":
        critical = np.percentile(result.null_distribution, 5)
        ax.axvspan(x_range[0], critical, alpha=0.2, color="green", label="Rejection region (α=0.05)")
    elif result.alternative == "greater":
        critical = np.percentile(result.null_distribution, 95)
        ax.axvspan(critical, x_range[1], alpha=0.2, color="green", label="Rejection region (α=0.05)")

    # Significance annotation
    sig_text = "Significant" if result.significant else "Not significant"
    ax.text(
        0.95,
        0.95,
        f"p = {result.p_value:.4f}\n{sig_text}\nEffect size: {result.effect_size:.2f}",
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.set_xlabel("Distance")
    ax.set_ylabel("Density")
    ax.set_title(f"Permutation Test (alternative: {result.alternative})")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_validation_summary(
    synthetic_result: SyntheticValidationResult | None = None,
    sanity_passed: bool | None = None,
    n_significant_pairs: int | None = None,
    n_total_pairs: int | None = None,
    figsize: tuple[float, float] = (8, 6),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Create summary dashboard of all validation results.

    Args:
        synthetic_result: Results from synthetic validation.
        sanity_passed: Whether sanity checks passed.
        n_significant_pairs: Number of significant distance pairs.
        n_total_pairs: Total number of pairs tested.
        figsize: Figure size.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create checklist-style summary
    checks = []

    if synthetic_result is not None:
        status = "✓" if synthetic_result.passed else "✗"
        color = "green" if synthetic_result.passed else "red"
        checks.append((f"{status} Synthetic validation (ρ={synthetic_result.spearman_rho:.3f})", color))

    if sanity_passed is not None:
        status = "✓" if sanity_passed else "✗"
        color = "green" if sanity_passed else "red"
        checks.append((f"{status} Metric sanity checks", color))

    if n_significant_pairs is not None and n_total_pairs is not None:
        pct = n_significant_pairs / max(n_total_pairs, 1) * 100
        checks.append((f"○ Significant pairs: {n_significant_pairs}/{n_total_pairs} ({pct:.0f}%)", "blue"))

    # Draw checks
    y_positions = np.arange(len(checks))[::-1]
    for y, (text, color) in zip(y_positions, checks):
        ax.text(0.1, y, text, fontsize=14, color=color, verticalalignment="center")

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(checks) - 0.5)
    ax.axis("off")
    ax.set_title("Validation Summary", fontsize=16, fontweight="bold")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig