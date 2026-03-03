"""Spectrum visualization functions."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from spectral_scripts.features.profile import SpectralProfile


def plot_spectrum(
    profile: SpectralProfile,
    spectrum_type: str = "bistochastic",
    ax: plt.Axes | None = None,
    color: str | None = None,
    label: str | None = None,
    show_gap: bool = True,
    **kwargs,
) -> plt.Axes:
    """
    Plot eigenvalue spectrum for a single profile.

    Args:
        profile: SpectralProfile to plot.
        spectrum_type: Which spectrum to plot:
            - "bistochastic": Bistochastic matrix eigenvalues
            - "symmetric": Symmetrized bistochastic eigenvalues
            - "laplacian": Laplacian eigenvalues
        ax: Matplotlib axes. Creates new figure if None.
        color: Line color. Uses default cycle if None.
        label: Legend label. Uses script name if None.
        show_gap: If True, annotate spectral gap.
        **kwargs: Additional arguments to plt.plot().

    Returns:
        Matplotlib axes with the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    # Select spectrum
    if spectrum_type == "bistochastic":
        spectrum = profile.spectral.bistochastic_spectrum
        gap = profile.spectral.bistochastic_gap
        ylabel = "Eigenvalue Magnitude"
    elif spectrum_type == "symmetric":
        spectrum = profile.spectral.symmetric_spectrum
        gap = profile.spectral.symmetric_gap
        ylabel = "Eigenvalue"
    elif spectrum_type == "laplacian":
        spectrum = profile.spectral.laplacian_spectrum
        gap = None
        ylabel = "Laplacian Eigenvalue"
    else:
        raise ValueError(f"Unknown spectrum_type: {spectrum_type}")

    # Plot
    x = np.arange(1, len(spectrum) + 1)
    label = label or profile.script
    ax.plot(x, spectrum, marker="o", markersize=3, label=label, color=color, **kwargs)

    # Annotate spectral gap
    if show_gap and gap is not None and len(spectrum) >= 2:
        ax.axhline(spectrum[1], linestyle="--", alpha=0.5, color=color or "gray")
        ax.annotate(
            f"Gap = {gap:.3f}",
            xy=(2, spectrum[1]),
            xytext=(5, spectrum[1] + 0.05),
            fontsize=9,
            alpha=0.8,
        )

    ax.set_xlabel("Eigenvalue Index")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{spectrum_type.title()} Spectrum: {profile.script}")
    ax.grid(True, alpha=0.3)

    return ax


def plot_spectrum_comparison(
    profiles: Sequence[SpectralProfile],
    spectrum_type: str = "bistochastic",
    top_k: int = 20,
    figsize: tuple[float, float] = (12, 6),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Compare spectra of multiple profiles.

    Args:
        profiles: Sequence of SpectralProfiles to compare.
        spectrum_type: Which spectrum to plot.
        top_k: Number of top eigenvalues to show.
        figsize: Figure size.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for profile in profiles:
        plot_spectrum(
            profile,
            spectrum_type=spectrum_type,
            ax=ax,
            show_gap=False,
        )

    ax.set_xlim(0.5, top_k + 0.5)
    ax.legend(loc="upper right")
    ax.set_title(f"{spectrum_type.title()} Spectrum Comparison")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_cumulative_spectra(
    profiles: Sequence[SpectralProfile],
    spectrum_type: str = "bistochastic",
    figsize: tuple[float, float] = (10, 6),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot cumulative distribution functions of spectra.

    Useful for visualizing the Wasserstein distance interpretation.

    Args:
        profiles: Sequence of SpectralProfiles.
        spectrum_type: Which spectrum to use.
        figsize: Figure size.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for profile in profiles:
        if spectrum_type == "bistochastic":
            cdf = profile.spectral.cumulative_bistochastic_spectrum()
        elif spectrum_type == "laplacian":
            cdf = profile.spectral.cumulative_laplacian_spectrum()
        else:
            raise ValueError(f"Unknown spectrum_type: {spectrum_type}")

        x = np.arange(1, len(cdf) + 1)
        ax.step(x, cdf, where="mid", label=profile.script)

    ax.set_xlabel("Eigenvalue Index")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title(f"Cumulative {spectrum_type.title()} Spectrum")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_spectral_features_comparison(
    profiles: Sequence[SpectralProfile],
    figsize: tuple[float, float] = (12, 8),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Compare scalar spectral features across profiles.

    Creates a grouped bar chart of key spectral features.

    Args:
        profiles: Sequence of SpectralProfiles.
        figsize: Figure size.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.ravel()

    features = [
        ("bistochastic_gap", "Bistochastic Gap"),
        ("bistochastic_entropy", "Bistochastic Entropy"),
        ("bistochastic_effective_rank", "Effective Rank"),
        ("symmetric_gap", "Symmetric Gap"),
        ("fiedler_value", "Fiedler Value"),
        ("laplacian_entropy", "Laplacian Entropy"),
    ]

    scripts = [p.script for p in profiles]
    x = np.arange(len(scripts))
    width = 0.6

    for ax, (attr, title) in zip(axes, features):
        values = [getattr(p.spectral, attr) for p in profiles]
        ax.bar(x, values, width, color="steelblue", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(scripts, rotation=45, ha="right")
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig