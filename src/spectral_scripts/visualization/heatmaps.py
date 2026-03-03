"""Heatmap visualization functions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy.typing import NDArray
from scipy.cluster import hierarchy

from spectral_scripts.core.confusion_matrix import ConfusionMatrix
from spectral_scripts.distance.matrix import DistanceMatrix


def plot_confusion_matrix(
    confusion: ConfusionMatrix,
    normalize: bool = True,
    show_values: bool = False,
    figsize: tuple[float, float] = (10, 8),
    cmap: str = "Blues",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot confusion matrix as heatmap.

    Args:
        confusion: ConfusionMatrix to plot.
        normalize: If True, normalize rows to sum to 1.
        show_values: If True, annotate cells with values.
        figsize: Figure size.
        cmap: Colormap name.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    matrix = confusion.matrix.copy()
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        matrix = matrix / row_sums
        fmt = ".2f"
        vmax = 1.0
    else:
        fmt = ".0f"
        vmax = None

    sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        vmax=vmax,
        annot=show_values,
        fmt=fmt,
        square=True,
        xticklabels=confusion.characters,
        yticklabels=confusion.characters,
        cbar_kws={"label": "Probability" if normalize else "Count"},
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix: {confusion.script}")

    # Only show tick labels if reasonable number of characters
    if confusion.size > 30:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_distance_matrix(
    distance_matrix: DistanceMatrix,
    figsize: tuple[float, float] = (10, 8),
    cmap: str = "viridis",
    annotate: bool = True,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot pairwise distance matrix as heatmap.

    Args:
        distance_matrix: DistanceMatrix to plot.
        figsize: Figure size.
        cmap: Colormap name.
        annotate: If True, show distance values in cells.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        distance_matrix.distances,
        ax=ax,
        cmap=cmap,
        annot=annotate,
        fmt=".3f",
        square=True,
        xticklabels=distance_matrix.scripts,
        yticklabels=distance_matrix.scripts,
        cbar_kws={"label": "Distance"},
    )

    ax.set_title(f"Distance Matrix ({distance_matrix.method})")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_distance_matrix_clustered(
    distance_matrix: DistanceMatrix,
    method: str = "average",
    figsize: tuple[float, float] = (12, 10),
    cmap: str = "viridis",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot distance matrix with hierarchical clustering dendrogram.

    Args:
        distance_matrix: DistanceMatrix to plot.
        method: Linkage method for clustering.
        figsize: Figure size.
        cmap: Colormap name.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib figure.
    """
    # Compute linkage
    condensed = distance_matrix.to_condensed()
    linkage = hierarchy.linkage(condensed, method=method)

    # Create clustered heatmap
    fig = plt.figure(figsize=figsize)

    # Use clustermap from seaborn
    g = sns.clustermap(
        distance_matrix.distances,
        row_linkage=linkage,
        col_linkage=linkage,
        xticklabels=distance_matrix.scripts,
        yticklabels=distance_matrix.scripts,
        cmap=cmap,
        figsize=figsize,
        cbar_kws={"label": "Distance"},
        annot=True,
        fmt=".3f",
    )

    g.ax_heatmap.set_title(f"Clustered Distance Matrix ({distance_matrix.method})")

    if save_path:
        g.savefig(save_path, dpi=150, bbox_inches="tight")

    return g.fig


def plot_distance_ranking(
    distance_matrix: DistanceMatrix,
    reference_script: str,
    figsize: tuple[float, float] = (10, 6),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot scripts ranked by distance from a reference script.

    Args:
        distance_matrix: DistanceMatrix to analyze.
        reference_script: Script to use as reference point.
        figsize: Figure size.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    ranked = distance_matrix.rank_by_distance(reference_script)
    scripts = [r[0] for r in ranked]
    distances = [r[1] for r in ranked]

    y_pos = np.arange(len(scripts))
    ax.barh(y_pos, distances, color="steelblue", alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(scripts)
    ax.set_xlabel("Distance")
    ax.set_title(f"Scripts Ranked by Distance from {reference_script}")
    ax.grid(True, alpha=0.3, axis="x")

    # Add value labels
    for i, (script, dist) in enumerate(ranked):
        ax.text(dist + 0.01, i, f"{dist:.3f}", va="center", fontsize=9)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig