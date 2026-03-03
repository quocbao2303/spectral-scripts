"""Visualization tools for spectral analysis."""

from spectral_scripts.visualization.spectra import (
    plot_spectrum,
    plot_spectrum_comparison,
    plot_cumulative_spectra,
)
from spectral_scripts.visualization.heatmaps import (
    plot_confusion_matrix,
    plot_distance_matrix,
    plot_distance_matrix_clustered,
)
from spectral_scripts.visualization.validation import (
    plot_synthetic_validation,
    plot_bootstrap_distribution,
    plot_permutation_null,
)

__all__ = [
    "plot_spectrum",
    "plot_spectrum_comparison",
    "plot_cumulative_spectra",
    "plot_confusion_matrix",
    "plot_distance_matrix",
    "plot_distance_matrix_clustered",
    "plot_synthetic_validation",
    "plot_bootstrap_distribution",
    "plot_permutation_null",
]