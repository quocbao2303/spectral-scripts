"""
Spectral Scripts: Spectral analysis of OCR confusion matrices.

This package provides tools for comparing writing systems through
spectral analysis of their OCR confusion patterns.
"""

__version__ = "0.1.0"

from spectral_scripts.core.confusion_matrix import ConfusionMatrix
from spectral_scripts.features.profile import SpectralProfile
from spectral_scripts.distance.matrix import compute_distance_matrix

__all__ = [
    "ConfusionMatrix",
    "SpectralProfile",
    "compute_distance_matrix",
]