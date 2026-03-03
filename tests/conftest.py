"""Shared test fixtures for spectral_scripts tests."""

from __future__ import annotations

import numpy as np
import pytest

from spectral_scripts.core.confusion_matrix import ConfusionMatrix
from spectral_scripts.features.profile import SpectralProfile, extract_profile


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def simple_confusion_matrix() -> ConfusionMatrix:
    """Simple 5x5 confusion matrix for basic tests."""
    matrix = np.array([
        [100, 5, 2, 1, 0],
        [3, 90, 4, 2, 1],
        [1, 3, 85, 8, 3],
        [2, 1, 6, 88, 3],
        [0, 1, 3, 2, 94],
    ], dtype=np.float64)
    
    return ConfusionMatrix(
        matrix=matrix,
        script="test_simple",
        characters=["a", "b", "c", "d", "e"],
    )


@pytest.fixture
def identity_confusion_matrix() -> ConfusionMatrix:
    """Perfect accuracy confusion matrix (diagonal only)."""
    n = 10
    matrix = np.eye(n) * 100
    
    return ConfusionMatrix(
        matrix=matrix,
        script="test_identity",
        characters=[chr(ord("a") + i) for i in range(n)],
    )


@pytest.fixture
def uniform_confusion_matrix() -> ConfusionMatrix:
    """Uniform confusion matrix (maximum entropy)."""
    n = 10
    matrix = np.ones((n, n)) * 10
    
    return ConfusionMatrix(
        matrix=matrix,
        script="test_uniform",
        characters=[chr(ord("a") + i) for i in range(n)],
    )


@pytest.fixture
def sparse_confusion_matrix() -> ConfusionMatrix:
    """Sparse confusion matrix with few off-diagonal entries."""
    n = 20
    matrix = np.eye(n) * 100
    
    # Add a few confusions
    matrix[0, 1] = 5
    matrix[1, 0] = 3
    matrix[5, 6] = 8
    matrix[10, 11] = 4
    
    return ConfusionMatrix(
        matrix=matrix,
        script="test_sparse",
        characters=[chr(ord("a") + i) for i in range(n)],
    )


@pytest.fixture
def realistic_confusion_matrix(rng: np.random.Generator) -> ConfusionMatrix:
    """Realistic confusion matrix with typical OCR patterns."""
    n = 26  # Full alphabet
    
    # Start with high diagonal
    matrix = np.eye(n) * 1000
    
    # Add realistic confusions
    # Similar letters confuse more often
    confusions = [
        (0, 14, 50),   # a-o
        (1, 3, 30),    # b-d
        (2, 4, 25),    # c-e
        (6, 16, 40),   # g-q
        (8, 11, 35),   # i-l
        (12, 13, 45),  # m-n
        (17, 13, 20),  # r-n
        (20, 21, 55),  # u-v
        (22, 21, 30),  # w-v
    ]
    
    for i, j, count in confusions:
        matrix[i, j] = count
        matrix[j, i] = count * 0.8  # Slight asymmetry
    
    # Add some random noise
    noise = rng.exponential(scale=5, size=(n, n))
    noise = np.triu(noise) + np.triu(noise, 1).T  # Make symmetric-ish
    np.fill_diagonal(noise, 0)
    matrix += noise
    
    return ConfusionMatrix(
        matrix=matrix,
        script="test_realistic",
        characters=[chr(ord("a") + i) for i in range(n)],
    )


@pytest.fixture
def sample_profiles(
    simple_confusion_matrix: ConfusionMatrix,
    identity_confusion_matrix: ConfusionMatrix,
    uniform_confusion_matrix: ConfusionMatrix,
) -> list[SpectralProfile]:
    """List of sample profiles for distance tests."""
    return [
        extract_profile(simple_confusion_matrix),
        extract_profile(identity_confusion_matrix),
        extract_profile(uniform_confusion_matrix),
    ]


@pytest.fixture
def two_similar_matrices(rng: np.random.Generator) -> tuple[ConfusionMatrix, ConfusionMatrix]:
    """Two similar confusion matrices for comparison tests."""
    n = 15
    
    # Base matrix
    base = np.eye(n) * 500
    
    # Add same confusion pattern
    for i in range(n - 1):
        base[i, i + 1] = 20 + rng.integers(0, 10)
        base[i + 1, i] = 15 + rng.integers(0, 10)
    
    # Create similar variant
    variant = base.copy()
    variant += rng.normal(0, 5, size=(n, n))
    variant = np.maximum(variant, 0)
    
    characters = [chr(ord("a") + i) for i in range(n)]
    
    cm1 = ConfusionMatrix(
        matrix=base,
        script="similar_1",
        characters=characters,
    )
    
    cm2 = ConfusionMatrix(
        matrix=variant,
        script="similar_2",
        characters=characters,
    )
    
    return cm1, cm2


@pytest.fixture
def two_different_matrices(rng: np.random.Generator) -> tuple[ConfusionMatrix, ConfusionMatrix]:
    """Two very different confusion matrices for comparison tests."""
    n = 15
    characters = [chr(ord("a") + i) for i in range(n)]
    
    # High accuracy, sparse confusions
    matrix1 = np.eye(n) * 1000
    matrix1[0, 1] = 20
    matrix1[1, 0] = 15
    
    # Lower accuracy, dense confusions
    matrix2 = np.eye(n) * 300
    matrix2 += rng.exponential(scale=30, size=(n, n))
    np.fill_diagonal(matrix2, np.diag(matrix2) + 200)
    
    cm1 = ConfusionMatrix(
        matrix=matrix1,
        script="different_1",
        characters=characters,
    )
    
    cm2 = ConfusionMatrix(
        matrix=matrix2,
        script="different_2",
        characters=characters,
    )
    
    return cm1, cm2