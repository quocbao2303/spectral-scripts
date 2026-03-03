"""Tests for spectral feature extraction."""

import numpy as np
import pytest

from spectral_scripts.core.confusion_matrix import ConfusionMatrix
from spectral_scripts.features.spectral import (
    SpectralFeatures,
    extract_spectral_features,
    compute_spectral_entropy,
    compute_effective_rank,
)


class TestComputeSpectralEntropy:
    """Tests for spectral entropy computation."""
    
    def test_uniform_distribution_max_entropy(self):
        """Test that uniform distribution has maximum entropy."""
        n = 10
        uniform = np.ones(n) / n
        entropy = compute_spectral_entropy(uniform)
        
        # Max entropy for n items is log(n)
        max_entropy = np.log(n)
        assert np.isclose(entropy, max_entropy, atol=0.1)
    
    def test_concentrated_distribution_low_entropy(self):
        """Test that concentrated distribution has low entropy."""
        concentrated = np.array([0.99, 0.01, 0, 0, 0])
        entropy = compute_spectral_entropy(concentrated)
        
        # Should be close to 0
        assert entropy < 0.5
    
    def test_single_element_zero_entropy(self):
        """Test that single non-zero element has zero entropy."""
        single = np.array([1.0, 0, 0, 0])
        entropy = compute_spectral_entropy(single)
        
        assert np.isclose(entropy, 0, atol=1e-10)
    
    def test_empty_array_zero_entropy(self):
        """Test that empty array returns zero entropy."""
        empty = np.array([])
        entropy = compute_spectral_entropy(empty)
        
        assert entropy == 0.0


class TestComputeEffectiveRank:
    """Tests for effective rank computation."""
    
    def test_max_entropy_gives_n(self):
        """Test that maximum entropy gives effective rank n."""
        n = 10
        max_entropy = np.log(n)
        eff_rank = compute_effective_rank(max_entropy)
        
        assert np.isclose(eff_rank, n)
    
    def test_zero_entropy_gives_one(self):
        """Test that zero entropy gives effective rank 1."""
        eff_rank = compute_effective_rank(0)
        
        assert np.isclose(eff_rank, 1)
    
    def test_intermediate_entropy(self):
        """Test intermediate entropy gives reasonable rank."""
        # Entropy of 2 should give exp(2) ≈ 7.4
        eff_rank = compute_effective_rank(2.0)
        
        assert np.isclose(eff_rank, np.exp(2))


class TestExtractSpectralFeatures:
    """Tests for spectral feature extraction."""
    
    def test_basic_extraction(self, simple_confusion_matrix):
        """Test that basic extraction works."""
        features = extract_spectral_features(simple_confusion_matrix)
        
        assert features.script == simple_confusion_matrix.script
        assert features.matrix_size == simple_confusion_matrix.size
        assert len(features.bistochastic_spectrum) > 0
        assert len(features.symmetric_spectrum) > 0
        assert len(features.laplacian_spectrum) > 0
    
    def test_bistochastic_gap_range(self, simple_confusion_matrix):
        """Test that bistochastic gap is in valid range."""
        features = extract_spectral_features(simple_confusion_matrix)
        
        # Gap = 1 - |λ₂|, should be in [0, 1]
        assert 0 <= features.bistochastic_gap <= 1
    
    def test_fiedler_value_non_negative(self, simple_confusion_matrix):
        """Test that Fiedler value is non-negative."""
        features = extract_spectral_features(simple_confusion_matrix)
        
        # Fiedler value (algebraic connectivity) should be ≥ 0
        assert features.fiedler_value >= -1e-10
    
    def test_effective_rank_reasonable(self, simple_confusion_matrix):
        """Test that effective rank is reasonable."""
        features = extract_spectral_features(simple_confusion_matrix)
        
        # Effective rank should be between 1 and matrix size
        assert 1 <= features.bistochastic_effective_rank <= simple_confusion_matrix.size
    
    def test_identity_matrix_high_gap(self, identity_confusion_matrix):
        """Test that identity matrix has low spectral gap (all eigenvalues ≈ 1 after Sinkhorn)."""
        features = extract_spectral_features(identity_confusion_matrix)
        
        # After Sinkhorn normalization, identity stays doubly stochastic.
        # All eigenvalues ≈ 1 → gap = 1 - |λ₂| ≈ 0
        assert features.bistochastic_gap < 0.5
    
    def test_uniform_matrix_low_gap(self, uniform_confusion_matrix):
        """Test that uniform matrix has high spectral gap (rank-1 after Sinkhorn)."""
        features = extract_spectral_features(uniform_confusion_matrix)
        
        # Uniform (all-ones) → rank-1 doubly stochastic.
        # λ₁ = 1, all other λ ≈ 0 → gap = 1 - 0 ≈ 1.0
        assert features.bistochastic_gap > 0.5
    
    def test_spectrum_truncation(self, realistic_confusion_matrix):
        """Test that spectrum is truncated correctly."""
        max_len = 10
        features = extract_spectral_features(
            realistic_confusion_matrix, 
            max_spectrum_length=max_len
        )
        
        assert len(features.bistochastic_spectrum) <= max_len
        assert len(features.laplacian_spectrum) <= max_len
    
    def test_to_feature_vector(self, simple_confusion_matrix):
        """Test conversion to feature vector."""
        features = extract_spectral_features(simple_confusion_matrix)
        vector = features.to_feature_vector()
        
        assert vector.ndim == 1
        assert len(vector) == 6  # Number of scalar features
        assert vector.dtype == np.float64
    
    def test_normalized_spectrum_sums_to_one(self, simple_confusion_matrix):
        """Test that normalized spectrum sums to 1."""
        features = extract_spectral_features(simple_confusion_matrix)
        
        normalized = features.normalized_bistochastic_spectrum()
        assert np.isclose(normalized.sum(), 1.0)
    
    def test_cumulative_spectrum_monotonic(self, simple_confusion_matrix):
        """Test that cumulative spectrum is monotonically increasing."""
        features = extract_spectral_features(simple_confusion_matrix)
        
        cumulative = features.cumulative_bistochastic_spectrum()
        assert np.all(np.diff(cumulative) >= -1e-10)
        assert np.isclose(cumulative[-1], 1.0)


class TestSpectralFeaturesDataclass:
    """Tests for SpectralFeatures dataclass methods."""
    
    def test_to_dict(self, simple_confusion_matrix):
        """Test conversion to dictionary."""
        features = extract_spectral_features(simple_confusion_matrix)
        d = features.to_dict()
        
        assert "script" in d
        assert "bistochastic_spectrum" in d
        assert "bistochastic_gap" in d
        assert isinstance(d["bistochastic_spectrum"], list)