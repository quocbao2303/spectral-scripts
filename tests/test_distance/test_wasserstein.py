"""Tests for Wasserstein distance computations."""

import numpy as np
import pytest

from spectral_scripts.distance.wasserstein import (
    wasserstein_1d,
    cumulative_wasserstein,
    spectral_distance,
    multi_spectrum_distance,
)
from spectral_scripts.features.profile import extract_profile


class TestWasserstein1D:
    """Tests for 1D Wasserstein distance."""
    
    def test_identical_distributions(self):
        """Test that identical distributions have zero distance."""
        p = np.array([0.3, 0.4, 0.2, 0.1])
        
        dist = wasserstein_1d(p, p)
        
        assert np.isclose(dist, 0, atol=1e-10)
    
    def test_symmetric(self):
        """Test that distance is symmetric."""
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.2, 0.5, 0.3])
        
        assert np.isclose(wasserstein_1d(p, q), wasserstein_1d(q, p))
    
    def test_non_negative(self):
        """Test that distance is non-negative."""
        rng = np.random.default_rng(42)
        
        for _ in range(10):
            p = rng.random(10)
            q = rng.random(10)
            
            dist = wasserstein_1d(p, q)
            assert dist >= 0
    
    def test_normalization(self):
        """Test that normalization works correctly."""
        p = np.array([10, 20, 30])  # Sum = 60
        q = np.array([5, 10, 15])   # Sum = 30
        
        # Should normalize to same total
        dist = wasserstein_1d(p, q, normalize=True)
        
        # After normalization, both are [1/6, 2/6, 3/6]
        # So distance should be 0
        assert np.isclose(dist, 0, atol=1e-10)
    
    def test_different_lengths_raises(self):
        """Test that different lengths raise error."""
        p = np.array([0.5, 0.5])
        q = np.array([0.33, 0.33, 0.34])
        
        with pytest.raises(ValueError, match="same length"):
            wasserstein_1d(p, q)
    
    def test_empty_arrays(self):
        """Test that empty arrays return zero distance."""
        p = np.array([])
        q = np.array([])
        
        dist = wasserstein_1d(p, q)
        assert dist == 0.0


class TestCumulativeWasserstein:
    """Tests for cumulative Wasserstein distance."""
    
    def test_identical_spectra(self):
        """Test that identical spectra have zero distance."""
        spectrum = np.array([1.0, 0.8, 0.5, 0.3, 0.1])
        
        dist = cumulative_wasserstein(spectrum, spectrum)
        
        assert np.isclose(dist, 0, atol=1e-10)
    
    def test_handles_different_lengths(self):
        """Test that different length spectra are handled."""
        s1 = np.array([1.0, 0.8, 0.5])
        s2 = np.array([1.0, 0.7, 0.4, 0.2, 0.1])
        
        # Should not raise
        dist = cumulative_wasserstein(s1, s2)
        
        assert dist >= 0
    
    def test_truncation(self):
        """Test spectrum truncation."""
        s1 = np.array([1.0, 0.8, 0.5, 0.3, 0.1])
        s2 = np.array([1.0, 0.7, 0.4, 0.2, 0.05])
        
        dist_full = cumulative_wasserstein(s1, s2, truncate_to=None)
        dist_truncated = cumulative_wasserstein(s1, s2, truncate_to=3)
        
        # Truncated should be different (generally)
        # Both should be valid distances
        assert dist_full >= 0
        assert dist_truncated >= 0
    
    def test_padding_with_zeros(self):
        """Test that shorter spectrum is padded with zeros."""
        s1 = np.array([1.0, 0.5])
        s2 = np.array([1.0, 0.5, 0.0, 0.0])  # Equivalent with padding
        
        # After padding, s1 becomes [1.0, 0.5, 0.0, 0.0]
        dist = cumulative_wasserstein(s1, s2)
        
        assert np.isclose(dist, 0, atol=1e-10)


class TestSpectralDistance:
    """Tests for spectral distance function."""
    
    def test_self_distance_zero(self, simple_confusion_matrix):
        """Test that distance to self is zero."""
        profile = extract_profile(simple_confusion_matrix)
        
        dist = spectral_distance(profile.spectral, profile.spectral)
        
        assert np.isclose(dist, 0, atol=1e-10)
    
    def test_symmetric(self, two_similar_matrices):
        """Test that distance is symmetric."""
        cm1, cm2 = two_similar_matrices
        p1 = extract_profile(cm1)
        p2 = extract_profile(cm2)
        
        dist1 = spectral_distance(p1.spectral, p2.spectral)
        dist2 = spectral_distance(p2.spectral, p1.spectral)
        
        assert np.isclose(dist1, dist2)
    
    def test_different_matrices_positive_distance(self, two_different_matrices):
        """Test that different matrices have positive distance."""
        cm1, cm2 = two_different_matrices
        p1 = extract_profile(cm1)
        p2 = extract_profile(cm2)
        
        dist = spectral_distance(p1.spectral, p2.spectral)
        
        assert dist > 0
    
    def test_similar_closer_than_different(
        self, two_similar_matrices, two_different_matrices
    ):
        """Test that similar matrices are closer than different ones."""
        sim1, sim2 = two_similar_matrices
        diff1, diff2 = two_different_matrices
        
        p_sim1 = extract_profile(sim1)
        p_sim2 = extract_profile(sim2)
        p_diff1 = extract_profile(diff1)
        p_diff2 = extract_profile(diff2)
        
        similar_dist = spectral_distance(p_sim1.spectral, p_sim2.spectral)
        different_dist = spectral_distance(p_diff1.spectral, p_diff2.spectral)
        
        assert similar_dist < different_dist
    
    def test_spectrum_type_options(self, simple_confusion_matrix):
        """Test different spectrum type options."""
        profile = extract_profile(simple_confusion_matrix)
        
        # Should not raise for any valid type
        for spectrum_type in ["bistochastic", "symmetric", "laplacian"]:
            dist = spectral_distance(
                profile.spectral, profile.spectral, 
                spectrum_type=spectrum_type
            )
            assert np.isclose(dist, 0, atol=1e-10)
    
    def test_invalid_spectrum_type(self, simple_confusion_matrix):
        """Test that invalid spectrum type raises error."""
        profile = extract_profile(simple_confusion_matrix)
        
        with pytest.raises(ValueError, match="Unknown spectrum_type"):
            spectral_distance(
                profile.spectral, profile.spectral,
                spectrum_type="invalid"
            )


class TestMultiSpectrumDistance:
    """Tests for multi-spectrum distance function."""
    
    def test_self_distance_zero(self, simple_confusion_matrix):
        """Test that distance to self is zero."""
        profile = extract_profile(simple_confusion_matrix)
        
        dist = multi_spectrum_distance(profile.spectral, profile.spectral)
        
        assert np.isclose(dist, 0, atol=1e-10)
    
    def test_custom_weights(self, two_similar_matrices):
        """Test with custom spectrum weights."""
        cm1, cm2 = two_similar_matrices
        p1 = extract_profile(cm1)
        p2 = extract_profile(cm2)
        
        # All weight on bistochastic
        dist_bisto = multi_spectrum_distance(
            p1.spectral, p2.spectral,
            weights={"bistochastic": 1.0, "symmetric": 0.0, "laplacian": 0.0}
        )
        
        # All weight on laplacian
        dist_lapl = multi_spectrum_distance(
            p1.spectral, p2.spectral,
            weights={"bistochastic": 0.0, "symmetric": 0.0, "laplacian": 1.0}
        )
        
        # Distances can be different
        assert dist_bisto >= 0
        assert dist_lapl >= 0
    
    def test_weights_normalized(self, two_similar_matrices):
        """Test that unnormalized weights are handled."""
        cm1, cm2 = two_similar_matrices
        p1 = extract_profile(cm1)
        p2 = extract_profile(cm2)
        
        # Weights that don't sum to 1
        dist = multi_spectrum_distance(
            p1.spectral, p2.spectral,
            weights={"bistochastic": 2.0, "symmetric": 1.0, "laplacian": 2.0}
        )
        
        assert dist >= 0