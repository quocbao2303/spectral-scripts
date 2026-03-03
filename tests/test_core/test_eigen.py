"""Tests for eigendecomposition utilities."""

import numpy as np
import pytest

from spectral_scripts.core.eigen import EigenResult, compute_eigen
from spectral_scripts.core.normalization import row_normalize, bistochastic_normalize


class TestComputeEigen:
    """Tests for compute_eigen function."""
    
    def test_stochastic_matrix_properties(self):
        """Test eigenvalues of stochastic matrix."""
        # Create a row-stochastic matrix
        matrix = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
        result = compute_eigen(matrix, source_type="stochastic")
        
        # Largest eigenvalue should be 1
        assert np.isclose(result.magnitudes[0], 1.0, atol=1e-6)
        
        # All magnitudes should be <= 1
        assert np.all(result.magnitudes <= 1.0 + 1e-6)
    
    def test_bistochastic_matrix_properties(self):
        """Test eigenvalues of bistochastic matrix."""
        matrix = np.random.rand(5, 5) + 0.1
        bistochastic = bistochastic_normalize(matrix)
        
        result = compute_eigen(bistochastic, source_type="bistochastic")
        
        # Largest eigenvalue should be 1
        assert np.isclose(result.magnitudes[0], 1.0, atol=1e-6)
    
    def test_symmetric_matrix_real_eigenvalues(self):
        """Test that symmetric matrices have real eigenvalues."""
        matrix = np.random.rand(5, 5)
        matrix = (matrix + matrix.T) / 2  # Symmetrize
        
        result = compute_eigen(matrix, source_type="symmetric")
        
        assert result.is_real
    
    def test_magnitudes_sorted_descending(self):
        """Test that magnitudes are sorted in descending order."""
        matrix = np.random.rand(5, 5)
        result = compute_eigen(matrix, source_type="general", verify=False)
        
        assert np.all(np.diff(result.magnitudes) <= 1e-10)
    
    def test_spectral_gap(self):
        """Test spectral gap calculation."""
        # Create matrix with known spectral gap
        matrix = np.array([[0.9, 0.1], [0.1, 0.9]])
        result = compute_eigen(matrix, source_type="stochastic")
        
        # λ₁ = 1, λ₂ = 0.8, gap = 1 - 0.8 = 0.2
        assert np.isclose(result.spectral_gap, 0.2, atol=1e-6)


class TestEigenResult:
    """Tests for EigenResult dataclass."""
    
    def test_top_k(self):
        """Test top_k returns correct number of eigenvalues."""
        eigenvalues = np.array([1, 0.5, 0.3, 0.1, 0.05], dtype=np.complex128)
        eigenvectors = np.eye(5, dtype=np.complex128)
        magnitudes = np.abs(eigenvalues)
        
        result = EigenResult(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            magnitudes=magnitudes,
            source_type="general",
        )
        
        top3 = result.top_k(3)
        assert len(top3) == 3
        np.testing.assert_array_equal(top3, [1, 0.5, 0.3])
    
    def test_normalized_spectrum(self):
        """Test normalized spectrum sums to 1."""
        eigenvalues = np.array([1, 0.5, 0.25], dtype=np.complex128)
        eigenvectors = np.eye(3, dtype=np.complex128)
        magnitudes = np.abs(eigenvalues)
        
        result = EigenResult(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            magnitudes=magnitudes,
            source_type="general",
        )
        
        normalized = result.normalized_spectrum()
        assert np.isclose(normalized.sum(), 1.0)
    
    def test_cumulative_spectrum(self):
        """Test cumulative spectrum is monotonically increasing."""
        eigenvalues = np.array([1, 0.5, 0.25, 0.125], dtype=np.complex128)
        eigenvectors = np.eye(4, dtype=np.complex128)
        magnitudes = np.abs(eigenvalues)
        
        result = EigenResult(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            magnitudes=magnitudes,
            source_type="general",
        )
        
        cumulative = result.cumulative_spectrum()
        
        assert np.all(np.diff(cumulative) >= 0)
        assert np.isclose(cumulative[-1], 1.0)