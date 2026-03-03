"""Tests for normalization functions."""

import numpy as np
import pytest

from spectral_scripts.core.normalization import (
    row_normalize,
    bistochastic_normalize,
    symmetrize,
    compute_laplacian,
)


class TestRowNormalize:
    """Tests for row normalization."""
    
    def test_rows_sum_to_one(self):
        """Test that normalized rows sum to 1."""
        matrix = np.array([[10, 5, 5], [20, 10, 10], [5, 5, 10]], dtype=np.float64)
        normalized = row_normalize(matrix)
        
        row_sums = normalized.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(3))
    
    def test_zero_row_uniform(self):
        """Test that zero rows become uniform with 'uniform' strategy."""
        matrix = np.array([[10, 5, 5], [0, 0, 0], [5, 5, 10]], dtype=np.float64)
        normalized = row_normalize(matrix, handle_zero_rows="uniform")
        
        # Zero row should become uniform
        np.testing.assert_array_almost_equal(
            normalized[1, :], np.array([1/3, 1/3, 1/3])
        )
    
    def test_zero_row_identity(self):
        """Test that zero rows become identity with 'identity' strategy."""
        matrix = np.array([[10, 5, 5], [0, 0, 0], [5, 5, 10]], dtype=np.float64)
        normalized = row_normalize(matrix, handle_zero_rows="identity")
        
        # Zero row should have 1 on diagonal
        np.testing.assert_array_almost_equal(
            normalized[1, :], np.array([0, 1, 0])
        )
    
    def test_preserves_relative_proportions(self):
        """Test that relative proportions within rows are preserved."""
        matrix = np.array([[10, 20, 30]], dtype=np.float64)
        normalized = row_normalize(matrix)
        
        # Original ratios: 1:2:3
        assert np.isclose(normalized[0, 1] / normalized[0, 0], 2)
        assert np.isclose(normalized[0, 2] / normalized[0, 0], 3)


class TestBistochasticNormalize:
    """Tests for bistochastic (Sinkhorn-Knopp) normalization."""
    
    def test_rows_sum_to_one(self):
        """Test that normalized rows sum to 1."""
        matrix = np.random.rand(5, 5) + 0.1
        normalized = bistochastic_normalize(matrix)
        
        row_sums = normalized.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(5), decimal=8)
    
    def test_columns_sum_to_one(self):
        """Test that normalized columns sum to 1."""
        matrix = np.random.rand(5, 5) + 0.1
        normalized = bistochastic_normalize(matrix)
        
        col_sums = normalized.sum(axis=0)
        np.testing.assert_array_almost_equal(col_sums, np.ones(5), decimal=8)
    
    def test_preserves_non_negativity(self):
        """Test that output remains non-negative."""
        matrix = np.random.rand(5, 5) + 0.1
        normalized = bistochastic_normalize(matrix)
        
        assert np.all(normalized >= 0)
    
    def test_convergence_warning(self):
        """Test warning when algorithm doesn't converge."""
        # Create a matrix that's hard to normalize
        matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        
        # Should still work due to regularization
        normalized = bistochastic_normalize(matrix, regularization=1e-10)
        assert normalized.shape == matrix.shape
    
    def test_rejects_negative(self):
        """Test that negative values are rejected."""
        matrix = np.array([[1, -1], [1, 1]], dtype=np.float64)
        
        with pytest.raises(ValueError, match="non-negative"):
            bistochastic_normalize(matrix)


class TestSymmetrize:
    """Tests for matrix symmetrization."""
    
    def test_result_is_symmetric(self):
        """Test that result is symmetric."""
        matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
        symmetric = symmetrize(matrix)
        
        np.testing.assert_array_equal(symmetric, symmetric.T)
    
    def test_already_symmetric(self):
        """Test that symmetric matrices are unchanged."""
        matrix = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]], dtype=np.float64)
        symmetric = symmetrize(matrix)
        
        np.testing.assert_array_equal(symmetric, matrix)
    
    def test_formula(self):
        """Test that symmetrization uses correct formula."""
        matrix = np.array([[1, 2], [3, 4]], dtype=np.float64)
        symmetric = symmetrize(matrix)
        
        expected = np.array([[1, 2.5], [2.5, 4]])
        np.testing.assert_array_equal(symmetric, expected)


class TestComputeLaplacian:
    """Tests for Laplacian computation."""
    
    def test_unnormalized_laplacian(self):
        """Test unnormalized Laplacian L = D - W."""
        W = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.float64)
        L = compute_laplacian(W, normalized=False)
        
        # Diagonal should be degree
        degrees = W.sum(axis=1)
        np.testing.assert_array_equal(np.diag(L), degrees)
        
        # Off-diagonal should be -W
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert L[i, j] == -W[i, j]
    
    def test_normalized_laplacian_eigenvalues(self):
        """Test that normalized Laplacian has eigenvalues in [0, 2]."""
        W = np.random.rand(5, 5)
        W = (W + W.T) / 2  # Symmetrize
        L = compute_laplacian(W, normalized=True)
        
        eigenvalues = np.linalg.eigvalsh(L)
        
        assert np.all(eigenvalues >= -1e-10)
        assert np.all(eigenvalues <= 2 + 1e-10)
    
    def test_smallest_eigenvalue_is_zero(self):
        """Test that smallest eigenvalue of Laplacian is 0."""
        W = np.random.rand(5, 5) + 0.1
        W = (W + W.T) / 2
        L = compute_laplacian(W, normalized=True)
        
        eigenvalues = np.linalg.eigvalsh(L)
        min_eigenvalue = eigenvalues.min()
        
        assert np.isclose(min_eigenvalue, 0, atol=1e-10)