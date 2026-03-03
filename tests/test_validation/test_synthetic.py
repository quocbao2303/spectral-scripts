"""Tests for synthetic ground truth validation."""

import numpy as np
import pytest

from spectral_scripts.validation.synthetic import (
    SyntheticMatrix,
    SyntheticValidationResult,
    generate_base_confusion_matrix,
    perturb_confusion_matrix,
    generate_synthetic_matrices,
    run_synthetic_validation,
)
from spectral_scripts.features.profile import SpectralProfile
from spectral_scripts.distance.wasserstein import spectral_distance


class TestGenerateBaseConfusionMatrix:
    """Tests for base confusion matrix generation."""
    
    def test_correct_shape(self):
        """Test that generated matrix has correct shape."""
        n_chars = 20
        matrix = generate_base_confusion_matrix(n_chars=n_chars)
        
        assert matrix.shape == (n_chars, n_chars)
    
    def test_non_negative(self):
        """Test that all values are non-negative."""
        matrix = generate_base_confusion_matrix()
        
        assert np.all(matrix >= 0)
    
    def test_diagonal_dominance(self):
        """Test that diagonal is generally larger than off-diagonal."""
        matrix = generate_base_confusion_matrix(accuracy=0.9)
        
        diagonal_sum = np.trace(matrix)
        total = matrix.sum()
        
        assert diagonal_sum / total > 0.8
    
    def test_reproducible_with_seed(self):
        """Test that same seed produces same matrix."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        
        m1 = generate_base_confusion_matrix(rng=rng1)
        m2 = generate_base_confusion_matrix(rng=rng2)
        
        np.testing.assert_array_equal(m1, m2)


class TestPerturbConfusionMatrix:
    """Tests for confusion matrix perturbation."""
    
    def test_high_similarity_close_to_original(self):
        """Test that high similarity produces matrix close to original."""
        base = generate_base_confusion_matrix()
        perturbed = perturb_confusion_matrix(base, similarity=0.95)
        
        # Normalized Frobenius distance should be small
        base_norm = base / base.sum()
        pert_norm = perturbed / perturbed.sum()
        dist = np.linalg.norm(base_norm - pert_norm, "fro")
        
        assert dist < 0.5
    
    def test_low_similarity_different(self):
        """Test that low similarity produces different matrix."""
        base = generate_base_confusion_matrix()
        perturbed = perturb_confusion_matrix(base, similarity=0.1)
        
        # Should be substantially different
        base_norm = base / base.sum()
        pert_norm = perturbed / perturbed.sum()
        dist = np.linalg.norm(base_norm - pert_norm, "fro")
        
        assert dist > 0.1
    
    def test_non_negative(self):
        """Test that perturbed matrix is non-negative."""
        base = generate_base_confusion_matrix()
        perturbed = perturb_confusion_matrix(base, similarity=0.5)
        
        assert np.all(perturbed >= 0)


class TestGenerateSyntheticMatrices:
    """Tests for synthetic matrix set generation."""
    
    def test_correct_number_generated(self):
        """Test that correct number of matrices are generated."""
        n_per_group = 3
        matrices = generate_synthetic_matrices(n_matrices_per_group=n_per_group)
        
        # Default 3 groups * n_per_group
        assert len(matrices) == 3 * n_per_group
    
    def test_all_have_confusion_matrix(self):
        """Test that all synthetic matrices have valid confusion matrices."""
        matrices = generate_synthetic_matrices(n_matrices_per_group=2)
        
        for sm in matrices:
            assert isinstance(sm, SyntheticMatrix)
            assert sm.confusion is not None
            assert sm.group in ["A", "B", "C"]
            assert 0 <= sm.similarity_to_base <= 1
    
    def test_groups_have_different_similarities(self):
        """Test that different groups have different similarity levels."""
        matrices = generate_synthetic_matrices(n_matrices_per_group=5)
        
        group_similarities = {}
        for sm in matrices:
            if sm.group not in group_similarities:
                group_similarities[sm.group] = []
            group_similarities[sm.group].append(sm.similarity_to_base)
        
        # Group A should have highest similarity, C lowest
        mean_A = np.mean(group_similarities["A"])
        mean_B = np.mean(group_similarities["B"])
        mean_C = np.mean(group_similarities["C"])
        
        assert mean_A > mean_B > mean_C


class TestRunSyntheticValidation:
    """Tests for full synthetic validation."""
    
    def test_returns_valid_result(self):
        """Test that validation returns valid result object."""
        def dummy_dist(p1: SpectralProfile, p2: SpectralProfile) -> float:
            return spectral_distance(p1.spectral, p2.spectral)
        
        result = run_synthetic_validation(
            distance_fn=dummy_dist,
            n_matrices_per_group=3,
            n_chars=15,
            threshold=0.5,
        )
        
        assert isinstance(result, SyntheticValidationResult)
        assert -1 <= result.spearman_rho <= 1
        assert -1 <= result.kendall_tau <= 1
        assert 0 <= result.rank_preservation <= 1
        assert result.mean_absolute_error >= 0
    
    def test_good_method_passes(self):
        """Test that a reasonable method passes validation."""
        def good_dist(p1: SpectralProfile, p2: SpectralProfile) -> float:
            return spectral_distance(p1.spectral, p2.spectral)
        
        result = run_synthetic_validation(
            distance_fn=good_dist,
            n_matrices_per_group=5,
            n_chars=20,
            threshold=0.5,  # Lower threshold for test reliability
        )
        
        # Should generally pass with spectral distance
        # (May occasionally fail due to randomness)
        assert result.spearman_rho > 0.3
    
    def test_random_method_likely_fails(self):
        """Test that random distance likely fails validation."""
        rng = np.random.default_rng(42)
        
        def random_dist(p1: SpectralProfile, p2: SpectralProfile) -> float:
            return rng.random()
        
        result = run_synthetic_validation(
            distance_fn=random_dist,
            n_matrices_per_group=5,
            n_chars=20,
            threshold=0.7,
        )
        
        # Random should have correlation near 0
        assert abs(result.spearman_rho) < 0.5
    
    def test_summary_generation(self):
        """Test that summary is generated correctly."""
        def dummy_dist(p1: SpectralProfile, p2: SpectralProfile) -> float:
            return spectral_distance(p1.spectral, p2.spectral)
        
        result = run_synthetic_validation(
            distance_fn=dummy_dist,
            n_matrices_per_group=2,
            n_chars=10,
        )
        
        summary = result.summary()
        
        assert "Spearman" in summary
        assert "Kendall" in summary
        assert "PASSED" in summary or "FAILED" in summary