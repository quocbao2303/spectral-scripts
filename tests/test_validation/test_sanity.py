"""Tests for metric sanity checks."""

import numpy as np
import pytest

from spectral_scripts.validation.sanity import (
    SanityCheckResult,
    check_non_negativity,
    check_identity,
    check_symmetry,
    check_triangle_inequality,
    check_self_minimum,
    run_sanity_checks,
)
from spectral_scripts.distance.matrix import DistanceMatrix, compute_distance_matrix
from spectral_scripts.features.profile import extract_profile


class TestCheckNonNegativity:
    """Tests for non-negativity check."""
    
    def test_all_positive_passes(self):
        """Test that all positive distances pass."""
        distances = DistanceMatrix(
            distances=np.array([[0, 0.5], [0.5, 0]]),
            scripts=["a", "b"],
            method="test",
            parameters={},
        )
        
        passed, violations = check_non_negativity(distances)
        
        assert passed
        assert len(violations) == 0
    
    def test_negative_value_fails(self):
        """Test that negative distances fail."""
        distances = DistanceMatrix(
            distances=np.array([[0, -0.5], [-0.5, 0]]),
            scripts=["a", "b"],
            method="test",
            parameters={},
        )
        
        passed, violations = check_non_negativity(distances)
        
        assert not passed
        assert len(violations) > 0


class TestCheckIdentity:
    """Tests for identity check (d(x,x) = 0)."""
    
    def test_zero_diagonal_passes(self):
        """Test that zero diagonal passes."""
        distances = DistanceMatrix(
            distances=np.array([[0, 0.5, 0.3], [0.5, 0, 0.4], [0.3, 0.4, 0]]),
            scripts=["a", "b", "c"],
            method="test",
            parameters={},
        )
        
        passed, violations = check_identity(distances)
        
        assert passed
        assert len(violations) == 0
    
    def test_nonzero_diagonal_fails(self):
        """Test that nonzero diagonal fails."""
        distances = DistanceMatrix(
            distances=np.array([[0.1, 0.5], [0.5, 0]]),
            scripts=["a", "b"],
            method="test",
            parameters={},
        )
        
        passed, violations = check_identity(distances)
        
        assert not passed
        assert len(violations) > 0


class TestCheckSymmetry:
    """Tests for symmetry check."""
    
    def test_symmetric_passes(self):
        """Test that symmetric matrix passes."""
        distances = DistanceMatrix(
            distances=np.array([[0, 0.5], [0.5, 0]]),
            scripts=["a", "b"],
            method="test",
            parameters={},
        )
        
        passed, violations = check_symmetry(distances)
        
        assert passed
        assert len(violations) == 0
    
    def test_asymmetric_fails(self):
        """Test that asymmetric matrix fails."""
        distances = DistanceMatrix(
            distances=np.array([[0, 0.5], [0.3, 0]]),
            scripts=["a", "b"],
            method="test",
            parameters={},
        )
        
        passed, violations = check_symmetry(distances)
        
        assert not passed
        assert len(violations) > 0


class TestCheckTriangleInequality:
    """Tests for triangle inequality check."""
    
    def test_valid_metric_passes(self):
        """Test that valid metric passes."""
        # Euclidean-like distances satisfy triangle inequality
        distances = DistanceMatrix(
            distances=np.array([
                [0, 1, 2],
                [1, 0, 1],
                [2, 1, 0],
            ]),
            scripts=["a", "b", "c"],
            method="test",
            parameters={},
        )
        
        passed, violations = check_triangle_inequality(distances)
        
        assert passed
        assert len(violations) == 0
    
    def test_violation_detected(self):
        """Test that triangle inequality violation is detected."""
        # d(a,c) = 10 > d(a,b) + d(b,c) = 1 + 1 = 2
        distances = DistanceMatrix(
            distances=np.array([
                [0, 1, 10],
                [1, 0, 1],
                [10, 1, 0],
            ]),
            scripts=["a", "b", "c"],
            method="test",
            parameters={},
        )
        
        passed, violations = check_triangle_inequality(distances)
        
        assert not passed
        assert len(violations) > 0


class TestCheckSelfMinimum:
    """Tests for self-minimum check (d(x,x) ≤ d(x,y))."""
    
    def test_valid_distances_pass(self):
        """Test that valid distances pass."""
        distances = DistanceMatrix(
            distances=np.array([[0, 0.5], [0.5, 0]]),
            scripts=["a", "b"],
            method="test",
            parameters={},
        )
        
        passed, violations = check_self_minimum(distances)
        
        assert passed
        assert len(violations) == 0
    
    def test_self_not_minimum_fails(self):
        """Test that fails when self-distance is not minimum."""
        distances = DistanceMatrix(
            distances=np.array([[0.5, 0.1], [0.1, 0]]),
            scripts=["a", "b"],
            method="test",
            parameters={},
        )
        
        passed, violations = check_self_minimum(distances)
        
        assert not passed
        assert len(violations) > 0


class TestRunSanityChecks:
    """Tests for complete sanity check suite."""
    
    def test_valid_metric_passes_all(self, sample_profiles):
        """Test that valid spectral distance passes all checks."""
        distance_matrix = compute_distance_matrix(
            sample_profiles,
            method="spectral",
        )
        
        result = run_sanity_checks(distance_matrix)
        
        assert isinstance(result, SanityCheckResult)
        assert result.non_negative
        assert result.identity
        assert result.symmetry
        # Triangle inequality should pass for spectral distance
    
    def test_summary_generation(self, sample_profiles):
        """Test that summary is generated correctly."""
        distance_matrix = compute_distance_matrix(
            sample_profiles,
            method="spectral",
        )
        
        result = run_sanity_checks(distance_matrix)
        summary = result.summary()
        
        assert "Non-negativity" in summary
        assert "Identity" in summary
        assert "Symmetry" in summary
        assert "Triangle inequality" in summary