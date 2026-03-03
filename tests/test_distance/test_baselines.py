"""Tests for baseline distance methods."""

import numpy as np
import pytest

from spectral_scripts.core.confusion_matrix import ConfusionMatrix
from spectral_scripts.distance.baselines import (
    frobenius_distance,
    accuracy_distance,
    character_overlap_distance,
    confusion_pattern_distance,
    compute_baseline_distances,
    BaselineDistances,
)


class TestFrobeniusDistance:
    """Tests for Frobenius norm distance."""
    
    def test_identical_matrices_zero(self):
        """Test that identical matrices have zero distance."""
        matrix = np.array([[10, 2], [1, 15]], dtype=np.float64)
        
        dist = frobenius_distance(matrix, matrix)
        
        assert np.isclose(dist, 0, atol=1e-10)
    
    def test_symmetric(self):
        """Test that distance is symmetric."""
        m1 = np.array([[10, 2], [1, 15]], dtype=np.float64)
        m2 = np.array([[8, 4], [3, 12]], dtype=np.float64)
        
        assert np.isclose(frobenius_distance(m1, m2), frobenius_distance(m2, m1))
    
    def test_normalization(self):
        """Test that normalization is applied correctly."""
        m1 = np.array([[100, 0], [0, 100]], dtype=np.float64)
        m2 = np.array([[50, 0], [0, 50]], dtype=np.float64)
        
        # After normalization, both are [[0.5, 0], [0, 0.5]]
        dist = frobenius_distance(m1, m2, normalize=True)
        
        assert np.isclose(dist, 0, atol=1e-10)
    
    def test_different_shapes_raises(self):
        """Test that different shapes raise error."""
        m1 = np.array([[1, 2], [3, 4]], dtype=np.float64)
        m2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
        
        with pytest.raises(ValueError, match="same shape"):
            frobenius_distance(m1, m2, align=False)


class TestAccuracyDistance:
    """Tests for accuracy difference distance."""
    
    def test_same_accuracy_zero(self, simple_confusion_matrix):
        """Test that same accuracy gives zero distance."""
        dist = accuracy_distance(simple_confusion_matrix, simple_confusion_matrix)
        
        assert dist == 0.0
    
    def test_range(self, simple_confusion_matrix, identity_confusion_matrix):
        """Test that distance is in valid range."""
        dist = accuracy_distance(simple_confusion_matrix, identity_confusion_matrix)
        
        assert 0 <= dist <= 1
    
    def test_symmetric(self, simple_confusion_matrix, uniform_confusion_matrix):
        """Test that distance is symmetric."""
        d1 = accuracy_distance(simple_confusion_matrix, uniform_confusion_matrix)
        d2 = accuracy_distance(uniform_confusion_matrix, simple_confusion_matrix)
        
        assert d1 == d2


class TestCharacterOverlapDistance:
    """Tests for character overlap (Jaccard) distance."""
    
    def test_same_characters_zero(self, simple_confusion_matrix):
        """Test that same characters gives zero distance."""
        dist = character_overlap_distance(
            simple_confusion_matrix, simple_confusion_matrix
        )
        
        assert dist == 0.0
    
    def test_no_overlap_one(self):
        """Test that no overlap gives distance 1."""
        cm1 = ConfusionMatrix(
            matrix=np.eye(3) * 100,
            script="test1",
            characters=["a", "b", "c"],
        )
        cm2 = ConfusionMatrix(
            matrix=np.eye(3) * 100,
            script="test2",
            characters=["x", "y", "z"],
        )
        
        dist = character_overlap_distance(cm1, cm2)
        
        assert dist == 1.0
    
    def test_partial_overlap(self):
        """Test partial character overlap."""
        cm1 = ConfusionMatrix(
            matrix=np.eye(3) * 100,
            script="test1",
            characters=["a", "b", "c"],
        )
        cm2 = ConfusionMatrix(
            matrix=np.eye(3) * 100,
            script="test2",
            characters=["b", "c", "d"],
        )
        
        # Overlap: {b, c}, Union: {a, b, c, d}
        # Jaccard = 2/4 = 0.5, Distance = 0.5
        dist = character_overlap_distance(cm1, cm2)
        
        assert np.isclose(dist, 0.5)


class TestConfusionPatternDistance:
    """Tests for confusion pattern Jaccard distance."""
    
    def test_same_patterns_zero(self, simple_confusion_matrix):
        """Test that same patterns gives zero distance."""
        dist = confusion_pattern_distance(
            simple_confusion_matrix, simple_confusion_matrix
        )
        
        assert dist == 0.0
    
    def test_no_confusions_zero(self, identity_confusion_matrix):
        """Test that matrices with no confusions have zero distance."""
        dist = confusion_pattern_distance(
            identity_confusion_matrix, identity_confusion_matrix
        )
        
        assert dist == 0.0


class TestComputeBaselineDistances:
    """Tests for combined baseline distance computation."""
    
    def test_returns_all_distances(self, simple_confusion_matrix, uniform_confusion_matrix):
        """Test that all baseline distances are computed."""
        result = compute_baseline_distances(
            simple_confusion_matrix, 
            uniform_confusion_matrix
        )
        
        assert isinstance(result, BaselineDistances)
        assert result.frobenius >= 0
        assert 0 <= result.accuracy <= 1
        assert 0 <= result.character_overlap <= 1
        assert 0 <= result.confusion_pattern <= 1
    
    def test_to_dict(self, simple_confusion_matrix, uniform_confusion_matrix):
        """Test conversion to dictionary."""
        result = compute_baseline_distances(
            simple_confusion_matrix,
            uniform_confusion_matrix
        )
        
        d = result.to_dict()
        
        assert "frobenius" in d
        assert "accuracy" in d
        assert "character_overlap" in d
        assert "confusion_pattern" in d
    
    def test_as_array(self, simple_confusion_matrix, uniform_confusion_matrix):
        """Test conversion to array."""
        result = compute_baseline_distances(
            simple_confusion_matrix,
            uniform_confusion_matrix
        )
        
        arr = result.as_array()
        
        assert arr.ndim == 1
        assert len(arr) == 4