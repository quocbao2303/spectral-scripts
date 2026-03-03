"""Tests for ConfusionMatrix class."""

import numpy as np
import pytest
from pathlib import Path
import tempfile

from spectral_scripts.core.confusion_matrix import ConfusionMatrix


class TestConfusionMatrixCreation:
    """Tests for ConfusionMatrix initialization."""
    
    def test_valid_creation(self):
        """Test creating a valid confusion matrix."""
        matrix = np.array([[10, 2], [1, 15]], dtype=np.float64)
        cm = ConfusionMatrix(
            matrix=matrix,
            script="test",
            characters=["a", "b"],
        )
        
        assert cm.size == 2
        assert cm.script == "test"
        assert cm.characters == ["a", "b"]
    
    def test_rejects_non_square(self):
        """Test that non-square matrices are rejected."""
        matrix = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        
        with pytest.raises(ValueError, match="must be square"):
            ConfusionMatrix(
                matrix=matrix,
                script="test",
                characters=["a", "b"],
            )
    
    def test_rejects_negative_values(self):
        """Test that negative values are rejected."""
        matrix = np.array([[10, -2], [1, 15]], dtype=np.float64)
        
        with pytest.raises(ValueError, match="negative"):
            ConfusionMatrix(
                matrix=matrix,
                script="test",
                characters=["a", "b"],
            )
    
    def test_rejects_mismatched_characters(self):
        """Test that character list must match matrix size."""
        matrix = np.array([[10, 2], [1, 15]], dtype=np.float64)
        
        with pytest.raises(ValueError, match="Characters length"):
            ConfusionMatrix(
                matrix=matrix,
                script="test",
                characters=["a", "b", "c"],
            )


class TestConfusionMatrixProperties:
    """Tests for ConfusionMatrix computed properties."""
    
    def test_accuracy(self, simple_confusion_matrix):
        """Test accuracy calculation."""
        cm = simple_confusion_matrix
        expected = (100 + 90 + 85 + 88 + 94) / cm.matrix.sum()
        
        assert np.isclose(cm.accuracy, expected)
    
    def test_accuracy_perfect(self, identity_confusion_matrix):
        """Test accuracy for perfect classifier."""
        assert np.isclose(identity_confusion_matrix.accuracy, 1.0)
    
    def test_error_rate(self, simple_confusion_matrix):
        """Test error rate is 1 - accuracy."""
        cm = simple_confusion_matrix
        assert np.isclose(cm.error_rate, 1 - cm.accuracy)
    
    def test_sparsity(self, sparse_confusion_matrix):
        """Test sparsity calculation."""
        cm = sparse_confusion_matrix
        n_zeros = np.sum(cm.matrix == 0)
        expected = n_zeros / cm.matrix.size
        
        assert np.isclose(cm.sparsity, expected)
    
    def test_total_observations(self, simple_confusion_matrix):
        """Test total observations count."""
        cm = simple_confusion_matrix
        assert cm.total_observations == int(cm.matrix.sum())


class TestConfusionMatrixMethods:
    """Tests for ConfusionMatrix methods."""
    
    def test_character_accuracy(self, simple_confusion_matrix):
        """Test per-character accuracy."""
        cm = simple_confusion_matrix
        
        # Character 'a' has 100 correct out of 108 total
        row_sum = cm.matrix[0, :].sum()
        expected = 100 / row_sum
        
        assert np.isclose(cm.character_accuracy("a"), expected)
    
    def test_character_accuracy_invalid(self, simple_confusion_matrix):
        """Test error for invalid character."""
        with pytest.raises(ValueError, match="not in matrix"):
            simple_confusion_matrix.character_accuracy("z")
    
    def test_top_confusions(self, simple_confusion_matrix):
        """Test top confusions extraction."""
        cm = simple_confusion_matrix
        top = cm.top_confusions(n=3)
        
        assert len(top) == 3
        assert all(len(t) == 3 for t in top)  # (true, pred, count)
        
        # Should be sorted by count descending
        counts = [t[2] for t in top]
        assert counts == sorted(counts, reverse=True)
    
    def test_subsample(self, simple_confusion_matrix, rng):
        """Test subsampling creates smaller matrix."""
        cm = simple_confusion_matrix
        subsampled = cm.subsample(0.5, rng=rng)
        
        assert subsampled.total_observations < cm.total_observations
        assert subsampled.total_observations == pytest.approx(
            cm.total_observations * 0.5, rel=0.1
        )
        assert subsampled.size == cm.size


class TestConfusionMatrixIO:
    """Tests for ConfusionMatrix save/load."""
    
    def test_save_load_roundtrip(self, simple_confusion_matrix):
        """Test saving and loading produces identical matrix."""
        cm = simple_confusion_matrix
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.npz"
            cm.to_npz(path)
            
            loaded = ConfusionMatrix.from_npz(path)
            
            assert loaded.script == cm.script
            assert loaded.characters == cm.characters
            np.testing.assert_array_equal(loaded.matrix, cm.matrix)
    
    def test_load_missing_file(self):
        """Test loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            ConfusionMatrix.from_npz("nonexistent.npz")