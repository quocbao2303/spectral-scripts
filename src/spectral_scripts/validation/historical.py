"""Historical relationship validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats

from spectral_scripts.features.profile import SpectralProfile
from spectral_scripts.distance.matrix import DistanceMatrix


# Known historical relationships between scripts
# Format: (script1, script2, relationship_type, expected_similarity)
# expected_similarity: "high" (same family), "medium" (related), "low" (unrelated)
DEFAULT_HISTORICAL_RELATIONSHIPS = [
    # Latin-derived scripts
    ("latin", "italian", "derived", "high"),
    ("latin", "french", "derived", "high"),
    ("latin", "spanish", "derived", "high"),
    ("latin", "german", "derived", "high"),
    
    # Greek relationships
    ("greek", "latin", "influenced", "medium"),
    ("greek", "cyrillic", "influenced", "medium"),
    
    # Cyrillic relationships
    ("cyrillic", "russian", "derived", "high"),
    ("cyrillic", "serbian", "derived", "high"),
    
    # Unrelated scripts
    ("latin", "chinese", "unrelated", "low"),
    ("greek", "arabic", "unrelated", "low"),
    ("cyrillic", "hebrew", "unrelated", "low"),
]


@dataclass(frozen=True)
class HistoricalValidationResult:
    """
    Result of historical relationship validation.

    Attributes:
        n_relationships_tested: Number of known relationships tested.
        n_correct_orderings: Number of correctly ordered relationships.
        ordering_accuracy: Fraction of correct orderings.
        spearman_rho: Spearman correlation with expected similarity ranks.
        spearman_pvalue: P-value for Spearman correlation.
        relationship_details: Details for each tested relationship.
        passed: Whether validation passed (ordering_accuracy > threshold).
        threshold: Threshold used for pass/fail.
    """

    n_relationships_tested: int
    n_correct_orderings: int
    ordering_accuracy: float
    spearman_rho: float
    spearman_pvalue: float
    relationship_details: list[dict[str, object]]
    passed: bool
    threshold: float

    def summary(self) -> str:
        """Generate human-readable summary."""
        status = "✓ PASSED" if self.passed else "✗ FAILED"
        lines = [
            f"=== Historical Validation {status} ===",
            f"Relationships tested: {self.n_relationships_tested}",
            f"Correct orderings: {self.n_correct_orderings}/{self.n_relationships_tested}",
            f"Ordering accuracy: {self.ordering_accuracy:.1%}",
            f"Spearman ρ: {self.spearman_rho:.4f} (p={self.spearman_pvalue:.2e})",
            f"Threshold: {self.threshold:.0%}",
        ]
        return "\n".join(lines)


def similarity_to_rank(similarity: str) -> int:
    """Convert similarity label to numeric rank."""
    mapping = {"high": 3, "medium": 2, "low": 1}
    return mapping.get(similarity, 0)


def run_historical_validation(
    distance_matrix: DistanceMatrix,
    relationships: list[tuple[str, str, str, str]] | None = None,
    threshold: float = 0.6,
) -> HistoricalValidationResult:
    """
    Validate distance matrix against known historical relationships.

    Tests whether computed distances respect known linguistic relationships:
    - Related scripts should have smaller distances
    - Unrelated scripts should have larger distances

    Note: This is SECONDARY validation. Synthetic validation is primary.

    Args:
        distance_matrix: Computed distance matrix.
        relationships: List of (script1, script2, relationship_type, expected_similarity).
            If None, uses DEFAULT_HISTORICAL_RELATIONSHIPS.
        threshold: Minimum ordering accuracy to pass.

    Returns:
        HistoricalValidationResult with validation metrics.
    """
    if relationships is None:
        relationships = DEFAULT_HISTORICAL_RELATIONSHIPS

    available_scripts = set(distance_matrix.scripts)
    details = []
    expected_ranks = []
    observed_distances = []

    # Filter to relationships where both scripts are available
    valid_relationships = [
        r for r in relationships
        if r[0] in available_scripts and r[1] in available_scripts
    ]

    if len(valid_relationships) == 0:
        return HistoricalValidationResult(
            n_relationships_tested=0,
            n_correct_orderings=0,
            ordering_accuracy=0.0,
            spearman_rho=0.0,
            spearman_pvalue=1.0,
            relationship_details=[],
            passed=False,
            threshold=threshold,
        )

    for script1, script2, rel_type, expected_sim in valid_relationships:
        distance = distance_matrix.get_distance(script1, script2)

        expected_rank = similarity_to_rank(expected_sim)
        expected_ranks.append(expected_rank)
        observed_distances.append(distance)

        details.append({
            "script1": script1,
            "script2": script2,
            "relationship": rel_type,
            "expected_similarity": expected_sim,
            "expected_rank": expected_rank,
            "observed_distance": distance,
        })

    # Convert to arrays
    expected_ranks = np.array(expected_ranks)
    observed_distances = np.array(observed_distances)

    # Spearman correlation (negative because high similarity = low distance)
    if len(expected_ranks) >= 3:
        spearman_result = stats.spearmanr(expected_ranks, -observed_distances)
        spearman_rho = float(spearman_result.correlation)
        spearman_pvalue = float(spearman_result.pvalue)
    else:
        spearman_rho = 0.0
        spearman_pvalue = 1.0

    # Count correct orderings
    # For each pair of relationships, check if ordering is correct
    n_correct = 0
    n_total = 0

    for i in range(len(valid_relationships)):
        for j in range(i + 1, len(valid_relationships)):
            expected_order = np.sign(expected_ranks[i] - expected_ranks[j])
            observed_order = np.sign(observed_distances[j] - observed_distances[i])
            # Note: signs are opposite because high rank = low distance

            if expected_order != 0:  # Only count if there's an expected ordering
                n_total += 1
                if expected_order == observed_order:
                    n_correct += 1

    ordering_accuracy = n_correct / max(n_total, 1)
    passed = ordering_accuracy >= threshold

    return HistoricalValidationResult(
        n_relationships_tested=len(valid_relationships),
        n_correct_orderings=n_correct,
        ordering_accuracy=ordering_accuracy,
        spearman_rho=spearman_rho,
        spearman_pvalue=spearman_pvalue,
        relationship_details=details,
        passed=passed,
        threshold=threshold,
    )


def validate_within_family_closer(
    distance_matrix: DistanceMatrix,
    families: dict[str, list[str]],
) -> dict[str, float]:
    """
    Check if scripts within same family are closer than across families.

    Args:
        distance_matrix: Computed distance matrix.
        families: Dictionary mapping family names to lists of script names.

    Returns:
        Dictionary with metrics:
        - within_family_mean: Mean within-family distance
        - between_family_mean: Mean between-family distance
        - separation_ratio: within / between (lower is better)
    """
    available = set(distance_matrix.scripts)
    
    within_distances = []
    between_distances = []

    family_to_scripts = {
        family: [s for s in scripts if s in available]
        for family, scripts in families.items()
    }

    # Within-family distances
    for family, scripts in family_to_scripts.items():
        for i, s1 in enumerate(scripts):
            for s2 in scripts[i + 1:]:
                d = distance_matrix.get_distance(s1, s2)
                within_distances.append(d)

    # Between-family distances
    family_names = list(family_to_scripts.keys())
    for i, f1 in enumerate(family_names):
        for f2 in family_names[i + 1:]:
            for s1 in family_to_scripts[f1]:
                for s2 in family_to_scripts[f2]:
                    d = distance_matrix.get_distance(s1, s2)
                    between_distances.append(d)

    within_mean = float(np.mean(within_distances)) if within_distances else 0.0
    between_mean = float(np.mean(between_distances)) if between_distances else 0.0

    separation_ratio = within_mean / between_mean if between_mean > 0 else float("inf")

    return {
        "within_family_mean": within_mean,
        "between_family_mean": between_mean,
        "separation_ratio": separation_ratio,
        "n_within_pairs": len(within_distances),
        "n_between_pairs": len(between_distances),
    }