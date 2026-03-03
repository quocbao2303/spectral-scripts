"""Validation tests for spectral distance methods."""

from spectral_scripts.validation.synthetic import (
    SyntheticValidationResult,
    generate_synthetic_matrices,
    run_synthetic_validation,
)
from spectral_scripts.validation.sanity import (
    SanityCheckResult,
    run_sanity_checks,
)
from spectral_scripts.validation.bootstrap import (
    BootstrapResult,
    bootstrap_distance,
    bootstrap_distance_matrix,
)
from spectral_scripts.validation.permutation import (
    PermutationResult,
    permutation_test,
)
from spectral_scripts.validation.historical import (
    HistoricalValidationResult,
    run_historical_validation,
)

__all__ = [
    "SyntheticValidationResult",
    "generate_synthetic_matrices",
    "run_synthetic_validation",
    "SanityCheckResult",
    "run_sanity_checks",
    "BootstrapResult",
    "bootstrap_distance",
    "bootstrap_distance_matrix",
    "PermutationResult",
    "permutation_test",
    "HistoricalValidationResult",
    "run_historical_validation",
]