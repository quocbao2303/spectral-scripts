#!/usr/bin/env python3
"""
Run synthetic ground truth validation for distance methods.

This is the PRIMARY validation criterion for the spectral distance method.
A method passes if it can recover known orderings of synthetic matrices.

Usage:
    python run_synthetic_validation.py
    python run_synthetic_validation.py --n-matrices 10 --script arabic
    python run_synthetic_validation.py --scripts latin greek cyrillic arabic
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spectral_scripts.features.profile import SpectralProfile, extract_profile
from spectral_scripts.distance.wasserstein import spectral_distance, multi_spectrum_distance
from spectral_scripts.distance.baselines import frobenius_distance
from spectral_scripts.validation.synthetic import (
    run_synthetic_validation,
    generate_synthetic_matrices,
)
from spectral_scripts.visualization.validation import plot_synthetic_validation

# Character set sizes for different scripts
SCRIPT_CHAR_COUNTS = {
    "latin": 26,
    "greek": 24,
    "cyrillic": 33,
    "arabic": 28,  # Core Arabic letters
}


def validate_method(
    method_name: str,
    distance_fn,
    n_matrices_per_group: int,
    n_chars: int,
    threshold: float,
    rng: np.random.Generator,
) -> dict:
    """Validate a single distance method."""
    print(f"\n--- {method_name} ---")
    
    result = run_synthetic_validation(
        distance_fn=distance_fn,
        n_matrices_per_group=n_matrices_per_group,
        n_chars=n_chars,
        threshold=threshold,
        rng=rng,
    )
    
    print(result.summary())
    
    return {
        "method": method_name,
        "spearman_rho": result.spearman_rho,
        "spearman_pvalue": result.spearman_pvalue,
        "kendall_tau": result.kendall_tau,
        "rank_preservation": result.rank_preservation,
        "mean_absolute_error": result.mean_absolute_error,
        "passed": result.passed,
        "threshold": result.threshold,
    }


def run_all_validations(
    n_matrices_per_group: int = 5,
    scripts: list[str] | None = None,
    threshold: float = 0.7,
    output_dir: Path | None = None,
    seed: int = 42,
) -> dict:
    """
    Run synthetic validation for all distance methods.
    
    Compares:
    1. Spectral distance (proposed method)
    2. Multi-spectrum distance (variant)
    3. Frobenius baseline
    4. Accuracy baseline
    
    Args:
        n_matrices_per_group: Matrices per similarity group.
        scripts: List of scripts to validate (latin, greek, cyrillic, arabic).
                 If None, uses all scripts.
        threshold: Spearman ρ threshold to pass.
        output_dir: Directory for outputs.
        seed: Random seed for reproducibility.
    
    Returns:
        Dictionary with all validation results.
    """
    if scripts is None:
        scripts = ["latin", "greek", "cyrillic", "arabic"]
    
    rng = np.random.default_rng(seed)
    
    results = {
        "parameters": {
            "n_matrices_per_group": n_matrices_per_group,
            "scripts": scripts,
            "threshold": threshold,
            "seed": seed,
        },
        "scripts": {},
    }
    
    print("=" * 70)
    print("SYNTHETIC GROUND TRUTH VALIDATION")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  Matrices per group: {n_matrices_per_group}")
    print(f"  Scripts: {', '.join(scripts)}")
    print(f"  Pass threshold: ρ ≥ {threshold}")
    print(f"  Random seed: {seed}")
    
    # Define distance functions
    def spectral_dist(p1: SpectralProfile, p2: SpectralProfile) -> float:
        return spectral_distance(p1.spectral, p2.spectral)
    
    def multi_spectrum_dist(p1: SpectralProfile, p2: SpectralProfile) -> float:
        return multi_spectrum_distance(p1.spectral, p2.spectral)
    
    def frobenius_dist(p1: SpectralProfile, p2: SpectralProfile) -> float:
        return frobenius_distance(p1.confusion.matrix, p2.confusion.matrix)
    
    def accuracy_dist(p1: SpectralProfile, p2: SpectralProfile) -> float:
        return abs(p1.interpretable.accuracy - p2.interpretable.accuracy)
    
    methods = [
        ("Spectral (proposed)", spectral_dist),
        ("Multi-spectrum", multi_spectrum_dist),
        ("Frobenius (baseline)", frobenius_dist),
        ("Accuracy diff (baseline)", accuracy_dist),
    ]
    
    # Validate for each script
    for script in scripts:
        if script not in SCRIPT_CHAR_COUNTS:
            print(f"\n✗ Unknown script: {script}")
            continue
        
        n_chars = SCRIPT_CHAR_COUNTS[script]
        
        print(f"\n{'=' * 70}")
        print(f"VALIDATING: {script.upper()} ({n_chars} characters)")
        print(f"{'=' * 70}")
        
        script_results = {
            "n_chars": n_chars,
            "methods": [],
        }
        
        # Run validations for this script
        for method_name, dist_fn in methods:
            # Use same RNG state for fair comparison
            method_rng = np.random.default_rng(seed)
            
            result = validate_method(
                method_name=method_name,
                distance_fn=dist_fn,
                n_matrices_per_group=n_matrices_per_group,
                n_chars=n_chars,
                threshold=threshold,
                rng=method_rng,
            )
            script_results["methods"].append(result)
        
        results["scripts"][script] = script_results
    
    # Summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    
    for script in scripts:
        if script not in results["scripts"]:
            continue
        
        script_results = results["scripts"][script]
        n_chars = script_results["n_chars"]
        
        print(f"\n{script.upper()} ({n_chars} chars):")
        print(f"{'Method':<30} {'ρ':>8} {'Passed':>8}")
        print("-" * 50)
        
        for method_result in script_results["methods"]:
            status = "✓" if method_result["passed"] else "✗"
            print(f"{method_result['method']:<30} {method_result['spearman_rho']:>8.3f} {status:>8}")
        
        # Check if proposed method outperforms baselines
        proposed_rho = script_results["methods"][0]["spearman_rho"]
        baseline_rhos = [r["spearman_rho"] for r in script_results["methods"][2:]]
        
        if proposed_rho > max(baseline_rhos):
            print("  ✓ Proposed method OUTPERFORMS all baselines")
        else:
            print("  ✗ Proposed method does NOT outperform all baselines")
    
    # Save results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_path = output_dir / "synthetic_validation.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {results_path}")
        
        # Generate plots for each script
        for script in scripts:
            if script not in results["scripts"]:
                continue
            
            n_chars = SCRIPT_CHAR_COUNTS[script]
            method_rng = np.random.default_rng(seed)
            
            proposed_result = run_synthetic_validation(
                distance_fn=spectral_dist,
                n_matrices_per_group=n_matrices_per_group,
                n_chars=n_chars,
                threshold=threshold,
                rng=method_rng,
            )
            
            fig = plot_synthetic_validation(
                proposed_result,
                save_path=output_dir / f"synthetic_validation_{script}.png"
            )
            print(f"Plot saved to: {output_dir / f'synthetic_validation_{script}.png'}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run synthetic ground truth validation for different scripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate all scripts
  python run_synthetic_validation.py
  
  # Validate only Arabic
  python run_synthetic_validation.py --scripts arabic
  
  # Validate Latin and Arabic with more matrices
  python run_synthetic_validation.py --scripts latin arabic --n-matrices 10
  
  # Custom output directory
  python run_synthetic_validation.py --output-dir my_results
        """
    )
    parser.add_argument(
        "--n-matrices", "-n",
        type=int,
        default=5,
        help="Number of matrices per similarity group (default: 5)"
    )
    parser.add_argument(
        "--scripts", "-s",
        nargs="+",
        choices=list(SCRIPT_CHAR_COUNTS.keys()),
        default=None,
        help="Scripts to validate (default: all)"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.7,
        help="Spearman ρ threshold to pass (default: 0.7)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("outputs/validation"),
        help="Output directory (default: outputs/validation)"
    )
    parser.add_argument(
        "--seed", "-S",
        dest="seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    run_all_validations(
        n_matrices_per_group=args.n_matrices,
        scripts=args.scripts,
        threshold=args.threshold,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()