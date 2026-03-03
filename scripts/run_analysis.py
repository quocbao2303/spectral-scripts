#!/usr/bin/env python3
"""
Main analysis script for spectral confusion profile analysis.

Usage:
    python run_analysis.py --input-dir data/confusion_matrices --output-dir outputs
    python run_analysis.py --input-dir data/confusion_matrices --method multi_spectrum
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Literal

import numpy as np

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spectral_scripts.core.confusion_matrix import ConfusionMatrix
from spectral_scripts.features.profile import SpectralProfile, extract_profile
from spectral_scripts.distance.matrix import compute_distance_matrix, DistanceMatrix
from spectral_scripts.distance.wasserstein import spectral_distance
from spectral_scripts.validation.sanity import run_sanity_checks
from spectral_scripts.statistics.corrections import correct_pvalues
from spectral_scripts.visualization.heatmaps import (
    plot_distance_matrix,
    plot_distance_matrix_clustered,
)
from spectral_scripts.visualization.spectra import (
    plot_spectrum_comparison,
    plot_spectral_features_comparison,
)


def load_confusion_matrices(input_dir: Path) -> list[ConfusionMatrix]:
    """Load all confusion matrices from a directory."""
    matrices = []
    
    for npz_file in sorted(input_dir.glob("*.npz")):
        try:
            cm = ConfusionMatrix.from_npz(npz_file)
            matrices.append(cm)
            print(f"  Loaded: {cm.script} ({cm.size} chars, {cm.accuracy:.1%} accuracy)")
        except Exception as e:
            print(f"  Warning: Failed to load {npz_file.name}: {e}")
    
    return matrices


def run_analysis(
    input_dir: Path,
    output_dir: Path,
    method: Literal["spectral", "multi_spectrum", "frobenius"] = "spectral",
    run_validation: bool = True,
    generate_plots: bool = True,
) -> dict:
    """
    Run complete spectral analysis pipeline.
    
    Args:
        input_dir: Directory containing .npz confusion matrices.
        output_dir: Directory for outputs.
        method: Distance method to use.
        run_validation: Whether to run sanity checks.
        generate_plots: Whether to generate visualization plots.
    
    Returns:
        Dictionary with analysis results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    results = {
        "method": method,
        "input_dir": str(input_dir),
        "scripts": [],
        "validation": {},
        "distances": {},
    }
    
    # Step 1: Load confusion matrices
    print("\n=== Loading Confusion Matrices ===")
    matrices = load_confusion_matrices(input_dir)
    
    if len(matrices) < 2:
        print("Error: Need at least 2 confusion matrices for comparison")
        return results
    
    print(f"\nLoaded {len(matrices)} matrices")
    
    # Step 2: Extract spectral profiles
    print("\n=== Extracting Spectral Profiles ===")
    profiles: list[SpectralProfile] = []
    
    for cm in matrices:
        profile = extract_profile(cm)
        profiles.append(profile)
        results["scripts"].append({
            "name": profile.script,
            "size": profile.size,
            "accuracy": profile.interpretable.accuracy,
            "spectral_gap": profile.spectral.bistochastic_gap,
            "effective_rank": profile.spectral.bistochastic_effective_rank,
        })
        print(f"  {profile.script}: gap={profile.spectral.bistochastic_gap:.4f}, "
              f"eff_rank={profile.spectral.bistochastic_effective_rank:.1f}")
    
    # Step 3: Compute distance matrix
    print(f"\n=== Computing Distance Matrix ({method}) ===")
    distance_matrix = compute_distance_matrix(profiles, method=method)
    
    # Save distance matrix
    distance_matrix.save(output_dir / "distance_matrix.npz")
    print(f"  Saved to: {output_dir / 'distance_matrix.npz'}")
    
    # Store distances in results
    for i, script1 in enumerate(distance_matrix.scripts):
        for j, script2 in enumerate(distance_matrix.scripts):
            if i < j:
                key = f"{script1}_vs_{script2}"
                results["distances"][key] = float(distance_matrix.distances[i, j])
    
    # Step 4: Sanity checks
    if run_validation:
        print("\n=== Running Sanity Checks ===")
        
        def dist_fn(p1: SpectralProfile, p2: SpectralProfile) -> float:
            return spectral_distance(p1.spectral, p2.spectral)
        
        sanity = run_sanity_checks(distance_matrix, profiles, dist_fn)
        print(sanity.summary())
        
        results["validation"]["sanity"] = {
            "all_passed": sanity.all_passed,
            "non_negative": sanity.non_negative,
            "identity": sanity.identity,
            "symmetry": sanity.symmetry,
            "triangle_inequality": sanity.triangle_inequality,
        }
    
    # Step 5: Generate plots
    if generate_plots:
        print("\n=== Generating Plots ===")
        
        # Distance matrix heatmap
        plot_distance_matrix(
            distance_matrix,
            save_path=figures_dir / "distance_matrix.png"
        )
        print(f"  Saved: distance_matrix.png")
        
        # Clustered distance matrix
        if len(profiles) >= 3:
            plot_distance_matrix_clustered(
                distance_matrix,
                save_path=figures_dir / "distance_matrix_clustered.png"
            )
            print(f"  Saved: distance_matrix_clustered.png")
        
        # Spectrum comparison
        plot_spectrum_comparison(
            profiles,
            save_path=figures_dir / "spectrum_comparison.png"
        )
        print(f"  Saved: spectrum_comparison.png")
        
        # Spectral features comparison
        plot_spectral_features_comparison(
            profiles,
            save_path=figures_dir / "spectral_features.png"
        )
        print(f"  Saved: spectral_features.png")
    
    # Step 6: Print summary
    print("\n=== Distance Summary ===")
    print(distance_matrix.summary())
    
    # Save results JSON
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Spectral analysis of OCR confusion matrices"
    )
    parser.add_argument(
        "--input-dir", "-i",
        type=Path,
        required=True,
        help="Directory containing .npz confusion matrices"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("outputs"),
        help="Output directory (default: outputs)"
    )
    parser.add_argument(
        "--method", "-m",
        choices=["spectral", "multi_spectrum", "frobenius"],
        default="spectral",
        help="Distance method (default: spectral)"
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Skip validation checks"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation"
    )
    
    args = parser.parse_args()
    
    if not args.input_dir.exists():
        print(f"Error: Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    run_analysis(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        method=args.method,
        run_validation=not args.no_validation,
        generate_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()