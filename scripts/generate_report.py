#!/usr/bin/env python3
"""
Generate comprehensive analysis report.

Creates a Markdown report with all analysis results, validation outcomes,
and embedded figures.

Usage:
    python generate_report.py --results-dir outputs
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spectral_scripts.distance.matrix import DistanceMatrix


def generate_report(
    results_dir: Path,
    output_path: Path | None = None,
) -> str:
    """
    Generate Markdown report from analysis results.
    
    Args:
        results_dir: Directory containing analysis outputs.
        output_path: Path for output report. If None, prints to stdout.
    
    Returns:
        Report as Markdown string.
    """
    lines = []
    
    # Header
    lines.extend([
        "# Spectral Confusion Profile Analysis Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
    ])
    
    # Load main results
    results_path = results_dir / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        
        lines.extend([
            "## Overview",
            "",
            f"- **Method:** {results.get('method', 'N/A')}",
            f"- **Scripts analyzed:** {len(results.get('scripts', []))}",
            "",
        ])
        
        # Script summary table
        if results.get("scripts"):
            lines.extend([
                "## Script Summary",
                "",
                "| Script | Size | Accuracy | Spectral Gap | Effective Rank |",
                "|--------|------|----------|--------------|----------------|",
            ])
            
            for script in results["scripts"]:
                lines.append(
                    f"| {script['name']} | {script['size']} | "
                    f"{script['accuracy']:.1%} | {script['spectral_gap']:.4f} | "
                    f"{script['effective_rank']:.1f} |"
                )
            
            lines.append("")
        
        # Validation results
        if results.get("validation", {}).get("sanity"):
            sanity = results["validation"]["sanity"]
            status = "✓ Passed" if sanity["all_passed"] else "✗ Failed"
            
            lines.extend([
                "## Validation Results",
                "",
                f"### Sanity Checks: {status}",
                "",
                "| Property | Status |",
                "|----------|--------|",
                f"| Non-negativity | {'✓' if sanity['non_negative'] else '✗'} |",
                f"| Identity (d(x,x)=0) | {'✓' if sanity['identity'] else '✗'} |",
                f"| Symmetry | {'✓' if sanity['symmetry'] else '✗'} |",
                f"| Triangle inequality | {'✓' if sanity['triangle_inequality'] else '✗'} |",
                "",
            ])
    
    # Load synthetic validation results
    synthetic_path = results_dir / "validation" / "synthetic_validation.json"
    if synthetic_path.exists():
        with open(synthetic_path) as f:
            synthetic = json.load(f)
        
        lines.extend([
            "### Synthetic Ground Truth Validation",
            "",
            "| Method | Spearman ρ | Kendall τ | Passed |",
            "|--------|------------|-----------|--------|",
        ])
        
        for method in synthetic.get("methods", []):
            status = "✓" if method["passed"] else "✗"
            lines.append(
                f"| {method['method']} | {method['spearman_rho']:.3f} | "
                f"{method['kendall_tau']:.3f} | {status} |"
            )
        
        lines.append("")
    
    # Distance matrix
    distance_path = results_dir / "distance_matrix.npz"
    if distance_path.exists():
        dm = DistanceMatrix.load(distance_path)
        
        lines.extend([
            "## Distance Matrix",
            "",
            f"**Method:** {dm.method}",
            "",
        ])
        
        # Format as table
        header = "| |" + "|".join(f" {s} " for s in dm.scripts) + "|"
        separator = "|" + "|".join("---" for _ in range(len(dm.scripts) + 1)) + "|"
        
        lines.append(header)
        lines.append(separator)
        
        for i, script in enumerate(dm.scripts):
            row = f"| **{script}** |"
            for j in range(len(dm.scripts)):
                row += f" {dm.distances[i, j]:.4f} |"
            lines.append(row)
        
        lines.append("")
        
        # Nearest neighbors
        lines.extend([
            "### Nearest Neighbors",
            "",
        ])
        
        for script in dm.scripts:
            ranked = dm.rank_by_distance(script)
            if ranked:
                nearest = ranked[0]
                lines.append(f"- **{script}** → {nearest[0]} (d={nearest[1]:.4f})")
        
        lines.append("")
    
    # Figures
    figures_dir = results_dir / "figures"
    if figures_dir.exists():
        figures = list(figures_dir.glob("*.png"))
        
        if figures:
            lines.extend([
                "## Figures",
                "",
            ])
            
            for fig_path in sorted(figures):
                # Use relative path
                rel_path = fig_path.relative_to(results_dir)
                title = fig_path.stem.replace("_", " ").title()
                lines.extend([
                    f"### {title}",
                    "",
                    f"![{title}]({rel_path})",
                    "",
                ])
    
    # Methodology note
    lines.extend([
        "---",
    ])
    
    report = "\n".join(lines)
    
    # Save or print
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
        print(f"Report saved to: {output_path}")
    
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Generate analysis report"
    )
    parser.add_argument(
        "--results-dir", "-r",
        type=Path,
        default=Path("outputs"),
        help="Directory containing analysis results (default: outputs)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output path for report (default: results_dir/report.md)"
    )
    
    args = parser.parse_args()
    
    if not args.results_dir.exists():
        print(f"Error: Results directory not found: {args.results_dir}")
        sys.exit(1)
    
    output_path = args.output or (args.results_dir / "report.md")
    
    report = generate_report(
        results_dir=args.results_dir,
        output_path=output_path,
    )
    
    # Also print to stdout
    print("\n" + "=" * 60)
    print("REPORT PREVIEW")
    print("=" * 60 + "\n")
    print(report[:2000] + "..." if len(report) > 2000 else report)


if __name__ == "__main__":
    main()