#!/usr/bin/env python3
"""
Prepare dataset for OCR pipeline.

Creates expected directory structure and generates sample ground truth files.

Usage:
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --scripts latin greek cyrillic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def create_directory_structure(base_dir: Path, scripts: list[str]) -> None:
    """Create expected directory structure."""
    print("Creating directory structure...")
    
    # Create image directories
    for script in scripts:
        img_dir = base_dir / "images" / script
        img_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Created: {img_dir}")
    
    # Create ground truth directory
    gt_dir = base_dir / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Created: {gt_dir}")
    
    # Create sample ground truth files
    for script in scripts:
        gt_file = gt_dir / f"{script}_labels.txt"
        if not gt_file.exists():
            gt_file.write_text(f"# Ground truth labels for {script}\n# Format: filename.png: ground truth text\n")
            print(f"  Created: {gt_file}")


def print_instructions(base_dir: Path, scripts: list[str]) -> None:
    """Print setup instructions."""
    print("\n" + "=" * 60)
    print("SETUP INSTRUCTIONS")
    print("=" * 60)
    
    print("\n1. Add images to these directories:")
    for script in scripts:
        print(f"   {base_dir / 'images' / script}/")
    
    print("\n2. Add ground truth labels to these files:")
    for script in scripts:
        print(f"   {base_dir / 'ground_truth' / f'{script}_labels.txt'}")
    
    print("\n   Label file format:")
    print('   image_001.png: "The quick brown fox"')
    print('   image_002.png: "jumps over the lazy dog"')
    
    print("\n3. Run the OCR pipeline:")
    print("   python scripts/run_ocr_pipeline.py")
    
    print("\n4. Run spectral analysis:")
    print("   make analyze")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset for OCR pipeline"
    )
    parser.add_argument(
        "--base-dir", "-d",
        type=Path,
        default=Path("data/raw"),
        help="Base directory for raw data (default: data/raw)"
    )
    parser.add_argument(
        "--scripts", "-s",
        nargs="+",
        default=["latin", "greek", "cyrillic"],
        help="Scripts to prepare (default: latin greek cyrillic)"
    )
    
    args = parser.parse_args()
    
    print(f"Preparing dataset in: {args.base_dir}")
    print(f"Scripts: {', '.join(args.scripts)}")
    print()
    
    create_directory_structure(args.base_dir, args.scripts)
    print_instructions(args.base_dir, args.scripts)


if __name__ == "__main__":
    main()