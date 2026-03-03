#!/usr/bin/env python3
"""
Text-to-Image Pipeline: Convert text files to rendered images with ground truth labels.

Stages:
1. Text parsing and cleaning
2. Text segmentation (line-by-line)
3. Character mapping
4. Image rendering
5. Image augmentation
6. Ground truth label generation
7. Dataset organization

Usage:
    python scripts/run_text_to_image.py --scripts latin greek cyrillic arabic
    python scripts/run_text_to_image.py --scripts cyrillic --no-augmentation
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = True) -> None:
    """Setup logging."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def print_header(title: str, width: int = 70) -> None:
    """Print a simple, centered header block."""
    sep = "=" * width
    print(f"\n{sep}")
    print(title.center(width))
    print(f"{sep}\n")


def find_font(script: str) -> Path:
    """Find appropriate font file for a script."""
    cwd = Path.cwd()

    # Font mapping for different scripts
    font_names = {
        "latin": "NotoSans-Regular.ttf",
        "greek": "NotoSans-Regular.ttf",
        "cyrillic": "NotoSans-Regular.ttf",
        "arabic": "NotoSansArabic-Regular.ttf",
    }

    font_name = font_names.get(script, "NotoSans-Regular.ttf")

    # Search paths in order of priority
    search_paths = [
        cwd / "fonts" / font_name,
        cwd / "fonts" / "NotoSans-Regular.ttf",  # Fallback
        cwd.parent / "fonts" / font_name,
        cwd.parent / "fonts" / "NotoSans-Regular.ttf",
        Path.home() / "fonts" / font_name,
        Path.home() / "fonts" / "NotoSans-Regular.ttf",
        Path("/Library/Fonts") / font_name,
        Path("/usr/share/fonts/opentype/noto") / font_name,  # Linux
    ]

    for path in search_paths:
        if path.exists():
            logger.debug(f"Found font for {script}: {path}")
            return path

    # Raise helpful error
    raise FileNotFoundError(
        f"\n{'='*70}\n"
        f"Font not found for {script}!\n"
        f"{'='*70}\n\n"
        f"Required font: {font_name}\n\n"
        f"Download fonts from: https://fonts.google.com/noto\n\n"
        f"Place in your 'fonts/' directory:\n"
        f"  • NotoSans-Regular.ttf (for Latin, Greek, Cyrillic)\n"
        f"  • NotoSansArabic-Regular.ttf (for Arabic)\n\n"
        f"Or install system-wide:\n"
        f"  Linux:  sudo apt install fonts-noto fonts-noto-core\n"
        f"  macOS:  brew install font-noto\n"
        f"{'='*70}\n"
    )


def render_text_to_image(
    text: str,
    script: str,
    font_size: int = 32,
    image_width: int = 400,
) -> "Image.Image":
    """Render text to image using appropriate font for the script."""
    from PIL import Image, ImageDraw, ImageFont

    try:
        font_path = find_font(script)
        font = ImageFont.truetype(str(font_path), font_size)
    except FileNotFoundError as e:
        logger.error(str(e))
        raise
    except Exception as e:
        logger.error(f"Failed to load font for {script}: {e}")
        raise

    # Create image with white background
    image = Image.new("RGB", (image_width, 100), color="white")
    draw = ImageDraw.Draw(image)

    # Draw text (PIL handles RTL languages like Arabic automatically)
    draw.text((10, 20), text, fill="black", font=font)

    return image


def augment_image(image: "Image.Image", aug_index: int) -> "Image.Image":
    """Apply augmentation to image."""
    from PIL import ImageEnhance

    augmented = image.copy()

    if aug_index == 0:
        # No augmentation
        pass
    elif aug_index == 1:
        # Slight brightness adjustment
        enhancer = ImageEnhance.Brightness(augmented)
        augmented = enhancer.enhance(0.9)
    elif aug_index == 2:
        # Slight contrast adjustment
        enhancer = ImageEnhance.Contrast(augmented)
        augmented = enhancer.enhance(1.1)

    return augmented


def process_script(
    script: str,
    input_dir: Path,
    output_dir: Path,
    num_augmentations: int = 3,
    verbose: bool = False,
) -> dict:
    """
    Process a single script (language/writing system).

    Returns:
        Dictionary with processing results and mapping of generated images to text
    """
    logger_script = logging.getLogger(f"text_to_image.{script}")

    results = {
        "script": script,
        "input_files": 0,
        "total_lines": 0,
        "total_images": 0,
        "ground_truth_mapping": {},  # filename -> text
        "errors": [],
    }

    script_input_dir = input_dir / script
    script_output_dir = output_dir / "images" / script

    # Create output directory
    script_output_dir.mkdir(parents=True, exist_ok=True)

    logger_script.info(f"Processing {script} from {script_input_dir}")

    # Find text files
    text_files = list(script_input_dir.glob("*.txt"))
    results["input_files"] = len(text_files)

    if not text_files:
        error_msg = f"No .txt files found in {script_input_dir}"
        logger_script.error(error_msg)
        results["errors"].append(error_msg)
        return results

    logger_script.info(f"Found {len(text_files)} text files")

    image_counter = 0

    # Process each text file
    for text_file in sorted(text_files):
        logger_script.debug(f"Processing {text_file.name}")

        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Process each line
            for line_idx, line in enumerate(lines):
                line = line.strip()

                if not line:
                    continue

                results["total_lines"] += 1

                # Render image for each augmentation
                for aug_idx in range(num_augmentations):
                    try:
                        # Render text to image with appropriate font for the script
                        image = render_text_to_image(
                            line,
                            script=script,
                            font_size=32,
                            image_width=400,
                        )

                        # Apply augmentation
                        augmented_image = augment_image(image, aug_idx)

                        # Generate filename with augmentation index
                        # Format: script_source_index_font_augN.png
                        base_name = text_file.stem  # Remove .txt
                        filename = f"{base_name}_{image_counter:04d}_aug{aug_idx}.png"

                        # Save image
                        image_path = script_output_dir / filename
                        augmented_image.save(image_path)

                        # Record in ground truth mapping
                        # IMPORTANT: Map the ACTUAL filename (with _augN) to the text
                        results["ground_truth_mapping"][filename] = line

                        results["total_images"] += 1

                        if verbose and results["total_images"] % 100 == 0:
                            logger_script.info(f"Generated {results['total_images']} images...")

                    except Exception as e:
                        error_msg = f"Error rendering image {image_counter}_aug{aug_idx}: {e}"
                        logger_script.warning(error_msg)
                        results["errors"].append(error_msg)

                image_counter += 1

        except Exception as e:
            error_msg = f"Error processing {text_file.name}: {e}"
            logger_script.error(error_msg)
            results["errors"].append(error_msg)

    logger_script.info(f"✓ Generated {results['total_images']} images for {script}")

    return results


def write_ground_truth_file(
    script: str,
    ground_truth_mapping: dict[str, str],
    output_dir: Path,
) -> Path:
    """
    Write ground truth labels to file.

    Args:
        script: Script name (latin, greek, cyrillic, arabic, etc.)
        ground_truth_mapping: Dict mapping filename -> text
        output_dir: Output directory

    Returns:
        Path to written ground truth file
    """
    gt_output_dir = output_dir / "ground_truth"
    gt_output_dir.mkdir(parents=True, exist_ok=True)

    gt_file = gt_output_dir / f"{script}_labels.txt"

    logger.info(f"Writing ground truth for {script}: {len(ground_truth_mapping)} entries")

    with open(gt_file, 'w', encoding='utf-8') as f:
        # Write header
        f.write(f"# Ground truth labels for {script}\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write(f"# Total images: {len(ground_truth_mapping)}\n")
        f.write(f"# Format: filename: text_content\n")
        f.write(f"# Note: Filenames include augmentation suffix (_aug0, _aug1, _aug2)\n")
        f.write("\n")

        # Write entries in sorted order
        for filename in sorted(ground_truth_mapping.keys()):
            text = ground_truth_mapping[filename]
            f.write(f"{filename}: {text}\n")

    return gt_file


def generate_dataset_summary(
    all_results: dict[str, dict],
    output_dir: Path,
) -> None:
    """Generate a summary JSON file."""
    import json

    summary = {
        "generated": datetime.now().isoformat(),
        "scripts": {},
    }

    total_images = 0

    for script, results in all_results.items():
        summary["scripts"][script] = {
            "input_files": results["input_files"],
            "total_lines": results["total_lines"],
            "total_images": results["total_images"],
            "errors": len(results["errors"]),
        }
        total_images += results["total_images"]

    summary["total_images"] = total_images

    summary_file = output_dir / "dataset_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Wrote summary to {summary_file}")


def discover_scripts(input_dir: Path) -> list[str]:
    """
    Automatically discover available scripts by finding subdirectories.

    Args:
        input_dir: Root directory containing script folders

    Returns:
        Sorted list of script names (subdirectory names)
    """
    if not input_dir.exists():
        logger.warning(f"Input directory does not exist: {input_dir}")
        return []

    # Find all subdirectories (each represents a script/language)
    scripts = [
        d.name for d in input_dir.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ]

    return sorted(scripts)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert text files to rendered images with ground truth labels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-discover all language folders in input directory
  python scripts/run_text_to_image.py

  # Process specific scripts
  python scripts/run_text_to_image.py --scripts latin greek cyrillic arabic

  # Only Arabic
  python scripts/run_text_to_image.py --scripts arabic

  # No augmentation (faster)
  python scripts/run_text_to_image.py --no-augmentation

  # Custom directories (auto-discovers subdirectories)
  python scripts/run_text_to_image.py \\
    --input-dir my_texts \\
    --output-dir my_images
        """
    )

    parser.add_argument(
        "--input-dir", "-i",
        type=Path,
        default=Path("data/raw/texts"),
        help="Input directory with text files (default: data/raw/texts)"
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("data/raw"),
        help="Output directory for images and ground truth (default: data/raw)"
    )

    parser.add_argument(
        "--scripts", "-s",
        nargs="+",
        default=None,
        help="Scripts to process (default: auto-discover from input directory)"
    )

    parser.add_argument(
        "--augmentations", "-a",
        type=int,
        default=3,
        help="Number of augmentations per image (default: 3)"
    )

    parser.add_argument(
        "--no-augmentation",
        action="store_true",
        help="Disable image augmentation (aug0 only)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(verbose=args.verbose)

    # Auto-discover scripts if not specified
    if args.scripts is None:
        args.scripts = discover_scripts(args.input_dir)
        if not args.scripts:
            logger.error(
                f"No script folders found in {args.input_dir}.\n"
                f"Please create subdirectories like:\n"
                f"  {args.input_dir}/latin/\n"
                f"  {args.input_dir}/greek/\n"
                f"  {args.input_dir}/cyrillic/\n"
                f"  {args.input_dir}/arabic/\n"
            )
            sys.exit(1)

    # Adjust augmentations
    num_augmentations = 1 if args.no_augmentation else args.augmentations

    print_header("TEXT-TO-IMAGE PIPELINE")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Scripts: {', '.join(args.scripts)}")
    logger.info(f"Augmentations: {num_augmentations}")

    # Process each script
    all_results: dict[str, dict] = {}
    total_images = 0
    total_errors = 0

    for script in args.scripts:
        print_header(f"PROCESSING {script.upper()}")

        results = process_script(
            script=script,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            num_augmentations=num_augmentations,
            verbose=args.verbose,
        )

        all_results[script] = results

        # Write ground truth file
        gt_file = write_ground_truth_file(
            script,
            results["ground_truth_mapping"],
            args.output_dir,
        )

        logger.info(f"✓ {script}: {results['total_images']} images generated")
        logger.info(f"✓ Ground truth: {gt_file}")

        total_images += results["total_images"]
        total_errors += len(results["errors"])

    # Final summary header + JSON summary
    print_header("PIPELINE COMPLETE")
    generate_dataset_summary(all_results, args.output_dir)

    # Human-readable summary under the header
    print("Script summary:")
    for script, results in all_results.items():
        print(
            f"  {script:<8}  "
            f"files={results['input_files']:3d}  "
            f"lines={results['total_lines']:5d}  "
            f"images={results['total_images']:5d}  "
            f"errors={len(results['errors']):2d}"
        )

    print("\nTotals:")
    print(f"  Total images: {total_images}")
    print(f"  Total errors: {total_errors}")
    print(f"  Images dir:   {args.output_dir}/images")
    print(f"  Ground truth: {args.output_dir}/ground_truth")

    logger.info("\nSummary:")
    logger.info(f"  Total images: {total_images}")
    logger.info(f"  Total errors: {total_errors}")
    logger.info(f"  Output directory: {args.output_dir}/images")
    logger.info(f"  Ground truth: {args.output_dir}/ground_truth")

    if total_errors > 0:
        logger.warning(f"\n⚠️  {total_errors} errors occurred during processing")
    else:
        logger.info("\n✓ All scripts processed successfully!")


if __name__ == "__main__":
    main()