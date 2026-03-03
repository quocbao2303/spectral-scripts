#!/usr/bin/env python3
"""
Run the complete OCR pipeline to generate confusion matrices.

Supports multiple OCR engines: Tesseract, EasyOCR, TrOCR, PaddleOCR, Surya
Supports any registered language/script (see config.py for available languages)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Ensure src is in path so we can import spectral_scripts
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spectral_scripts.ocr_pipeline.config import (
    PipelineConfig,
    LANGUAGE_REGISTRY,
)
from spectral_scripts.ocr_pipeline.data_ingestion import (
    ImageDataLoader,
    GroundTruthLoader,
)
from spectral_scripts.ocr_pipeline.ocr_engine import get_ocr_engine
from spectral_scripts.ocr_pipeline.character_matching import CharacterMatcher
from spectral_scripts.ocr_pipeline.matrix_builder import MultiScriptMatrixBuilder
from spectral_scripts.ocr_pipeline.validation import PipelineValidator
from spectral_scripts.ocr_pipeline.export import ConfusionMatrixExporter

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = True, log_file: Path | None = None) -> None:
    """Configure logging with optional file output."""
    level = logging.INFO if verbose else logging.WARNING

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def print_header(title: str, width: int = 70) -> None:
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width + "\n")


def print_section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def validate_inputs(config: PipelineConfig, log: logging.Logger) -> bool:
    """Validate that required directories and files exist."""
    log.info("Validating input directories...")

    if not config.input_dir.exists():
        log.error(f"Input directory not found: {config.input_dir}")
        return False

    if not config.ground_truth_dir.exists():
        log.error(f"Ground truth directory not found: {config.ground_truth_dir}")
        return False

    for script in config.scripts:
        script_dir = config.input_dir / script
        if not script_dir.exists():
            log.warning(f"Missing directory: {script_dir}")
        else:
            image_count = len(list(script_dir.glob("*.*")))
            log.info(f"  {script}: {image_count} image files found")

    return True


def run_pipeline(config: PipelineConfig, dry_run: bool = False) -> dict[str, Any]:
    """Run the complete OCR pipeline."""
    log = logging.getLogger("pipeline")

    results = {
        "status": "running",
        "engine": config.ocr.engine,
        "scripts": config.scripts,
        "matrices": {},
        "statistics": {},
        "errors": [],
        "warnings": [],
    }

    ocr_engine = None

    try:
        # STEP 1: Validate inputs
        print_header("STEP 1: VALIDATING INPUTS")

        if not validate_inputs(config, log):
            results["status"] = "failed"
            results["errors"].append("Input validation failed")
            return results

        log.info("✓ Input validation passed\n")

        # STEP 2: Load ground truth
        print_header("STEP 2: LOADING DATA")

        image_loader = ImageDataLoader(config)
        gt_loader = GroundTruthLoader(config)

        log.info("Scanning images:")
        for script in config.scripts:
            count = image_loader.get_image_count(script)
            log.info(f"  {script}: {count} images")

        log.info("\nLoading ground truth labels:")
        ground_truth = gt_loader.load_all()

        for script, gt_dict in ground_truth.items():
            log.info(f"  {script}: {len(gt_dict)} labels")

        log.info("✓ Data loading complete\n")

        # STEP 3: Validate data consistency
        print_header("STEP 3: VALIDATING DATA CONSISTENCY")

        validator = PipelineValidator(config)

        dummy_images = {
            script: [None] * image_loader.get_image_count(script)
            for script in config.scripts
        }

        input_validation = validator.validate_inputs(dummy_images, ground_truth)
        log.info(input_validation.summary())

        if not input_validation.overall_passed:
            results["status"] = "failed"
            results["errors"].extend(input_validation.errors)
            log.error("✗ Data validation failed\n")
            return results

        log.info("✓ Data validation passed\n")

        if dry_run:
            log.info("DRY RUN MODE: Stopping after validation")
            results["status"] = "success_dry_run"
            return results

        # STEP 4: Initialize OCR engine
        print_header("STEP 4: INITIALIZING OCR ENGINE")

        log.info(f"OCR Engine: {config.ocr.engine}")
        log.info(f"Device: {config.ocr.device}")
        log.info(f"Batch size: {config.ocr.batch_size}")

        try:
            ocr_engine = get_ocr_engine(config)
            log.info(f"✓ {config.ocr.engine.upper()} OCR engine initialized\n")
        except Exception as e:
            log.error(f"✗ Failed to initialize OCR engine: {e}\n")
            results["status"] = "failed"
            results["errors"].append(f"OCR initialization failed: {str(e)}")
            return results

        # STEP 5: Process images
        print_header("STEP 5: PROCESSING IMAGES WITH OCR")

        matcher = CharacterMatcher(config)
        builder = MultiScriptMatrixBuilder(config)

        total_processed = 0
        total_skipped = 0
        total_errors = 0

        for script in config.scripts:
            # Check if engine supports this script
            if hasattr(ocr_engine, "supports_script"):
                if not ocr_engine.supports_script(script):
                    log.warning(
                        f"Skipping {script}: not supported by {config.ocr.engine}"
                    )
                    results["warnings"].append(
                        f"{script} not supported by {config.ocr.engine}"
                    )
                    continue

            print_section(f"Processing {script.upper()}")

            script_gt = ground_truth.get(script, {})
            processed = 0
            skipped = 0
            errors = 0

            for img_data in image_loader.iter_images(script):
                try:
                    gt_text = script_gt.get(img_data.filename)

                    if gt_text is None:
                        skipped += 1
                        continue

                    # Run OCR
                    ocr_result = ocr_engine.recognize(img_data.image, script)

                    # Match to ground truth
                    confusion_pairs = matcher.get_confusion_pairs(
                        gt_text, ocr_result.text
                    )

                    # Add to builder
                    builder.add_pairs(script, confusion_pairs)

                    processed += 1
                    total_processed += 1

                    if processed % 50 == 0:
                        log.info(
                            f"  Processed: {processed} | "
                            f"Skipped: {skipped} | Errors: {errors}"
                        )

                except Exception as e:
                    errors += 1
                    total_errors += 1
                    log.debug(f"  Error processing {img_data.filename}: {e}")

                finally:
                    # Close image to free resources
                    img_data.close()

            total_skipped += skipped

            log.info(
                f"  Total for {script}: {processed} processed | "
                f"{skipped} skipped | {errors} errors\n"
            )

            # Add to statistics
            results["statistics"][script] = {
                "processed": processed,
                "skipped": skipped,
                "errors": errors,
            }

            # Cleanup GPU memory between scripts
            if hasattr(ocr_engine, "cleanup"):
                ocr_engine.cleanup()

        log.info("✓ Image processing complete")
        log.info(f"  Total processed: {total_processed}")
        log.info(f"  Total skipped: {total_skipped}")
        log.info(f"  Total errors: {total_errors}\n")

        # STEP 6: Build confusion matrices
        print_header("STEP 6: BUILDING CONFUSION MATRICES")

        matrices = builder.build_all()

        log.info(builder.summary())

        for script, matrix in matrices.items():
            log.info(f"\n{script}:")
            log.info(f"  Size: {matrix.size} × {matrix.size}")
            log.info(f"  Total observations: {matrix.total_observations:,}")
            log.info(f"  Accuracy: {matrix.accuracy:.2%}")

        # STEP 7: Validate matrices
        print_header("STEP 7: VALIDATING CONFUSION MATRICES")

        matrix_validations = validator.validate_all_matrices(matrices)

        for script, validation in matrix_validations.items():
            log.info(f"\n{script}:")
            log.info(validation.summary())

            if validation.warnings:
                for warning in validation.warnings:
                    results["warnings"].append(f"{script}: {warning}")

        # STEP 8: Export results
        print_header("STEP 8: EXPORTING RESULTS")

        config.output_dir.mkdir(parents=True, exist_ok=True)

        exporter = ConfusionMatrixExporter(config)

        log.info("Exporting matrices:")
        export_paths = exporter.export_all_matrices(matrices, formats=["npz", "json"])

        for script, paths in export_paths.items():
            results["matrices"][script] = {}
            for fmt, path in paths.items():
                log.info(f"  {script}.{fmt} → {path}")
                results["matrices"][script][fmt] = str(path)

        log.info("\nGenerating report:")
        report_path = config.output_dir / f"{config.ocr.engine}_pipeline_report.md"
        exporter.generate_report(matrices, report_path)
        log.info(f"  report.md → {report_path}")

        results["report"] = str(report_path)

        # Save pipeline results to the same directory
        results_path = config.output_dir / "pipeline_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"  results.json → {results_path}")

        # Summary
        print_header("PIPELINE SUMMARY")

        log.info("Status: SUCCESS ✓")
        log.info(f"OCR Engine: {config.ocr.engine}")
        log.info(f"Processed: {total_processed} images")
        log.info(f"Scripts: {', '.join(config.scripts)}")
        log.info(f"Output directory: {config.output_dir}")

        if results["warnings"]:
            log.warning(f"\nWarnings ({len(results['warnings'])}):")
            for warning in results["warnings"][:5]:
                log.warning(f"  - {warning}")
            if len(results["warnings"]) > 5:
                log.warning(f"  ... and {len(results['warnings']) - 5} more")

        results["status"] = "success"
        return results

    except KeyboardInterrupt:
        log.info("\n✗ Pipeline interrupted by user")
        results["status"] = "interrupted"
        return results

    except Exception as e:
        log.exception(f"✗ Pipeline failed with error: {e}")
        results["status"] = "failed"
        results["errors"].append(f"Pipeline error: {str(e)}")
        return results

    finally:
        # Cleanup OCR engine
        if ocr_engine is not None and hasattr(ocr_engine, "cleanup"):
            ocr_engine.cleanup()


def main():
    """Main entry point."""
    available_scripts = sorted(list(LANGUAGE_REGISTRY.keys()))

    parser = argparse.ArgumentParser(
        description="Run OCR pipeline to generate confusion matrices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available languages: {', '.join(available_scripts)}

Examples:
  # Run with PaddleOCR (Recommended for Arabic)
  python scripts/run_ocr_pipeline.py --engine paddle --scripts arabic

  # Run with Surya (High accuracy multilingual)
  python scripts/run_ocr_pipeline.py --engine surya

  # Run with all default scripts
  python scripts/run_ocr_pipeline.py --engine trocr

  # Run with Tesseract
  python scripts/run_ocr_pipeline.py --engine tesseract

  # Dry run (validate only, no OCR)
  python scripts/run_ocr_pipeline.py --engine paddle --dry-run
        """,
    )

    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument("--config", "-c", type=Path, help="YAML configuration file")

    # UPDATED: Added paddle, surya, and glm to choices
    config_group.add_argument(
        "--engine",
        "-e",
        choices=["trocr", "tesseract", "easyocr", "paddle", "surya", "glm"],
        help="OCR engine to use",
    )
    config_group.add_argument(
        "--trocr-model-type",
        choices=["printed", "handwritten"],
        help="TrOCR model type",
    )

    dir_group = parser.add_argument_group("Directories")
    dir_group.add_argument("--input-dir", "-i", type=Path, help="Input directory")
    dir_group.add_argument("--ground-truth-dir", "-g", type=Path, help="Ground truth dir")
    dir_group.add_argument("--output-dir", "-o", type=Path, help="Output directory")

    proc_group = parser.add_argument_group("Processing")
    proc_group.add_argument(
        "--scripts",
        "-s",
        nargs="+",
        choices=available_scripts,
        help=f"Scripts to process (available: {', '.join(available_scripts)})",
    )
    proc_group.add_argument("--dry-run", action="store_true", help="Validate only")

    sys_group = parser.add_argument_group("System")
    sys_group.add_argument("--device", choices=["cpu", "cuda", "mps"], help="Device")
    sys_group.add_argument("--batch-size", type=int, help="Batch size")
    sys_group.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    sys_group.add_argument("--log-file", type=Path, help="Log file path")

    args = parser.parse_args()

    setup_logging(verbose=args.verbose, log_file=args.log_file)
    log = logging.getLogger("main")

    print("\n" + "=" * 70)
    print("  OCR PIPELINE - Confusion Matrix Generation")
    print("=" * 70)

    try:
        if args.config and args.config.exists():
            log.info(f"Loading configuration from: {args.config}")
            config = PipelineConfig.from_yaml(args.config)
        else:
            # Use defaults from config.py (latin, greek, cyrillic, arabic)
            config = PipelineConfig(
                input_dir=args.input_dir or Path("data/raw/images"),
                ground_truth_dir=args.ground_truth_dir or Path("data/raw/ground_truth"),
                output_dir=args.output_dir or Path("data/confusion_matrices"),
                scripts=args.scripts or ["latin", "greek", "cyrillic", "arabic"],
                verbose=args.verbose,
            )

        if args.engine:
            config.ocr.engine = args.engine
        if args.trocr_model_type:
            config.ocr.trocr_model_type = args.trocr_model_type
        if args.device:
            config.ocr.device = args.device
        if args.batch_size:
            config.ocr.batch_size = args.batch_size

    except Exception as e:
        log.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    print_header("CONFIGURATION")
    log.info(f"Input directory:      {config.input_dir}")
    log.info(f"Ground truth dir:     {config.ground_truth_dir}")
    log.info(f"Output directory:     {config.output_dir}")
    log.info(f"Scripts:              {', '.join(config.scripts)}")
    log.info(f"OCR Engine:           {config.ocr.engine}")
    log.info(f"Device:               {config.ocr.device}")

    if args.dry_run:
        log.info("\nMODE:                 DRY RUN (validation only)")

    try:
        results = run_pipeline(config, dry_run=args.dry_run)

        results_file = config.output_dir / "pipeline_results.json"
        config.output_dir.mkdir(parents=True, exist_ok=True)

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        log.info(f"\nResults saved to: {results_file}")

        if results["status"] in ("success", "success_dry_run"):
            log.info("\n✓ Pipeline completed successfully!")
            sys.exit(0)
        else:
            log.error(f"\n✗ Pipeline failed: {results['status']}")
            sys.exit(1)

    except KeyboardInterrupt:
        log.info("\n✗ Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        log.exception(f"✗ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()