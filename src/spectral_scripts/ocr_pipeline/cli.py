"""Command-line interface for OCR pipeline."""

from __future__ import annotations

import sys
from pathlib import Path


def main():
    """Main entry point for run-ocr-pipeline command."""
    # Import and run the main function from run_ocr_pipeline script
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "scripts"))
    
    from run_ocr_pipeline import main as run_main
    run_main()


if __name__ == "__main__":
    main()