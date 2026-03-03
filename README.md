# Spectral Scripts

Spectral analysis of OCR confusion matrices for measuring the similarity between writing systems.

> Compare writing systems through their OCR error patterns using eigenvalue-based spectral methods and Wasserstein distance.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

This project provides a reproducible pipeline for quantifying the similarity between writing systems (e.g. Latin, Greek, Cyrillic, Arabic) using the confusion patterns produced by OCR engines. The method works in four stages:

1. **Doubly stochastic normalization** — Remove accuracy bias with the Sinkhorn-Knopp algorithm
2. **Spectral feature extraction** — Compute eigenvalue-based metrics from normalized confusion matrices
3. **Wasserstein distance** — Compare spectral distributions using optimal transport
4. **Validation** — Test against synthetic ground truth and verify metric properties

The pipeline supports multiple OCR engines (Tesseract, PaddleOCR, GLM-OCR) and generates distance matrices, dendrograms, and spectral comparison plots.

---

## Quick start

### Prerequisites

- Python 3.11 or later
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (for the default pipeline)
- Optional: PaddleOCR or GLM-OCR for improved Arabic support

### Installation

```bash
git clone https://github.com/quocbao2303/spectral-scripts.git
cd spectral-scripts
pip install -e ".[dev]"
```

### Run the full pipeline

```bash
# Generate text images, run OCR, build confusion matrices, analyze, validate
make tesseract-pipeline

# Or with PaddleOCR (better for Arabic)
make paddle-pipeline

# Or with GLM-OCR (vision-language model)
make install-glm   # one-time setup
make glm-pipeline
```

### Run analysis only (using pre-computed confusion matrices)

If you already have confusion matrices in `data/confusion_matrices/`:

```bash
make all
```

### View results

```bash
make summary                     # print results summary
open outputs/tesseract/figures/  # view generated figures
```

---

## How it works

```
Raw texts
  → Text-to-image rendering (multiple fonts, sizes, augmentations)
  → OCR engine (Tesseract / PaddleOCR / GLM-OCR)
  → Confusion matrix construction
  → Sinkhorn-Knopp doubly stochastic normalization
  → Eigenvalue decomposition → spectral features
  → Wasserstein distance between spectra
  → Distance matrix + clustering + validation
```

### Spectral features

| Feature | What it measures |
|---------|-----------------|
| Spectral gap (1 − |λ₂|) | Cluster structure in confusions |
| Effective rank (exp H) | Dimensionality of confusion pattern |
| Fiedler value (μ₂) | Algebraic connectivity of confusion graph |
| Spectral entropy | Uniformity of eigenvalue distribution |

### Wasserstein distance

The pipeline compares spectra using the Wasserstein-1 (Earth Mover's) distance on cumulative distributions. This metric handles spectra of different lengths (different-sized alphabets) and satisfies all properties of a true mathematical metric.

---

## Project structure

```
spectral-scripts/
├── Makefile                         # Pipeline automation
├── pyproject.toml                   # Project configuration
├── ocr_config.yaml                  # OCR engine settings
│
├── src/spectral_scripts/
│   ├── core/                        # Confusion matrix, normalization, eigendecomposition
│   ├── features/                    # Spectral and interpretable feature extraction
│   ├── distance/                    # Wasserstein, Frobenius, baseline distances
│   ├── validation/                  # Synthetic validation, sanity checks, bootstrap
│   ├── statistics/                  # Multiple testing corrections
│   ├── visualization/               # Spectrum plots, heatmaps, dendrograms
│   └── ocr_pipeline/                # OCR engine abstraction layer
│
├── scripts/
│   ├── run_text_to_image.py         # Generate text images with augmentation
│   ├── run_ocr_pipeline.py          # Run OCR and build confusion matrices
│   ├── run_analysis.py              # Spectral analysis and distance computation
│   ├── run_synthetic_validation.py  # Validation suite
│   ├── prepare_dataset.py           # Dataset preparation utilities
│   └── generate_report.py           # Markdown report generation
│
├── tests/                           # Unit tests (pytest)
├── fonts/                           # Noto Sans fonts for text rendering
│
└── data/
    ├── confusion_matrices/          # Pre-computed confusion matrices (.npz)
    │   └── tesseract/               # Tesseract results (included)
    └── raw/
        ├── texts/                   # Source texts per script
        └── ground_truth/            # Ground truth for OCR alignment
```

---

## Python API

```python
from spectral_scripts import ConfusionMatrix
from spectral_scripts.features.profile import extract_profile
from spectral_scripts.distance.wasserstein import spectral_distance

# Load confusion matrices
latin = ConfusionMatrix.from_npz("data/confusion_matrices/tesseract/latin.npz")
greek = ConfusionMatrix.from_npz("data/confusion_matrices/tesseract/greek.npz")

# Extract spectral profiles
latin_prof = extract_profile(latin)
greek_prof = extract_profile(greek)

# Compute distance
d = spectral_distance(latin_prof.spectral, greek_prof.spectral)
print(f"Latin–Greek distance: {d:.4f}")
```

---

## Command reference

| Task | Command |
|------|---------|
| Full end-to-end (Tesseract) | `make tesseract-pipeline` |
| Full end-to-end (PaddleOCR) | `make paddle-pipeline` |
| Analysis only | `make all` |
| Quick analysis | `make quick-analyze` |
| Validation only | `make quick-validate` |
| Run tests | `make test` |
| View results | `make summary` |
| Clean outputs | `make clean` |
| All commands | `make help` |

---

## Testing

```bash
make test                    # run all tests
make lint                    # code quality checks

# Or directly with pytest
python -m pytest tests/ -v
python -m pytest --cov=spectral_scripts --cov-report=html
```

---

## Methodology

The mathematical foundations are described in the accompanying course paper. Key references:

- **Sinkhorn-Knopp algorithm**: Sinkhorn (1967), Knight (2008)
- **Spectral graph theory**: Chung (1997), Mohar (1991)
- **Wasserstein distance**: Villani (2009), Peyré & Cuturi (2019)
- **OCR evaluation**: Rice et al. (1996), Smith (2007)

---

## Citation

```bibtex
@software{nguyen2026spectral,
  title   = {Spectral Scripts: Spectral Analysis of OCR Confusion Matrices
             for Script Comparison},
  author  = {Nguyen, Quoc Bao},
  year    = {2026},
  url     = {https://github.com/quocbao2303/spectral-scripts}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.