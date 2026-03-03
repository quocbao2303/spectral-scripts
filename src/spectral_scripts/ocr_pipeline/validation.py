"""Validation checks for OCR pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import numpy as np

from spectral_scripts.core.confusion_matrix import ConfusionMatrix
from spectral_scripts.ocr_pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a single validation check."""

    passed: bool
    check_name: str
    message: str
    details: dict[str, Any] | None = None


@dataclass
class PipelineValidationReport:
    """Complete validation report for pipeline inputs."""

    checks: list[ValidationResult]
    overall_passed: bool
    warnings: list[str]
    errors: list[str]

    def summary(self) -> str:
        """Generate human-readable summary."""
        status = "✓ PASSED" if self.overall_passed else "✗ FAILED"
        lines = [
            f"=== Pipeline Validation: {status} ===",
            "",
            "Checks:",
        ]
        
        for check in self.checks:
            icon = "✓" if check.passed else "✗"
            lines.append(f"  {icon} {check.check_name}: {check.message}")
        
        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for warning in self.warnings:
                lines.append(f"  ⚠ {warning}")
        
        if self.errors:
            lines.append("")
            lines.append("Errors:")
            for error in self.errors:
                lines.append(f"  ✗ {error}")
        
        return "\n".join(lines)


@dataclass
class MatrixValidationReport:
    """Validation report for a single confusion matrix."""

    checks: list[ValidationResult]
    overall_passed: bool
    warnings: list[str]
    errors: list[str]

    def summary(self) -> str:
        """Generate human-readable summary."""
        status = "✓ PASSED" if self.overall_passed else "✗ FAILED"
        lines = [
            f"=== Matrix Validation: {status} ===",
            "",
            "Checks:",
        ]
        
        for check in self.checks:
            icon = "✓" if check.passed else "✗"
            lines.append(f"  {icon} {check.check_name}: {check.message}")
        
        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for warning in self.warnings:
                lines.append(f"  ⚠ {warning}")
        
        if self.errors:
            lines.append("")
            lines.append("Errors:")
            for error in self.errors:
                lines.append(f"  ✗ {error}")
        
        return "\n".join(lines)


class PipelineValidator:
    """Validate OCR pipeline inputs and outputs."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def validate_inputs(
        self,
        images: dict[str, list],
        ground_truth: dict[str, dict[str, str]],
    ) -> PipelineValidationReport:
        """
        Validate pipeline inputs before processing.

        Checks:
        - Images exist for each script
        - Ground truth exists for each script
        - Coverage is adequate
        - Directory structure is correct
        """
        checks = []
        warnings = []
        errors = []

        # Check 1: Images exist
        for script in self.config.scripts:
            img_count = len(images.get(script, []))
            if img_count == 0:
                checks.append(ValidationResult(
                    passed=False,
                    check_name=f"Images for {script}",
                    message=f"No images found for {script}",
                ))
                errors.append(f"No images for {script}")
            else:
                checks.append(ValidationResult(
                    passed=True,
                    check_name=f"Images for {script}",
                    message=f"Found {img_count} images",
                ))

        # Check 2: Ground truth exists
        for script in self.config.scripts:
            gt_count = len(ground_truth.get(script, {}))
            img_count = len(images.get(script, []))
            
            if gt_count == 0:
                checks.append(ValidationResult(
                    passed=False,
                    check_name=f"Ground truth for {script}",
                    message="No ground truth found",
                ))
                errors.append(f"No ground truth for {script}")
            else:
                coverage = gt_count / img_count if img_count > 0 else 0
                passed = coverage >= 0.8
                checks.append(ValidationResult(
                    passed=passed,
                    check_name=f"Ground truth for {script}",
                    message=f"{gt_count}/{img_count} images have ground truth ({coverage:.0%})",
                ))
                if coverage < 0.8:
                    warnings.append(
                        f"Low ground truth coverage for {script}: {coverage:.0%}"
                    )
                elif coverage < 1.0:
                    warnings.append(
                        f"Incomplete ground truth for {script}: {coverage:.0%} "
                        f"({img_count - gt_count} missing)"
                    )

        # Check 3: Directory structure
        input_dir_exists = self.config.input_dir.exists()
        gt_dir_exists = self.config.ground_truth_dir.exists()
        
        checks.append(ValidationResult(
            passed=input_dir_exists,
            check_name="Input directory exists",
            message=f"{self.config.input_dir}",
        ))
        
        checks.append(ValidationResult(
            passed=gt_dir_exists,
            check_name="Ground truth directory exists",
            message=f"{self.config.ground_truth_dir}",
        ))
        
        if not input_dir_exists:
            errors.append(f"Input directory not found: {self.config.input_dir}")
        if not gt_dir_exists:
            errors.append(f"Ground truth directory not found: {self.config.ground_truth_dir}")

        overall_passed = len(errors) == 0

        return PipelineValidationReport(
            checks=checks,
            overall_passed=overall_passed,
            warnings=warnings,
            errors=errors,
        )

    def validate_matrix(self, matrix: ConfusionMatrix) -> MatrixValidationReport:
        """
        Validate a generated confusion matrix.

        Checks:
        - Matrix is square
        - Matrix has reasonable size
        - Diagonal dominance (accuracy)
        - No all-zero rows/columns
        """
        checks = []
        warnings = []
        errors = []

        # Check 1: Square matrix
        is_square = matrix.matrix.shape[0] == matrix.matrix.shape[1]
        checks.append(ValidationResult(
            passed=is_square,
            check_name="Square matrix",
            message=f"Shape: {matrix.matrix.shape}",
        ))
        if not is_square:
            errors.append(f"Matrix is not square: {matrix.matrix.shape}")

        # Check 2: Reasonable size
        size = matrix.size
        size_ok = 5 <= size <= 500
        checks.append(ValidationResult(
            passed=size_ok,
            check_name="Matrix size",
            message=f"{size} characters",
        ))
        if size < 5:
            warnings.append(f"Matrix very small: {size}x{size}")
        elif size > 500:
            warnings.append(f"Matrix very large: {size}x{size}")

        # Check 3: Accuracy
        accuracy = matrix.accuracy
        accuracy_ok = accuracy > 0.5
        checks.append(ValidationResult(
            passed=accuracy_ok,
            check_name="Accuracy",
            message=f"{accuracy:.1%}",
        ))
        if accuracy < 0.3:
            errors.append(f"Accuracy too low: {accuracy:.1%}")
        elif accuracy < 0.5:
            warnings.append(f"Low accuracy: {accuracy:.1%}")
        elif accuracy > 0.99:
            warnings.append(f"Suspiciously high accuracy: {accuracy:.1%}")

        # Check 4: Total observations
        total = matrix.total_observations
        total_ok = total >= 50
        checks.append(ValidationResult(
            passed=total_ok,
            check_name="Sample size",
            message=f"{total} observations",
        ))
        if total < 50:
            warnings.append(f"Low sample size: {total}")

        # Check 5: Zero rows/columns
        zero_rows = np.sum(matrix.matrix.sum(axis=1) == 0)
        zero_cols = np.sum(matrix.matrix.sum(axis=0) == 0)
        
        no_zero_structure = zero_rows == 0 and zero_cols == 0
        checks.append(ValidationResult(
            passed=no_zero_structure,
            check_name="No zero rows/columns",
            message=f"{zero_rows} zero rows, {zero_cols} zero columns",
        ))
        if zero_rows > 0 or zero_cols > 0:
            warnings.append(
                f"Matrix has {zero_rows} zero rows, {zero_cols} zero columns"
            )

        # Check 6: Diagonal dominance
        diagonal_sum = np.trace(matrix.matrix)
        total_sum = matrix.matrix.sum()
        diagonal_ratio = diagonal_sum / total_sum if total_sum > 0 else 0
        
        diagonal_ok = diagonal_ratio > 0.3
        checks.append(ValidationResult(
            passed=diagonal_ok,
            check_name="Diagonal dominance",
            message=f"Diagonal {diagonal_ratio:.1%} of total",
        ))
        if diagonal_ratio < 0.3:
            warnings.append(
                f"Low diagonal dominance: {diagonal_ratio:.1%}. "
                "Matrix may have random patterns."
            )

        # Check 7: Consistency
        row_sums = matrix.matrix.sum(axis=1)
        col_sums = matrix.matrix.sum(axis=0)
        
        row_variance = np.var(row_sums)
        col_variance = np.var(col_sums)
        
        consistent = row_variance < total_sum and col_variance < total_sum
        checks.append(ValidationResult(
            passed=consistent,
            check_name="Consistency",
            message=f"Row variance: {row_variance:.1f}, Col variance: {col_variance:.1f}",
        ))
        if row_variance > total_sum:
            warnings.append(
                f"High variance in row sums: {row_variance:.1f}. "
                "Some characters may have different accuracy levels."
            )

        overall_passed = len(errors) == 0 and all(c.passed for c in checks)

        return MatrixValidationReport(
            checks=checks,
            overall_passed=overall_passed,
            warnings=warnings,
            errors=errors,
        )

    def validate_all_matrices(
        self, matrices: dict[str, ConfusionMatrix]
    ) -> dict[str, MatrixValidationReport]:
        """Validate all confusion matrices."""
        return {
            script: self.validate_matrix(matrix)
            for script, matrix in matrices.items()
        }

    def validate_directory_structure(self) -> bool:
        """Check if all required directories exist."""
        required_dirs = [
            self.config.input_dir,
            self.config.ground_truth_dir,
            self.config.output_dir,
        ]
        
        for dir_path in required_dirs:
            if not dir_path.exists():
                logger.warning(f"Creating directory: {dir_path}")
                dir_path.mkdir(parents=True, exist_ok=True)
        
        return all(d.exists() for d in required_dirs)

    def get_validation_summary(
        self, matrices: dict[str, ConfusionMatrix]
    ) -> dict[str, object]:
        """Generate a summary of validation results."""
        validations = self.validate_all_matrices(matrices)
        
        summary = {
            "total_matrices": len(matrices),
            "passed_validation": sum(
                1 for v in validations.values() if v.overall_passed
            ),
            "failed_validation": sum(
                1 for v in validations.values() if not v.overall_passed
            ),
            "details": {
                script: {
                    "passed": validation.overall_passed,
                    "checks": [
                        {
                            "name": c.check_name,
                            "passed": c.passed,
                            "message": c.message,
                        }
                        for c in validation.checks
                    ],
                    "warnings": validation.warnings,
                    "errors": validation.errors,
                }
                for script, validation in validations.items()
            },
        }
        
        return summary