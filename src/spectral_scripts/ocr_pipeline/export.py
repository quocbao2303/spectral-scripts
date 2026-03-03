"""Export confusion matrices to various formats."""

from __future__ import annotations

from pathlib import Path
import json
import csv
import logging
from datetime import datetime

import numpy as np

from spectral_scripts.core.confusion_matrix import ConfusionMatrix
from spectral_scripts.ocr_pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


class ConfusionMatrixExporter:
    """Export confusion matrices to files."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.output_dir = config.output_dir

    def export_npz(self, matrix: ConfusionMatrix, filename: str | None = None) -> Path:
        """Export to .npz format (for spectral analysis)."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            filename = f"{matrix.script}.npz"
        
        output_path = self.output_dir / filename
        matrix.to_npz(output_path)
        
        logger.info(f"Exported {matrix.script} to {output_path}")
        return output_path

    def export_csv(self, matrix: ConfusionMatrix, filename: str | None = None) -> Path:
        """Export to CSV format (human-readable)."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            filename = f"{matrix.script}.csv"
        
        output_path = self.output_dir / filename
        
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            
            # Header row with characters
            header = [""] + matrix.characters
            writer.writerow(header)
            
            # Data rows
            for i, char in enumerate(matrix.characters):
                row = [char] + [str(int(x)) for x in matrix.matrix[i, :]]
                writer.writerow(row)
        
        logger.info(f"Exported {matrix.script} to {output_path}")
        return output_path

    def export_json(self, matrix: ConfusionMatrix, filename: str | None = None) -> Path:
        """Export to JSON format (with metadata)."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            filename = f"{matrix.script}.json"
        
        output_path = self.output_dir / filename
        
        data = {
            "script": matrix.script,
            "characters": matrix.characters,
            "matrix": matrix.matrix.tolist(),
            "metadata": {
                "accuracy": matrix.accuracy,
                "error_rate": matrix.error_rate,
                "total_observations": matrix.total_observations,
                "sparsity": matrix.sparsity,
                "size": matrix.size,
                "exported_at": datetime.now().isoformat(),
            },
            "top_confusions": [
                {"true": t, "predicted": p, "count": int(c)}
                for t, p, c in matrix.top_confusions(10)
            ],
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {matrix.script} to {output_path}")
        return output_path

    def export_all_formats(self, matrix: ConfusionMatrix) -> dict[str, Path]:
        """Export to all formats."""
        return {
            "npz": self.export_npz(matrix),
            "csv": self.export_csv(matrix),
            "json": self.export_json(matrix),
        }

    def export_all_matrices(
        self,
        matrices: dict[str, ConfusionMatrix],
        formats: list[str] = ["npz"],
    ) -> dict[str, dict[str, Path]]:
        """Export all matrices to specified formats."""
        results = {}
        
        for script, matrix in matrices.items():
            results[script] = {}
            
            if "npz" in formats:
                results[script]["npz"] = self.export_npz(matrix)
            if "csv" in formats:
                results[script]["csv"] = self.export_csv(matrix)
            if "json" in formats:
                results[script]["json"] = self.export_json(matrix)
        
        return results

    def generate_report(
        self,
        matrices: dict[str, ConfusionMatrix],
        output_path: Path | None = None,
    ) -> str:
        """Generate a summary report for all matrices."""
        engine_name = self.config.ocr.engine.upper()
        lines = [
            f"# OCR Pipeline Export Report - {engine_name}",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            "",
            f"- **Scripts:** {', '.join(matrices.keys())}",
            f"- **Total matrices:** {len(matrices)}",
            "",
            "## Per-Script Statistics",
            "",
        ]

        for script, matrix in matrices.items():
            lines.extend([
                f"### {script.title()}",
                "",
                f"- **Size:** {matrix.size} × {matrix.size}",
                f"- **Total observations:** {matrix.total_observations:,}",
                f"- **Accuracy:** {matrix.accuracy:.1%}",
                f"- **Error rate:** {matrix.error_rate:.1%}",
                f"- **Sparsity:** {matrix.sparsity:.1%}",
                "",
                "**Top confusions:**",
                "",
            ])
            
            for true_char, pred_char, count in matrix.top_confusions(5):
                lines.append(f"- '{true_char}' → '{pred_char}': {int(count)} times")
            
            lines.append("")

        report = "\n".join(lines)
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report)
            logger.info(f"Report saved to {output_path}")
        
        return report