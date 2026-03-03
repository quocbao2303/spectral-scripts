"""OCR Pipeline: From images to confusion matrices."""

from spectral_scripts.ocr_pipeline.config import PipelineConfig
from spectral_scripts.ocr_pipeline.data_ingestion import ImageDataLoader, GroundTruthLoader
from spectral_scripts.ocr_pipeline.ocr_engine import TesseractOCR, EasyOCREngine
from spectral_scripts.ocr_pipeline.character_matching import CharacterMatcher
from spectral_scripts.ocr_pipeline.matrix_builder import ConfusionMatrixBuilder
from spectral_scripts.ocr_pipeline.validation import PipelineValidator
from spectral_scripts.ocr_pipeline.export import ConfusionMatrixExporter

__all__ = [
    "PipelineConfig",
    "ImageDataLoader",
    "GroundTruthLoader",
    "TesseractOCR",
    "EasyOCREngine",
    "CharacterMatcher",
    "ConfusionMatrixBuilder",
    "PipelineValidator",
    "ConfusionMatrixExporter",
]