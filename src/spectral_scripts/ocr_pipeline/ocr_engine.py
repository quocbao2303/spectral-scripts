"""OCR engine wrappers for Tesseract, EasyOCR, and TrOCR."""

from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from pathlib import Path

# Prevent PaddleOCR/OpenMP hangs on macOS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from PIL import Image
import numpy as np

from spectral_scripts.ocr_pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Result from OCR processing."""

    text: str
    confidence: float
    characters: list[str]
    char_confidences: list[float]
    bboxes: list[tuple[int, int, int, int]] | None = None
    raw_output: Any = None

    @property
    def mean_confidence(self) -> float:
        """Mean confidence across all characters."""
        if not self.char_confidences:
            return self.confidence
        return sum(self.char_confidences) / len(self.char_confidences)

    @classmethod
    def empty(cls) -> "OCRResult":
        """Create an empty result for error cases."""
        return cls(
            text="",
            confidence=0.0,
            characters=[],
            char_confidences=[],
        )


class OCREngine(ABC):
    """Abstract base class for OCR engines."""

    @abstractmethod
    def recognize(self, image: Image.Image, script: str) -> OCRResult:
        """Recognize text in an image."""
        raise NotImplementedError

    @abstractmethod
    def recognize_characters(
        self, image: Image.Image, script: str
    ) -> list[tuple[str, float]]:
        """Recognize individual characters with confidences."""
        raise NotImplementedError

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the OCR engine is available."""
        raise NotImplementedError

    def supports_script(self, script: str) -> bool:
        """Check if engine supports a specific script."""
        return True  # Override in subclasses

    def cleanup(self) -> None:
        """Cleanup resources (GPU memory, etc.)."""
        # Override in subclasses when needed
        return


class TesseractOCR(OCREngine):
    """Tesseract OCR wrapper."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._tesseract = None
        self._check_availability()

    def _check_availability(self) -> None:
        """Check if Tesseract is available."""
        try:
            import pytesseract

            self._tesseract = pytesseract
            pytesseract.get_tesseract_version()
            logger.info("Tesseract OCR initialized successfully")
        except Exception as e:
            logger.error(f"Tesseract not available: {e}")
            self._tesseract = None

    def is_available(self) -> bool:
        return self._tesseract is not None

    def recognize(self, image: Image.Image, script: str) -> OCRResult:
        """Recognize text using Tesseract."""
        if not self.is_available():
            raise RuntimeError("Tesseract is not available")

        lang = self.config.get_tesseract_lang(script)

        # Convert to grayscale if needed
        if image.mode != "L":
            image = image.convert("L")

        ocr_config = f"--psm {self.config.ocr.psm} --oem {self.config.ocr.oem}"

        try:
            data = self._tesseract.image_to_data(
                image,
                lang=lang,
                config=ocr_config,
                output_type=self._tesseract.Output.DICT,
            )

            text_parts: list[str] = []
            characters: list[str] = []
            confidences: list[float] = []
            bboxes: list[tuple[int, int, int, int]] = []

            for i, conf in enumerate(data["conf"]):
                if conf == -1:
                    continue

                char_text = data["text"][i]
                if not char_text:
                    continue

                text_parts.append(char_text)

                for char in char_text:
                    characters.append(char)
                    confidences.append(float(conf) / 100.0)
                    bboxes.append(
                        (
                            data["left"][i],
                            data["top"][i],
                            data["width"][i],
                            data["height"][i],
                        )
                    )

            full_text = "".join(text_parts)
            overall_conf = (
                sum(confidences) / len(confidences) if confidences else 0.0
            )

            return OCRResult(
                text=full_text,
                confidence=overall_conf,
                characters=characters,
                char_confidences=confidences,
                bboxes=bboxes,
                raw_output=data,
            )

        except Exception as e:
            logger.error(f"Tesseract recognition failed: {e}")
            return OCRResult.empty()

    def recognize_characters(
        self, image: Image.Image, script: str
    ) -> list[tuple[str, float]]:
        result = self.recognize(image, script)
        return list(zip(result.characters, result.char_confidences))


class EasyOCREngine(OCREngine):
    """EasyOCR wrapper - Note: Does NOT support Greek."""

    SUPPORTED_SCRIPTS = {"latin", "cyrillic"}

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._readers: dict[str, Any] = {}
        self._easyocr = None
        self._available = self._check_availability()

    def _check_availability(self) -> bool:
        try:
            import easyocr

            self._easyocr = easyocr
            logger.info("EasyOCR available (Note: Greek not supported)")
            return True
        except ImportError as e:
            logger.error(f"EasyOCR not available: {e}")
            return False

    def is_available(self) -> bool:
        return self._available

    def supports_script(self, script: str) -> bool:
        return script.lower() in self.SUPPORTED_SCRIPTS

    def _get_reader(self, script: str) -> Any:
        """Get or create EasyOCR reader for a script."""
        if not self.supports_script(script):
            raise ValueError(
                f"EasyOCR does not support '{script}'. "
                f"Supported: {self.SUPPORTED_SCRIPTS}"
            )

        key = script.lower()
        if key not in self._readers:
            langs = self.config.get_easyocr_langs(script)
            use_gpu = self.config.ocr.device in ("cuda", "mps")
            self._readers[key] = self._easyocr.Reader(langs, gpu=use_gpu)
            logger.info(f"Created EasyOCR reader for {script} (GPU: {use_gpu})")
        return self._readers[key]

    def recognize(self, image: Image.Image, script: str) -> OCRResult:
        """Recognize text using EasyOCR."""
        if not self.is_available():
            raise RuntimeError("EasyOCR is not available")

        if not self.supports_script(script):
            logger.error(f"EasyOCR does not support script: {script}")
            return OCRResult.empty()

        try:
            reader = self._get_reader(script)
            img_array = np.array(image)
            results = reader.readtext(img_array)

            text_parts: list[str] = []
            characters: list[str] = []
            confidences: list[float] = []
            bboxes: list[tuple[int, int, int, int]] = []

            for bbox, text, conf in results:
                text_parts.append(text)

                for char in text:
                    characters.append(char)
                    confidences.append(float(conf))

                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                bboxes.append(
                    (
                        int(min(x_coords)),
                        int(min(y_coords)),
                        int(max(x_coords) - min(x_coords)),
                        int(max(y_coords) - min(y_coords)),
                    )
                )

            full_text = " ".join(text_parts)
            overall_conf = (
                sum(confidences) / len(confidences) if confidences else 0.0
            )

            return OCRResult(
                text=full_text,
                confidence=overall_conf,
                characters=characters,
                char_confidences=confidences,
                bboxes=bboxes,
                raw_output=results,
            )

        except Exception as e:
            logger.error(f"EasyOCR recognition failed for {script}: {e}")
            return OCRResult.empty()

    def recognize_characters(
        self, image: Image.Image, script: str
    ) -> list[tuple[str, float]]:
        result = self.recognize(image, script)
        return list(zip(result.characters, result.char_confidences))


class TrOCREngine(OCREngine):
    """TrOCR (Transformer-based OCR) wrapper."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._torch = None
        self._TrOCRProcessor = None
        self._VisionEncoderDecoderModel = None
        self._device: str | None = None
        self._model_cache: dict[str, tuple[Any, Any]] = {}
        self._available = self._check_availability()

    def _check_availability(self) -> bool:
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            import torch

            self._torch = torch
            self._TrOCRProcessor = TrOCRProcessor
            self._VisionEncoderDecoderModel = VisionEncoderDecoderModel
            self._device = self._setup_device()

            logger.info(f"TrOCR available (device: {self._device})")
            return True
        except ImportError as e:
            logger.error(f"TrOCR dependencies not available: {e}")
            return False

    def _setup_device(self) -> str:
        device = self.config.ocr.device

        if device == "cuda" and self._torch.cuda.is_available():
            logger.info("Using CUDA device")
            return "cuda"
        elif device == "mps" and self._torch.backends.mps.is_available():
            logger.info("Using Apple Silicon (MPS)")
            return "mps"
        else:
            if device != "cpu":
                logger.warning(f"{device} not available, falling back to CPU")
            return "cpu"

    def is_available(self) -> bool:
        return self._available

    def _get_model(self, script: str) -> tuple[Any, Any]:
        if script in self._model_cache:
            return self._model_cache[script]

        model_name = self.config.get_trocr_model(script)

        logger.info(f"Loading TrOCR model for {script}: {model_name}")

        processor = self._TrOCRProcessor.from_pretrained(model_name)
        model = self._VisionEncoderDecoderModel.from_pretrained(model_name)
        model.to(self._device)
        model.eval()

        self._model_cache[script] = (processor, model)
        logger.info(f"Model loaded successfully for {script}")

        return processor, model

    def recognize(self, image: Image.Image, script: str) -> OCRResult:
        if not self.is_available():
            raise RuntimeError("TrOCR is not available")

        try:
            processor, model = self._get_model(script)

            if image.mode != "RGB":
                image = image.convert("RGB")

            # Resize if too small
            width, height = image.size
            if height < 32:
                scale = 32 / height
                new_width = int(width * scale)
                image = image.resize((new_width, 32), Image.Resampling.LANCZOS)

            pixel_values = processor(images=image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self._device)

            with self._torch.no_grad():
                generated_ids = model.generate(pixel_values, max_new_tokens=128)

            generated_text = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()

            if not generated_text:
                return OCRResult.empty()

            characters = list(generated_text)
            confidences = [0.85] * len(characters)

            return OCRResult(
                text=generated_text,
                confidence=0.85,
                characters=characters,
                char_confidences=confidences,
                bboxes=None,
                raw_output={"text": generated_text, "model": "trocr"},
            )

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.error("CUDA out of memory - reduce batch_size or use CPU")
                self.cleanup()
            else:
                logger.error(f"TrOCR runtime error: {e}")
            return OCRResult.empty()
        except Exception as e:
            logger.error(f"TrOCR recognition failed: {e}")
            return OCRResult.empty()

    def recognize_characters(
        self, image: Image.Image, script: str
    ) -> list[tuple[str, float]]:
        result = self.recognize(image, script)
        return list(zip(result.characters, result.char_confidences))

    def cleanup(self) -> None:
        """Clear GPU memory."""
        if self._torch is not None and self._device == "cuda":
            self._torch.cuda.empty_cache()
            logger.debug("Cleared CUDA cache")


class SuryaOCREngine(OCREngine):
    """Surya OCR wrapper - State-of-the-art open source OCR."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._available = self._check_availability()
        self._det_predictor = None
        self._rec_predictor = None
        self._foundation_predictor = None
        self._loaded = False

    def _check_availability(self) -> bool:
        """Check if Surya is available."""
        try:
            from surya.detection import DetectionPredictor
            from surya.recognition import RecognitionPredictor, FoundationPredictor
            logger.info("Surya OCR available")
            return True
        except ImportError as e:
            logger.error(f"Surya OCR not available: {e}")
            return False

    def is_available(self) -> bool:
        return self._available

    def _lazy_load(self):
        """Load models only when needed."""
        if self._loaded:
            return

        try:
            logger.info("Loading Surya OCR models...")

            from surya.detection import DetectionPredictor
            from surya.recognition import RecognitionPredictor, FoundationPredictor

            # Initialize foundation predictor first (required by RecognitionPredictor)
            self._foundation_predictor = FoundationPredictor()

            # Initialize detection predictor
            self._det_predictor = DetectionPredictor()

            # Initialize recognition predictor with foundation predictor
            self._rec_predictor = RecognitionPredictor(self._foundation_predictor)

            self._loaded = True
            logger.info("Surya models loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Surya models: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            raise

    def recognize(self, image: Image.Image, script: str) -> OCRResult:
        if not self.is_available():
            raise RuntimeError("Surya is not installed")

        try:
            self._lazy_load()

            # Use generic OCR task (language is handled internally by Surya)
            task_names = ['ocr_without_boxes']

            # Call RecognitionPredictor directly - it handles detection internally
            rec_results = self._rec_predictor(
                images=[image],
                task_names=task_names,
                det_predictor=self._det_predictor,
                sort_lines=True,
                math_mode=False
            )

            text_lines = []
            characters = []
            char_confidences = []

            if rec_results and len(rec_results) > 0:
                result = rec_results[0]

                # Extract text from OCRResult
                if hasattr(result, 'text_lines'):
                    for text_line in result.text_lines:
                        text = getattr(text_line, 'text', '')
                        if text:
                            text_lines.append(text)
                            conf = getattr(text_line, 'confidence', 0.8)
                            for char in text:
                                characters.append(char)
                                char_confidences.append(float(conf))
                elif hasattr(result, 'text'):
                    text = result.text
                    if text:
                        text_lines.append(text)
                        for char in text:
                            characters.append(char)
                            char_confidences.append(0.8)

            full_text = "\n".join(text_lines)
            mean_conf = sum(char_confidences) / len(char_confidences) if char_confidences else 0.8

            return OCRResult(
                text=full_text,
                confidence=mean_conf,
                characters=characters,
                char_confidences=char_confidences,
                bboxes=None,
                raw_output=rec_results
            )

        except Exception as e:
            logger.error(f"Surya recognition failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return OCRResult.empty()

    def recognize_characters(
        self, image: Image.Image, script: str
    ) -> list[tuple[str, float]]:
        """Recognize individual characters with confidences."""
        result = self.recognize(image, script)
        return list(zip(result.characters, result.char_confidences))

    def cleanup(self) -> None:
        """Clean up models and free memory."""
        self._det_predictor = None
        self._rec_predictor = None
        self._foundation_predictor = None
        self._loaded = False
        logger.debug("Surya models cleaned up")


class PaddleOCREngine(OCREngine):
    """PaddleOCR wrapper - Excellent for Arabic and multilingual (v3.0+)."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._ocr = None
        self._available = self._check_availability()

    def _check_availability(self) -> bool:
        try:
            from paddleocr import PaddleOCR
            logger.info("PaddleOCR available")
            return True
        except ImportError as e:
            logger.error(f"PaddleOCR not available: {e}")
            return False

    def is_available(self) -> bool:
        return self._available

    def _lazy_load(self, script: str):
        """Load PaddleOCR only when needed"""
        if self._ocr is None:
            try:
                from paddleocr import PaddleOCR

                # Map script to language code
                lang_map = {'arabic': 'ar', 'latin': 'en', 'cyrillic': 'ru', 'greek': 'en'}
                lang = lang_map.get(script, 'en')

                logger.info(f"Loading PaddleOCR for language: {lang}")

                # PaddleOCR v3.3.2 API - removed show_log parameter
                device = "gpu" if self.config.ocr.device == "cuda" else "cpu"

                self._ocr = PaddleOCR(
                    lang=lang,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    device=device
                )
                logger.info("PaddleOCR loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load PaddleOCR: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                raise

    def recognize(self, image: Image.Image, script: str) -> OCRResult:
        if not self.is_available():
            raise RuntimeError("PaddleOCR not available")

        try:
            self._lazy_load(script)

            # Convert PIL image to numpy array for PaddleOCR
            img_array = np.array(image)

            # Use ocr() method. In v3.3.2 (PaddleX version), explicit det/rec args are unsupported.
            # Result structure: [{ 'rec_texts': [...], 'rec_scores': [...], 'dt_polys': [...] }]
            result = self._ocr.ocr(img_array)

            if not result or not isinstance(result, list):
                return OCRResult.empty()

            text_parts = []
            characters = []
            char_confidences = []
            bboxes = []

            # Check if we got any text. If not, try fallback to recognition-only on full image.
            has_text = any(len(page.get("rec_texts", [])) > 0 for page in result)
            
            if not has_text:
                logger.debug("PaddleOCR detection found no text, falling back to direct recognition")
                try:
                    import paddlex
                    # Use a recognition-only model. Map script to a reasonable model name.
                    lang_map = {'arabic': 'ar', 'latin': 'en', 'cyrillic': 'ru', 'greek': 'en'}
                    lang = lang_map.get(script, 'en')
                    rec_model_name = f"{lang}_PP-OCRv5_mobile_rec"
                    
                    # Load model (will use cache)
                    rec_model = paddlex.create_model(rec_model_name)
                    rec_gen = rec_model.predict(img_array)
                    for rec_res in rec_gen:
                        # rec_res is a TextRecResult object
                        text = getattr(rec_res, "rec_text", "")
                        score = getattr(rec_res, "rec_score", 0.0)
                        if text:
                            text_parts.append(text)
                            for char in text:
                                characters.append(char)
                                char_confidences.append(float(score))
                            # Bbox is the whole image
                            bboxes.append((0, 0, image.width, image.height))
                except Exception as e:
                    logger.warning(f"PaddleOCR fallback recognition failed: {e}")

            if not text_parts:
                # Process the original pipeline results if fallback wasn't triggered or failed
                for page_result in result:
                    texts = page_result.get("rec_texts", [])
                    scores = page_result.get("rec_scores", [])
                    polys = page_result.get("dt_polys", [])

                    for i, text in enumerate(texts):
                        if not text:
                            continue
                        text_parts.append(text)
                        score = scores[i] if i < len(scores) else 0.0
                        for char in text:
                            characters.append(char)
                            char_confidences.append(float(score))
                        
                        if i < len(polys):
                            poly = polys[i]
                            if poly is not None and len(poly) > 0:
                                xs = [p[0] for p in poly]
                                ys = [p[1] for p in poly]
                                bboxes.append((
                                    int(min(xs)), int(min(ys)),
                                    int(max(xs) - min(xs)), int(max(ys) - min(ys))
                                ))

            full_text = "\n".join(text_parts) if text_parts else ""
            mean_conf = sum(char_confidences) / len(char_confidences) if char_confidences else 0.0

            return OCRResult(
                text=full_text,
                confidence=mean_conf,
                characters=characters,
                char_confidences=char_confidences,
                bboxes=bboxes
            )

        except Exception as e:
            logger.error(f"PaddleOCR recognition failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return OCRResult.empty()

    def recognize_characters(self, image: Image.Image, script: str) -> list[tuple[str, float]]:
        res = self.recognize(image, script)
        return list(zip(res.characters, res.char_confidences))

    def cleanup(self) -> None:
        """Clean up resources"""
        self._ocr = None


class GLMOCREngine(OCREngine):
    """GLM-OCR wrapper - Multimodal OCR using GLM-V architecture."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._processor = None
        self._model = None
        self._device: str | None = None
        self._torch = None
        self._available = self._check_availability()

    def _check_availability(self) -> bool:
        try:
            from transformers import AutoProcessor, AutoModel
            import torch

            self._torch = torch
            self._device = self._setup_device()
            logger.info(f"GLM-OCR available (device: {self._device})")
            return True
        except ImportError as e:
            logger.error(f"GLM-OCR dependencies not available: {e}")
            return False

    def _setup_device(self) -> str:
        device = self.config.ocr.device
        if device == "cuda" and self._torch.cuda.is_available():
            return "cuda"
        elif device == "mps" and self._torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def is_available(self) -> bool:
        return self._available

    def _lazy_load(self):
        if self._model is None:
            from transformers import AutoProcessor, AutoModel
            model_path = "zai-org/GLM-OCR"
            logger.info(f"Loading GLM-OCR model from {model_path}...")
            
            self._processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            self._model = AutoModel.from_pretrained(
                model_path,
                torch_dtype="auto",
                trust_remote_code=True,
                device_map="auto" if self._device != "cpu" else None,
            )
            if self._device == "cpu":
                self._model.to("cpu")
            self._model.eval()
            logger.info("GLM-OCR model loaded successfully")

    def recognize(self, image: Image.Image, script: str) -> OCRResult:
        if not self.is_available():
            raise RuntimeError("GLM-OCR is not available")

        try:
            self._lazy_load()

            if image.mode != "RGB":
                image = image.convert("RGB")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "Text Recognition:"},
                    ],
                }
            ]

            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self._model.device)

            inputs.pop("token_type_ids", None)

            with self._torch.no_grad():
                generated_ids = self._model.generate(**inputs, max_new_tokens=8192)

            output_text = self._processor.decode(
                generated_ids[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            ).strip()

            if not output_text:
                return OCRResult.empty()

            characters = list(output_text)
            # Use a default confidence of 0.9 as GLM doesn't easily return character confidences via generic generate
            confidences = [0.9] * len(characters)

            return OCRResult(
                text=output_text,
                confidence=0.9,
                characters=characters,
                char_confidences=confidences,
                bboxes=None,
                raw_output={"text": output_text, "model": "glm-ocr"},
            )

        except Exception as e:
            logger.error(f"GLM-OCR recognition failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return OCRResult.empty()

    def recognize_characters(
        self, image: Image.Image, script: str
    ) -> list[tuple[str, float]]:
        result = self.recognize(image, script)
        return list(zip(result.characters, result.char_confidences))

    def cleanup(self) -> None:
        if self._torch is not None and self._device == "cuda":
            self._torch.cuda.empty_cache()
            logger.debug("Cleared CUDA cache for GLM-OCR")


def get_ocr_engine(config: PipelineConfig) -> OCREngine:
    """Factory function to get appropriate OCR engine."""
    engine_type = config.ocr.engine

    engines_to_try: list[str] = [engine_type]

    # Define fallback order (no more PaddleOCR)
    fallback_order: dict[str, list[str]] = {
        "trocr": ["tesseract", "easyocr"],
        "tesseract": ["trocr", "easyocr"],
        "easyocr": ["tesseract", "trocr"],
    }

    engines_to_try.extend(fallback_order.get(engine_type, []))

    for engine_name in engines_to_try:
        engine = _create_engine(engine_name, config)
        if engine is not None and engine.is_available():
            if engine_name != engine_type:
                logger.warning(
                    f"{engine_type} not available, using {engine_name}"
                )
            logger.info(f"Using {engine_name} engine")
            return engine

    raise RuntimeError(
        "No OCR engine available. Install one of: "
        "pytesseract, torch+transformers (TrOCR), or easyocr"
    )


def _create_engine(engine_type: str, config: PipelineConfig) -> OCREngine | None:
    """Create an OCR engine instance."""
    try:
        if engine_type == "trocr":
            return TrOCREngine(config)
        elif engine_type == "tesseract":
            return TesseractOCR(config)
        elif engine_type == "easyocr":
            return EasyOCREngine(config)
        elif engine_type == "paddle":
            return PaddleOCREngine(config)
        elif engine_type == "surya":
            return SuryaOCREngine(config)
        elif engine_type == "glm":
            return GLMOCREngine(config)
        else:
            logger.error(f"Unknown OCR engine: {engine_type}")
            return None
    except Exception as e:
        logger.error(f"Failed to create {engine_type} engine: {e}")
        return None