"""Configuration for OCR pipeline with support for Latin, Greek, Cyrillic, and Arabic."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import logging
import yaml

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# LANGUAGE REGISTRY - Explicitly configured for 4 scripts
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LanguageConfig:
    """Configuration for a single language/script."""

    charset: str
    tesseract_lang: str | None = None
    easyocr_langs: list[str] | None = None
    trocr_model: str = "microsoft/trocr-base-printed"
    supported_engines: set[str] = field(default_factory=lambda: {"trocr"})


# Built-in languages - explicitly configured
LANGUAGE_REGISTRY: dict[str, LanguageConfig] = {
    # ═══════════════════════════════════════════════════════════════════════════
    # LATIN - English, Western European languages
    # ═══════════════════════════════════════════════════════════════════════════
    "latin": LanguageConfig(
        charset=(
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789"
            ".,!?;:'\"-()[]{}/ "
        ),
        tesseract_lang="eng",
        easyocr_langs=["en"],
        trocr_model="microsoft/trocr-base-printed",
        supported_engines={"tesseract", "easyocr", "trocr", "paddle", "surya", "glm"},
    ),
    # ═══════════════════════════════════════════════════════════════════════════
    # GREEK - Modern Greek
    # ═══════════════════════════════════════════════════════════════════════════
    "greek": LanguageConfig(
        charset=(
            "αβγδεζηθικλμνξοπρστυφχψω"
            "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
            "άέήίόύώϊϋΐΰ"
            "0123456789"
            ".,!?;:'\"-()[]{}/ "
        ),
        tesseract_lang="ell",
        easyocr_langs=None,  # EasyOCR doesn't support Greek
        trocr_model="microsoft/trocr-base-printed",
        supported_engines={"tesseract", "trocr", "paddle", "surya", "glm"},
    ),
    # ═══════════════════════════════════════════════════════════════════════════
    # CYRILLIC - Russian, Bulgarian, Serbian, etc.
    # ═══════════════════════════════════════════════════════════════════════════
    "cyrillic": LanguageConfig(
        charset=(
            "абвгдежзийклмнопрстуфхцчшщъыьэюя"
            "АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
            "ёЁ"
            "0123456789"
            ".,!?;:'\"-()[]{}/ "
        ),
        tesseract_lang="rus",
        easyocr_langs=["ru"],
        trocr_model="microsoft/trocr-base-printed",
        supported_engines={"tesseract", "easyocr", "trocr", "paddle", "surya", "glm"},
    ),
    # ═══════════════════════════════════════════════════════════════════════════
    # ARABIC - Modern Standard Arabic (MSA) + Gulf, Levantine
    # Core 28 letters + hamzas + diacritics + numerals
    # ═══════════════════════════════════════════════════════════════════════════
    "arabic": LanguageConfig(
        charset=(
            # Core Arabic letters (28)
            "ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوى"
            # Hamza variants (often combined with vowels, but including for clarity)
            "ـ"  # Tatweel (letter extender)
            # Diacritical marks (harakat) - vowel marks
            "ًٌٍَُِّْ"
            # Additional diacritics
            "ٰٕٓٔ"
            # Arabic-Indic numerals (0-9)
            "٠١٢٣٤٥٦٧٨٩"
            # Common punctuation
            "،؛؟!-( )"
            # Tanween marks
            "ـــ"
        ),
        tesseract_lang="ara",
        easyocr_langs=["ar"],  # EasyOCR claims to support Arabic but unreliably
        trocr_model="microsoft/trocr-base-printed",
        # ★ Recommended: paddle or surya for Arabic
        supported_engines={"tesseract", "easyocr", "trocr", "paddle", "surya", "glm"},
    ),
}

def register_language(
    script: str,
    charset: str,
    tesseract_lang: str | None = None,
    easyocr_langs: list[str] | None = None,
    trocr_model: str = "microsoft/trocr-base-printed",
    supported_engines: set[str] | None = None,
) -> None:
    """
    Register a new language dynamically at runtime.

    Args:
        script: Script identifier (e.g., 'thai', 'hebrew')
        charset: String of all supported characters
        tesseract_lang: Tesseract language code
        easyocr_langs: List of EasyOCR language codes
        trocr_model: TrOCR model identifier
        supported_engines: Set of supported OCR engines

    Example:
        register_language(
            'thai',
            charset='กขคงจฉชซญยดตถทธนบปพฟมยรลวศษสหฬอ',
            tesseract_lang='tha',
            easyocr_langs=['th'],
        )
    """
    if supported_engines is None:
        supported_engines = {"trocr"}

    LANGUAGE_REGISTRY[script] = LanguageConfig(
        charset=charset,
        tesseract_lang=tesseract_lang,
        easyocr_langs=easyocr_langs,
        trocr_model=trocr_model,
        supported_engines=supported_engines,
    )
    logger.info(f"Registered language: {script}")


def get_engine_script_support() -> dict[str, set[str]]:
    """
    Dynamically build engine support from language registry.

    Returns:
        Dict mapping engine name to set of supported scripts
    """
    # Initialize support with all known engines
    support: dict[str, set[str]] = {
        "tesseract": set(),
        "easyocr": set(),
        "trocr": set(),
        "paddle": set(),
        "surya": set(),
        "glm": set(),
    }

    # Build support from language registry
    for script, config in LANGUAGE_REGISTRY.items():
        for engine in config.supported_engines:
            if engine in support:
                support[engine].add(script)

    return support


# ═══════════════════════════════════════════════════════════════════════════════
# OCR CONFIGURATION CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class OCRConfig:
    """Configuration for OCR engine."""

    engine: Literal["tesseract", "easyocr", "trocr", "paddle", "surya", "glm"] = "surya"
    confidence_threshold: float = 0.3
    psm: int = 6
    oem: int = 3

    trocr_model_type: Literal["printed", "handwritten"] = "printed"
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    batch_size: int = 32


@dataclass
class ImageConfig:
    """Configuration for image processing."""

    min_width: int = 100
    min_height: int = 50
    max_width: int = 4000
    max_height: int = 3000
    supported_formats: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tiff", ".bmp")


@dataclass
class MatchingConfig:
    """Configuration for character matching."""

    use_edit_distance: bool = True
    max_edit_distance: int = 2
    case_sensitive: bool = True
    normalize_unicode: bool = True


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""

    input_dir: Path = field(default_factory=lambda: Path("data/raw/images"))
    ground_truth_dir: Path = field(
        default_factory=lambda: Path("data/raw/ground_truth")
    )
    output_dir: Path = field(
        default_factory=lambda: Path("data/confusion_matrices")
    )

    # Default to all 4 explicitly supported scripts
    scripts: list[str] = field(
        default_factory=lambda: ["latin", "greek", "cyrillic", "arabic"]
    )

    ocr: OCRConfig = field(default_factory=OCRConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)

    max_workers: int = 4
    batch_size: int = 50
    verbose: bool = True

    def __post_init__(self):
        """Validate configuration after initialization and set up output directory."""
        self._validate_engine_script_support()
        self._validate_scripts_exist()
        
        # Ensure output directory is engine-specific to prevent overwriting
        # If output_dir is a base directory (data/confusion_matrices or outputs),
        # append the engine name as a subdirectory.
        if self.output_dir.name in ("confusion_matrices", "outputs"):
            self.output_dir = self.output_dir / self.ocr.engine
            logger.info(f"Using engine-specific output directory: {self.output_dir}")

    def _validate_scripts_exist(self) -> None:
        """Check that all requested scripts are registered."""
        available = set(LANGUAGE_REGISTRY.keys())
        requested = set(self.scripts)
        missing = requested - available

        if missing:
            raise ValueError(
                f"Unknown scripts: {missing}. "
                f"Available: {available}. "
                f"Register new scripts with register_language()."
            )

    def _validate_engine_script_support(self) -> None:
        """Check that the selected engine supports all requested scripts."""
        engine = self.ocr.engine
        support = get_engine_script_support()
        supported = support.get(engine, set())
        unsupported = set(self.scripts) - supported

        if unsupported:
            logger.warning(
                f"Engine '{engine}' does not support scripts: {unsupported}. "
                f"Supported: {supported}. "
                f"These scripts will be skipped."
            )
            self.scripts = [s for s in self.scripts if s in supported]
            if not self.scripts:
                raise ValueError(
                    f"No scripts remaining after filtering. "
                    f"Engine '{engine}' supports: {supported}. "
                    f"Available engines: {list(support.keys())}"
                )

    def get_charset(self, script: str) -> str:
        """Get character set for a script."""
        if script not in LANGUAGE_REGISTRY:
            raise ValueError(
                f"Unknown script: {script}. "
                f"Available: {list(LANGUAGE_REGISTRY.keys())}"
            )
        return LANGUAGE_REGISTRY[script].charset

    def get_tesseract_lang(self, script: str) -> str:
        """Get Tesseract language code for a script."""
        config = LANGUAGE_REGISTRY.get(script)
        if not config or not config.tesseract_lang:
            raise ValueError(
                f"Tesseract not configured for '{script}'. "
                f"Use register_language() to add support."
            )
        return config.tesseract_lang

    def get_easyocr_langs(self, script: str) -> list[str]:
        """Get EasyOCR language codes for a script."""
        config = LANGUAGE_REGISTRY.get(script)
        if not config or not config.easyocr_langs:
            raise ValueError(
                f"EasyOCR not configured for '{script}'. "
                f"Use register_language() to add support."
            )
        return config.easyocr_langs

    def get_trocr_model(self, script: str) -> str:
        """Get TrOCR model for a script."""
        config = LANGUAGE_REGISTRY.get(script)
        if not config:
            raise ValueError(f"Unknown script: {script}")
        return config.trocr_model

    def engine_supports_script(self, script: str) -> bool:
        """Check if current engine supports a script."""
        config = LANGUAGE_REGISTRY.get(script)
        if not config:
            return False
        return self.ocr.engine in config.supported_engines

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        if "ocr" in data:
            ocr_data = dict(data["ocr"])
            ocr_data.pop("paddleocr_use_angle_cls", None)
            ocr_data.pop("paddleocr_use_gpu", None)
            data["ocr"] = OCRConfig(**ocr_data)

        if "image" in data:
            if "supported_formats" in data["image"]:
                data["image"]["supported_formats"] = tuple(
                    data["image"]["supported_formats"]
                )
            data["image"] = ImageConfig(**data["image"])

        if "matching" in data:
            data["matching"] = MatchingConfig(**data["matching"])

        for key in ["input_dir", "ground_truth_dir", "output_dir"]:
            if key in data:
                data[key] = Path(data[key])

        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "input_dir": str(self.input_dir),
            "ground_truth_dir": str(self.ground_truth_dir),
            "output_dir": str(self.output_dir),
            "scripts": self.scripts,
            "ocr": {
                "engine": self.ocr.engine,
                "confidence_threshold": self.ocr.confidence_threshold,
                "psm": self.ocr.psm,
                "oem": self.ocr.oem,
                "trocr_model_type": self.ocr.trocr_model_type,
                "device": self.ocr.device,
                "batch_size": self.ocr.batch_size,
            },
            "image": {
                "min_width": self.image.min_width,
                "min_height": self.image.min_height,
                "max_width": self.image.max_width,
                "max_height": self.image.max_height,
                "supported_formats": list(self.image.supported_formats),
            },
            "matching": {
                "use_edit_distance": self.matching.use_edit_distance,
                "max_edit_distance": self.matching.max_edit_distance,
                "case_sensitive": self.matching.case_sensitive,
                "normalize_unicode": self.matching.normalize_unicode,
            },
            "max_workers": self.max_workers,
            "batch_size": self.batch_size,
            "verbose": self.verbose,
        }

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)