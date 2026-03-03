"""Configuration for text-to-image pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import logging

logger = logging.getLogger(__name__)

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Font paths for different scripts
FONT_PATHS = {
    "latin": PROJECT_ROOT / "fonts" / "NotoSans-Regular.ttf",
    "greek": PROJECT_ROOT / "fonts" / "NotoSans-Regular.ttf",
    "cyrillic": PROJECT_ROOT / "fonts" / "NotoSans-Regular.ttf",
    "arabic": PROJECT_ROOT / "fonts" / "NotoSansArabic-Regular.ttf",
}

# Character sets for each script
CHARSETS = {
    "latin": (
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        ".,!?;:'\"-()[]{}/ "
    ),
    "greek": (
        "αβγδεζηθικλμνξοπρστυφχψω"
        "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
        "άέήίόύώϊϋΐΰ"
        "0123456789"
        ".,!?;:'\"-()[]{}/ "
    ),
    "cyrillic": (
        "абвгдежзийклмнопрстуфхцчшщъыьэюя"
        "АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
        "ёЁ"
        "0123456789"
        ".,!?;:'\"-()[]{}/ "
    ),
    "arabic": (
        # Core Arabic letters (28)
        "ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوى"
        # Hamza variant
        "ـ"  # Tatweel (letter extender)
        # Diacritical marks (harakat) - vowel marks
        "ًٌٍَُِّْ"
        # Additional diacritics
        "ٰٕٓٔ"
        # Arabic-Indic numerals (0-9)
        "٠١٢٣٤٥٦٧٨٩"
        # Common punctuation
        "،؛؟!-( )"
    ),
}


@dataclass
class RenderConfig:
    """Configuration for image rendering."""

    font_size: int = 24
    line_spacing: float = 1.4
    padding: int = 20
    background_color: tuple[int, int, int] = (255, 255, 255)
    text_color: tuple[int, int, int] = (0, 0, 0)
    dpi: int = 150
    max_width: int = 800
    max_height: int = 600


@dataclass
class SegmentConfig:
    """Configuration for text segmentation."""

    chars_per_segment: int = 100
    min_chars_per_segment: int = 20
    max_lines_per_segment: int = 5
    preserve_words: bool = True


@dataclass
class AugmentationConfig:
    """Configuration for image augmentation."""

    enabled: bool = True
    num_variations: int = 3

    # Noise
    noise_probability: float = 0.3
    noise_intensity: float = 0.02

    # Blur
    blur_probability: float = 0.3
    blur_kernel: int = 3

    # Rotation
    rotation_probability: float = 0.3
    max_rotation_degrees: float = 2.0

    # Brightness/Contrast
    brightness_probability: float = 0.3
    brightness_range: tuple[float, float] = (0.8, 1.2)
    contrast_range: tuple[float, float] = (0.8, 1.2)


@dataclass
class TextToImageConfig:
    """Complete configuration for text-to-image pipeline."""

    # Paths
    input_dir: Path = field(default_factory=lambda: Path("data/raw/texts"))
    output_dir: Path = field(default_factory=lambda: Path("data/raw"))

    # Scripts to process
    # Supported: latin, greek, cyrillic, arabic
    scripts: list[str] = field(default_factory=lambda: ["latin", "greek", "cyrillic", "arabic"])

    # Component configs
    render: RenderConfig = field(default_factory=RenderConfig)
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)

    # Processing
    max_segments_per_file: int | None = None  # None = unlimited
    random_seed: int = 42
    verbose: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Convert string paths to Path objects
        if isinstance(self.input_dir, str):
            self.input_dir = Path(self.input_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

        # Validate scripts exist in registry
        available_scripts = set(CHARSETS.keys())
        invalid_scripts = set(self.scripts) - available_scripts
        if invalid_scripts:
            logger.warning(
                f"Unknown scripts: {invalid_scripts}. "
                f"Available: {available_scripts}"
            )

        # Validate fonts exist
        for script in self.scripts:
            font_path = FONT_PATHS.get(script)
            if font_path and not font_path.exists():
                logger.warning(
                    f"Font not found for {script}: {font_path}. "
                    f"Please download from Google Fonts (https://fonts.google.com/noto)"
                )

    def get_charset(self, script: str) -> str:
        """Get character set for a script."""
        if script not in CHARSETS:
            logger.warning(f"Unknown script: {script}")
            return ""
        return CHARSETS[script]

    def get_font_path(self, script: str) -> Path:
        """Get font path for a script."""
        if script not in FONT_PATHS:
            logger.warning(f"No font configured for script: {script}")
            return FONT_PATHS.get("latin", Path("fonts/NotoSans-Regular.ttf"))
        return FONT_PATHS[script]

    def get_input_path(self, script: str) -> Path:
        """Get input directory for a script."""
        return self.input_dir / script

    def get_output_images_path(self, script: str) -> Path:
        """Get output images directory for a script."""
        return self.output_dir / "images" / script

    def get_output_labels_path(self, script: str) -> Path:
        """Get output labels file path for a script."""
        return self.output_dir / "ground_truth" / f"{script}_labels.txt"