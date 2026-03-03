"""Data ingestion: Load images and organize by script."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator
import logging
from contextlib import contextmanager

from PIL import Image
import numpy as np

from spectral_scripts.ocr_pipeline.config import PipelineConfig, ImageConfig

logger = logging.getLogger(__name__)


@dataclass
class ImageData:
    """Container for a loaded image with metadata."""

    path: Path
    image: Image.Image
    script: str
    filename: str
    width: int
    height: int
    mode: str
    _closed: bool = False

    def to_numpy(self) -> np.ndarray:
        """Convert image to numpy array."""
        return np.array(self.image)

    def to_grayscale(self) -> Image.Image:
        """Convert to grayscale."""
        return self.image.convert("L")

    def close(self) -> None:
        """Close the image to free resources."""
        if not self._closed and self.image is not None:
            try:
                self.image.close()
            except Exception:
                pass
            self._closed = True

    def __del__(self):
        """Ensure image is closed on garbage collection."""
        self.close()


@contextmanager
def open_image(path: Path, script: str, image_config: ImageConfig):
    """
    Context manager for safely opening and closing images.

    Usage:
        with open_image(path, script, config) as img_data:
            # use img_data.image
        # image is automatically closed
    """
    img_data = None
    try:
        img = Image.open(path)

        # Validate dimensions
        if not (
            image_config.min_width <= img.width <= image_config.max_width
            and image_config.min_height <= img.height <= image_config.max_height
        ):
            img.close()
            raise ValueError(f"Image dimensions invalid: {img.width}x{img.height}")

        # Create ImageData with a copy of the image data
        img_data = ImageData(
            path=path,
            image=img.copy(),
            script=script,
            filename=path.name,
            width=img.width,
            height=img.height,
            mode=img.mode,
        )

        # Close original file handle immediately
        img.close()

        yield img_data

    finally:
        if img_data is not None:
            img_data.close()


class ImageDataLoader:
    """Load and validate images from directories."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.image_config = config.image
        self._loaded_count = 0

    def iter_images(self, script: str) -> Generator[ImageData, None, None]:
        """
        Iterator for loading images lazily.

        IMPORTANT: The caller is responsible for closing each image after use,
        OR use the context manager version below.

        Yields:
            ImageData objects (caller must call .close() when done)
        """
        script_dir = self.config.input_dir / script

        if not script_dir.exists():
            logger.warning(f"Script directory not found: {script_dir}")
            return

        count = 0
        for fmt in self.image_config.supported_formats:
            for path in sorted(script_dir.glob(f"*{fmt}")):
                try:
                    img_data = self._load_single_image(path, script)
                    if img_data is not None:
                        count += 1
                        if count % 100 == 0:
                            logger.info(f"  Loaded {count} images...")
                        yield img_data
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")

        logger.info(f"  Total loaded for {script}: {count} images")

    def iter_images_safe(self, script: str):
        """
        Generator that yields context managers for safe image handling.

        Usage:
            for img_ctx in loader.iter_images_safe(script):
                with img_ctx as img_data:
                    # process img_data
                # image is automatically closed
        """
        script_dir = self.config.input_dir / script

        if not script_dir.exists():
            logger.warning(f"Script directory not found: {script_dir}")
            return

        for fmt in self.image_config.supported_formats:
            for path in sorted(script_dir.glob(f"*{fmt}")):
                yield open_image(path, script, self.image_config)

    def _load_single_image(self, path: Path, script: str) -> ImageData | None:
        """Load and validate a single image."""
        img = None
        try:
            img = Image.open(path)

            # Validate dimensions
            if not self._validate_dimensions(img):
                logger.debug(f"Image dimensions invalid: {path}")
                return None

            # Copy image data to memory and close file handle
            img_copy = img.copy()

            return ImageData(
                path=path,
                image=img_copy,
                script=script,
                filename=path.name,
                width=img.width,
                height=img.height,
                mode=img.mode,
            )

        except Exception as e:
            logger.debug(f"Error loading image {path}: {e}")
            return None

        finally:
            # Always close the original file handle
            if img is not None:
                try:
                    img.close()
                except Exception:
                    pass

    def _validate_dimensions(self, img: Image.Image) -> bool:
        """Check if image dimensions are within acceptable range."""
        cfg = self.image_config
        return (
            cfg.min_width <= img.width <= cfg.max_width
            and cfg.min_height <= img.height <= cfg.max_height
        )

    def get_image_count(self, script: str) -> int:
        """Count images for a script without loading them."""
        script_dir = self.config.input_dir / script
        if not script_dir.exists():
            return 0

        count = 0
        for fmt in self.image_config.supported_formats:
            count += len(list(script_dir.glob(f"*{fmt}")))
        return count

    def get_image_paths(self, script: str) -> list[Path]:
        """Get all image paths for a script without loading them."""
        script_dir = self.config.input_dir / script
        if not script_dir.exists():
            return []

        paths = []
        for fmt in self.image_config.supported_formats:
            paths.extend(sorted(script_dir.glob(f"*{fmt}")))
        return paths

    def validate_directory_structure(self) -> dict[str, bool]:
        """Validate that expected directories exist."""
        results = {}
        for script in self.config.scripts:
            script_dir = self.config.input_dir / script
            results[script] = script_dir.exists() and script_dir.is_dir()
        return results


class GroundTruthLoader:
    """Load ground truth labels from text files."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._cache: dict[str, dict[str, str]] = {}

    def load_for_script(self, script: str) -> dict[str, str]:
        """
        Load ground truth labels for a script.

        Returns:
            Dictionary mapping filename to ground truth text.
        """
        # Return cached version if available
        if script in self._cache:
            return self._cache[script]

        labels_file = self.config.ground_truth_dir / f"{script}_labels.txt"

        if not labels_file.exists():
            logger.warning(f"Ground truth file not found: {labels_file}")
            self._cache[script] = {}
            return {}

        labels = {}
        with open(labels_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                if ":" not in line:
                    logger.debug(f"Invalid format at line {line_num}: {line}")
                    continue

                filename, text = line.split(":", 1)
                filename = filename.strip()
                text = text.strip().strip('"').strip("'")

                labels[filename] = text

        logger.info(f"Loaded {len(labels)} ground truth labels for {script}")
        self._cache[script] = labels
        return labels

    def load_all(self) -> dict[str, dict[str, str]]:
        """Load ground truth for all scripts."""
        return {script: self.load_for_script(script) for script in self.config.scripts}

    def get_text_for_image(self, script: str, filename: str) -> str | None:
        """Get ground truth text for a specific image."""
        labels = self.load_for_script(script)
        return labels.get(filename)

    def clear_cache(self) -> None:
        """Clear the ground truth cache."""
        self._cache.clear()