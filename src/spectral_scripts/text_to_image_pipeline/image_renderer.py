"""Render text segments as images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from spectral_scripts.text_to_image_pipeline.config import TextToImageConfig
from spectral_scripts.text_to_image_pipeline.text_segmenter import TextSegment

logger = logging.getLogger(__name__)

# Font path - relative to project root (where you run make from)
FONT_PATH = Path.cwd() / "fonts" / "NotoSans-Regular.ttf"


def get_font_path() -> Path:
    """Get the NotoSans font path."""
    if FONT_PATH.exists():
        return FONT_PATH
    
    raise FileNotFoundError(
        f"Font not found: {FONT_PATH}\n"
        f"Please ensure NotoSans-Regular.ttf is in the 'fonts' directory.\n"
        f"Current working directory: {Path.cwd()}\n"
        f"Download from: https://fonts.google.com/noto/specimen/Noto+Sans"
    )


@dataclass
class RenderedImage:
    """A rendered image with metadata."""

    image: Image.Image
    segment: TextSegment
    font_name: str
    augmentation: str | None
    filename: str

    @property
    def ground_truth(self) -> str:
        """Get ground truth text (normalized)."""
        return self.segment.text.replace("\n", " ")


class ImageRenderer:
    """Render text segments as images."""

    def __init__(self, config: TextToImageConfig):
        self.config = config
        self.render_config = config.render
        self.aug_config = config.augmentation
        self._font: ImageFont.FreeTypeFont | None = None
        self._font_name: str = "NotoSans-Regular"
        self._rng = np.random.default_rng(config.random_seed)
        self._load_font()

    def _load_font(self) -> None:
        """Load the NotoSans font."""
        font_path = get_font_path()
        
        self._font = ImageFont.truetype(str(font_path), self.render_config.font_size)
        self._font_name = font_path.stem
        
        logger.info(f"✓ Loaded font: {font_path} (size={self.render_config.font_size})")

    def render_segment(
        self,
        segment: TextSegment,
        apply_augmentation: bool = False,
    ) -> list[RenderedImage]:
        """Render a text segment as one or more images."""
        base_image = self._render_text(segment.text)
        results = []

        filename = (
            f"{segment.script}_{segment.source_file}_{segment.segment_index:04d}"
            f"_{self._font_name}.png"
        ).replace(".txt", "")

        results.append(
            RenderedImage(
                image=base_image.copy(),
                segment=segment,
                font_name=self._font_name,
                augmentation="original",
                filename=filename,
            )
        )

        if apply_augmentation and self.aug_config.enabled:
            for i in range(self.aug_config.num_variations):
                aug_image, aug_type = self._apply_augmentation(base_image)
                aug_filename = (
                    f"{segment.script}_{segment.source_file}_{segment.segment_index:04d}"
                    f"_{self._font_name}_aug{i}.png"
                ).replace(".txt", "")

                results.append(
                    RenderedImage(
                        image=aug_image,
                        segment=segment,
                        font_name=self._font_name,
                        augmentation=aug_type,
                        filename=aug_filename,
                    )
                )

        return results

    def _render_text(self, text: str) -> Image.Image:
        """Render text to an image."""
        cfg = self.render_config
        lines = text.split("\n")

        # Calculate dimensions
        dummy_img = Image.new("RGB", (1, 1))
        dummy_draw = ImageDraw.Draw(dummy_img)

        line_heights = []
        line_widths = []
        for line in lines:
            bbox = dummy_draw.textbbox((0, 0), line, font=self._font)
            line_widths.append(bbox[2] - bbox[0])
            line_heights.append(bbox[3] - bbox[1])

        max_width = max(line_widths) if line_widths else 100
        line_height = max(line_heights) if line_heights else cfg.font_size

        img_width = min(max_width + 2 * cfg.padding, cfg.max_width)
        img_height = min(
            int(len(lines) * line_height * cfg.line_spacing + 2 * cfg.padding),
            cfg.max_height,
        )

        # Create and draw
        image = Image.new("RGB", (img_width, img_height), cfg.background_color)
        draw = ImageDraw.Draw(image)

        y = cfg.padding
        for line in lines:
            if y + line_height > img_height - cfg.padding:
                break
            draw.text((cfg.padding, y), line, fill=cfg.text_color, font=self._font)
            y += int(line_height * cfg.line_spacing)

        return image

    def _apply_augmentation(self, image: Image.Image) -> tuple[Image.Image, str]:
        """Apply random augmentation to an image."""
        img = image.copy()
        augmentations_applied = []

        if self._rng.random() < self.aug_config.noise_probability:
            img = self._add_noise(img)
            augmentations_applied.append("noise")

        if self._rng.random() < self.aug_config.blur_probability:
            img = img.filter(
                ImageFilter.GaussianBlur(radius=self.aug_config.blur_kernel / 2)
            )
            augmentations_applied.append("blur")

        if self._rng.random() < self.aug_config.rotation_probability:
            angle = self._rng.uniform(
                -self.aug_config.max_rotation_degrees,
                self.aug_config.max_rotation_degrees,
            )
            img = img.rotate(
                angle, fillcolor=self.render_config.background_color, expand=False
            )
            augmentations_applied.append("rotation")

        if self._rng.random() < self.aug_config.brightness_probability:
            img = self._adjust_brightness_contrast(img)
            augmentations_applied.append("brightness")

        aug_type = "+".join(augmentations_applied) if augmentations_applied else "none"
        return img, aug_type

    def _add_noise(self, image: Image.Image) -> Image.Image:
        """Add Gaussian noise to image."""
        img_array = np.array(image, dtype=np.float32)
        noise = self._rng.normal(
            0, self.aug_config.noise_intensity * 255, img_array.shape
        )
        noisy = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)

    def _adjust_brightness_contrast(self, image: Image.Image) -> Image.Image:
        """Adjust brightness and contrast."""
        img_array = np.array(image, dtype=np.float32)

        brightness = self._rng.uniform(*self.aug_config.brightness_range)
        img_array = img_array * brightness

        contrast = self._rng.uniform(*self.aug_config.contrast_range)
        mean = img_array.mean()
        img_array = (img_array - mean) * contrast + mean

        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)

    def render_all(
        self,
        segments: list[TextSegment],
        script: str,
        apply_augmentation: bool = True,
    ) -> list[RenderedImage]:
        """Render all segments for a script."""
        all_rendered = []

        for i, segment in enumerate(segments):
            rendered = self.render_segment(
                segment, apply_augmentation=apply_augmentation
            )
            all_rendered.extend(rendered)

            if (i + 1) % 100 == 0:
                logger.info(f"  Rendered {i + 1}/{len(segments)} segments...")

        logger.info(f"✓ Rendered {len(all_rendered)} images for {script}")
        return all_rendered