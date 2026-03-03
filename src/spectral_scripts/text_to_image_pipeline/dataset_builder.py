"""Build the final dataset structure for OCR pipeline."""

from __future__ import annotations

from pathlib import Path
import logging
import json
from datetime import datetime

from spectral_scripts.text_to_image_pipeline.config import TextToImageConfig
from spectral_scripts.text_to_image_pipeline.text_loader import TextLoader
from spectral_scripts.text_to_image_pipeline.text_segmenter import TextSegmenter
from spectral_scripts.text_to_image_pipeline.image_renderer import ImageRenderer, RenderedImage

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """Build complete dataset from text files."""
    
    def __init__(self, config: TextToImageConfig):
        self.config = config
        self.loader = TextLoader(config)
        self.segmenter = TextSegmenter(config)
        self.renderer = ImageRenderer(config)
    
    def build_for_script(self, script: str) -> dict:
        """Build dataset for a single script."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Building dataset for: {script.upper()}")
        logger.info(f"{'='*60}")
        
        stats = {
            "script": script,
            "started_at": datetime.now().isoformat(),
        }
        
        # Step 1: Load text files
        logger.info("Step 1: Loading text files...")
        texts = self.loader.load_all_for_script(script)
        stats["files_loaded"] = len(texts)
        stats["total_chars_loaded"] = sum(t.char_count for t in texts)
        
        if not texts:
            logger.warning(f"No text files found for {script}")
            return stats
        
        # Step 2: Segment text
        logger.info("Step 2: Segmenting text...")
        segments = self.segmenter.segment_all(texts)
        stats["segments_created"] = len(segments)
        
        # Character coverage analysis
        coverage = self.segmenter.get_character_coverage(segments, script)
        stats["character_coverage"] = coverage["coverage_percent"]
        stats["missing_characters"] = coverage["missing"]
        
        # Step 3: Render images
        logger.info("Step 3: Rendering images...")
        rendered = self.renderer.render_all(
            segments, 
            script, 
            apply_augmentation=self.config.augmentation.enabled
        )
        stats["images_rendered"] = len(rendered)
        
        # Step 4: Save images and labels
        logger.info("Step 4: Saving images and labels...")
        self._save_dataset(rendered, script)
        
        stats["completed_at"] = datetime.now().isoformat()
        
        return stats
    
    def build_all(self) -> dict:
        """Build datasets for all configured scripts."""
        all_stats = {
            "started_at": datetime.now().isoformat(),
            "config": {
                "chars_per_segment": self.config.segment.chars_per_segment,
                "augmentation_enabled": self.config.augmentation.enabled,
                "num_variations": self.config.augmentation.num_variations,
            },
            "scripts": {},
        }
        
        for script in self.config.scripts:
            stats = self.build_for_script(script)
            all_stats["scripts"][script] = stats
        
        all_stats["completed_at"] = datetime.now().isoformat()
        
        # Save summary
        summary_path = self.config.output_dir / "dataset_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\nDataset summary saved to: {summary_path}")
        
        return all_stats
    
    def _save_dataset(self, rendered: list[RenderedImage], script: str) -> None:
        """Save rendered images and ground truth labels."""
        # Create output directories
        images_dir = self.config.get_output_images_path(script)
        images_dir.mkdir(parents=True, exist_ok=True)
        
        labels_path = self.config.get_output_labels_path(script)
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save images and collect labels
        labels = []
        
        for rendered_img in rendered:
            # Save image
            img_path = images_dir / rendered_img.filename
            rendered_img.image.save(img_path, "PNG")
            
            # Collect label
            ground_truth = rendered_img.ground_truth
            labels.append(f'{rendered_img.filename}: "{ground_truth}"')
        
        # Save labels file
        with open(labels_path, "w", encoding="utf-8") as f:
            f.write(f"# Ground truth labels for {script}\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write(f"# Total images: {len(labels)}\n")
            f.write("#\n")
            for label in labels:
                f.write(label + "\n")
        
        logger.info(f"  Saved {len(rendered)} images to: {images_dir}")
        logger.info(f"  Saved labels to: {labels_path}")