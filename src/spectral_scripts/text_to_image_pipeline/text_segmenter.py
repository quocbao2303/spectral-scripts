"""Segment text into chunks for image rendering."""

from __future__ import annotations

from dataclasses import dataclass
import logging

from spectral_scripts.text_to_image_pipeline.config import TextToImageConfig
from spectral_scripts.text_to_image_pipeline.text_loader import LoadedText

logger = logging.getLogger(__name__)


@dataclass
class TextSegment:
    """A segment of text ready for rendering."""
    
    text: str
    source_file: str
    segment_index: int
    script: str
    
    @property
    def char_count(self) -> int:
        return len(self.text)
    
    @property
    def line_count(self) -> int:
        return len(self.text.split("\n"))
    
    def get_characters(self) -> list[str]:
        """Get list of unique characters in this segment."""
        return sorted(set(self.text))


class TextSegmenter:
    """Segment loaded text into chunks for rendering."""
    
    def __init__(self, config: TextToImageConfig):
        self.config = config
        self.segment_config = config.segment
    
    def segment_text(self, loaded: LoadedText) -> list[TextSegment]:
        """Segment a loaded text into renderable chunks."""
        text = loaded.cleaned_text
        if not text:
            return []
        
        segments = []
        
        # Split into lines first
        lines = text.split("\n")
        
        current_segment = []
        current_chars = 0
        segment_index = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if adding this line would exceed limits
            line_chars = len(line)
            
            if (current_chars + line_chars > self.segment_config.chars_per_segment or
                len(current_segment) >= self.segment_config.max_lines_per_segment):
                
                # Save current segment if it has content
                if current_segment:
                    segment_text = "\n".join(current_segment)
                    if len(segment_text) >= self.segment_config.min_chars_per_segment:
                        segments.append(TextSegment(
                            text=segment_text,
                            source_file=loaded.filename,
                            segment_index=segment_index,
                            script=loaded.script,
                        ))
                        segment_index += 1
                
                # Start new segment
                current_segment = [line]
                current_chars = line_chars
            else:
                current_segment.append(line)
                current_chars += line_chars
        
        # Don't forget the last segment
        if current_segment:
            segment_text = "\n".join(current_segment)
            if len(segment_text) >= self.segment_config.min_chars_per_segment:
                segments.append(TextSegment(
                    text=segment_text,
                    source_file=loaded.filename,
                    segment_index=segment_index,
                    script=loaded.script,
                ))
        
        return segments
    
    def segment_all(self, texts: list[LoadedText]) -> list[TextSegment]:
        """Segment all loaded texts."""
        all_segments = []
        
        for loaded in texts:
            segments = self.segment_text(loaded)
            
            # Apply limit if configured
            if self.config.max_segments_per_file:
                segments = segments[:self.config.max_segments_per_file]
            
            all_segments.extend(segments)
            logger.debug(f"  {loaded.filename}: {len(segments)} segments")
        
        logger.info(f"Created {len(all_segments)} segments from {len(texts)} files")
        return all_segments
    
    def get_character_coverage(self, segments: list[TextSegment], script: str) -> dict:
        """Analyze character coverage across segments."""
        charset = set(self.config.get_charset(script))
        
        # Count occurrences of each character
        char_counts = {}
        for segment in segments:
            for char in segment.text:
                if char in charset:
                    char_counts[char] = char_counts.get(char, 0) + 1
        
        # Find missing characters
        covered = set(char_counts.keys())
        missing = charset - covered - {"\n", " "}
        
        return {
            "total_charset": len(charset),
            "covered": len(covered),
            "missing": sorted(missing),
            "coverage_percent": len(covered) / len(charset) * 100 if charset else 0,
            "char_counts": dict(sorted(char_counts.items(), key=lambda x: -x[1])),
        }