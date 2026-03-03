"""Load and clean text files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import unicodedata
import logging
import re

from spectral_scripts.text_to_image_pipeline.config import TextToImageConfig

logger = logging.getLogger(__name__)


@dataclass
class LoadedText:
    """Container for loaded text with metadata."""
    
    filename: str
    script: str
    original_text: str
    cleaned_text: str
    line_count: int
    char_count: int
    
    @property
    def is_empty(self) -> bool:
        return len(self.cleaned_text.strip()) == 0


class TextLoader:
    """Load and preprocess text files."""
    
    def __init__(self, config: TextToImageConfig):
        self.config = config
    
    def load_file(self, path: Path, script: str) -> LoadedText:
        """Load and clean a single text file."""
        # Try different encodings
        for encoding in ["utf-8", "utf-8-sig", "cp1251", "iso-8859-1"]:
            try:
                with open(path, "r", encoding=encoding) as f:
                    text = f.read()
                break
            except UnicodeDecodeError:
                continue
        else:
            logger.warning(f"Could not decode {path}, skipping")
            return LoadedText(
                filename=path.name,
                script=script,
                original_text="",
                cleaned_text="",
                line_count=0,
                char_count=0,
            )
        
        # Clean the text
        cleaned = self._clean_text(text, script)
        
        return LoadedText(
            filename=path.name,
            script=script,
            original_text=text,
            cleaned_text=cleaned,
            line_count=len(cleaned.split("\n")),
            char_count=len(cleaned),
        )
    
    def load_all_for_script(self, script: str) -> list[LoadedText]:
        """Load all text files for a script."""
        input_dir = self.config.get_input_path(script)
        
        if not input_dir.exists():
            logger.warning(f"Input directory not found: {input_dir}")
            return []
        
        texts = []
        for txt_file in sorted(input_dir.glob("*.txt")):
            loaded = self.load_file(txt_file, script)
            if not loaded.is_empty:
                texts.append(loaded)
                logger.debug(f"  Loaded {txt_file.name}: {loaded.char_count} chars")
        
        logger.info(f"Loaded {len(texts)} files for {script}")
        return texts
    
    def load_all(self) -> dict[str, list[LoadedText]]:
        """Load all text files for all scripts."""
        all_texts = {}
        for script in self.config.scripts:
            all_texts[script] = self.load_all_for_script(script)
        return all_texts
    
    def _clean_text(self, text: str, script: str) -> str:
        """Clean and normalize text."""
        # Normalize Unicode
        text = unicodedata.normalize("NFC", text)
        
        # Remove/replace problematic characters
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")
        text = text.replace("\t", " ")
        
        # Remove multiple spaces
        text = re.sub(r" +", " ", text)
        
        # Remove multiple newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        
        # Filter to valid charset (optional, can be strict or lenient)
        charset = set(self.config.get_charset(script))
        if charset:
            # Keep characters that are in charset or are newlines
            filtered_chars = []
            for char in text:
                if char in charset or char == "\n":
                    filtered_chars.append(char)
                elif char.isspace():
                    filtered_chars.append(" ")
                # Skip characters not in charset
            text = "".join(filtered_chars)
        
        # Clean up again after filtering
        text = re.sub(r" +", " ", text)
        text = text.strip()
        
        return text
    
    def get_statistics(self, texts: dict[str, list[LoadedText]]) -> dict:
        """Get statistics about loaded texts."""
        stats = {}
        for script, text_list in texts.items():
            total_chars = sum(t.char_count for t in text_list)
            total_lines = sum(t.line_count for t in text_list)
            
            # Character frequency
            char_freq = {}
            for t in text_list:
                for char in t.cleaned_text:
                    char_freq[char] = char_freq.get(char, 0) + 1
            
            stats[script] = {
                "files": len(text_list),
                "total_chars": total_chars,
                "total_lines": total_lines,
                "unique_chars": len(char_freq),
                "top_chars": sorted(char_freq.items(), key=lambda x: -x[1])[:20],
            }
        
        return stats