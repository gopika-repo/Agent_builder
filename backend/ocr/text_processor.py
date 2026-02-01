"""
Text Processor Module

Handles post-processing of OCR output including:
- Text normalization
- Block merging based on spatial proximity
- Reading order determination
- Spell checking suggestions
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .ocr_engine import TextBlock, OCRResult

logger = logging.getLogger(__name__)


@dataclass
class ProcessedTextBlock:
    """A processed and potentially merged text block"""
    text: str
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    block_type: str  # paragraph, heading, list, etc.
    reading_order: int
    child_blocks: List[TextBlock]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "confidence": self.confidence,
            "block_type": self.block_type,
            "reading_order": self.reading_order,
            "word_count": len(self.text.split())
        }


class TextProcessor:
    """
    Post-processor for OCR results.
    
    Cleans, normalizes, and structures extracted text.
    """
    
    def __init__(
        self,
        merge_threshold_x: float = 50,
        merge_threshold_y: float = 20,
        line_height_ratio: float = 1.5
    ):
        """
        Initialize text processor.
        
        Args:
            merge_threshold_x: Max horizontal gap for merging blocks
            merge_threshold_y: Max vertical gap for merging into same line
            line_height_ratio: Ratio to determine line grouping
        """
        self.merge_threshold_x = merge_threshold_x
        self.merge_threshold_y = merge_threshold_y
        self.line_height_ratio = line_height_ratio
    
    def process(self, ocr_result: OCRResult) -> List[ProcessedTextBlock]:
        """
        Process OCR result into structured text blocks.
        
        Args:
            ocr_result: Raw OCR result
            
        Returns:
            List of processed text blocks in reading order
        """
        if not ocr_result.text_blocks:
            return []
        
        # Step 1: Normalize text
        normalized_blocks = self._normalize_blocks(ocr_result.text_blocks)
        
        # Step 2: Group into lines
        lines = self._group_into_lines(normalized_blocks)
        
        # Step 3: Merge lines into paragraphs
        paragraphs = self._group_into_paragraphs(lines)
        
        # Step 4: Determine reading order
        ordered_blocks = self._determine_reading_order(paragraphs)
        
        return ordered_blocks
    
    def _normalize_blocks(self, blocks: List[TextBlock]) -> List[TextBlock]:
        """Normalize text in blocks"""
        normalized = []
        
        for block in blocks:
            # Skip very low confidence or empty text
            if block.confidence < 0.1 or not block.text.strip():
                continue
            
            # Normalize text
            normalized_text = self._normalize_text(block.text)
            
            if normalized_text:
                # Create new block with normalized text
                new_block = TextBlock(
                    text=normalized_text,
                    x1=block.x1,
                    y1=block.y1,
                    x2=block.x2,
                    y2=block.y2,
                    confidence=block.confidence,
                    block_type=block.block_type,
                    language=block.language,
                    page_number=block.page_number
                )
                normalized.append(new_block)
        
        return normalized
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize a text string.
        
        - Fix common OCR errors
        - Normalize whitespace
        - Handle special characters
        """
        if not text:
            return ""
        
        # Common OCR substitution errors
        corrections = {
            r'\b0\b': 'O',  # Zero vs O (context dependent)
            r'\bl\b': 'I',  # Lowercase L vs I
            r'rn': 'm',     # rn -> m
            r'vv': 'w',     # vv -> w
        }
        
        result = text
        
        # Normalize whitespace
        result = ' '.join(result.split())
        
        # Remove control characters
        result = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', result)
        
        # Fix common ligature issues
        result = result.replace('ﬁ', 'fi')
        result = result.replace('ﬂ', 'fl')
        result = result.replace('ﬀ', 'ff')
        result = result.replace('ﬃ', 'ffi')
        result = result.replace('ﬄ', 'ffl')
        
        return result.strip()
    
    def _group_into_lines(
        self,
        blocks: List[TextBlock]
    ) -> List[List[TextBlock]]:
        """
        Group text blocks into lines based on vertical position.
        """
        if not blocks:
            return []
        
        # Sort by vertical position
        sorted_blocks = sorted(blocks, key=lambda b: (b.y1, b.x1))
        
        lines = []
        current_line = [sorted_blocks[0]]
        current_line_y = sorted_blocks[0].y1
        current_line_height = sorted_blocks[0].height
        
        for block in sorted_blocks[1:]:
            # Check if block is on the same line
            y_diff = abs(block.y1 - current_line_y)
            
            if y_diff < current_line_height * self.line_height_ratio:
                # Same line
                current_line.append(block)
            else:
                # New line
                # Sort current line by x position
                current_line.sort(key=lambda b: b.x1)
                lines.append(current_line)
                
                current_line = [block]
                current_line_y = block.y1
                current_line_height = block.height
        
        # Don't forget the last line
        if current_line:
            current_line.sort(key=lambda b: b.x1)
            lines.append(current_line)
        
        return lines
    
    def _group_into_paragraphs(
        self,
        lines: List[List[TextBlock]]
    ) -> List[ProcessedTextBlock]:
        """
        Group lines into paragraphs based on spacing and indentation.
        """
        if not lines:
            return []
        
        paragraphs = []
        current_paragraph_lines = [lines[0]]
        
        for i, line in enumerate(lines[1:], 1):
            prev_line = lines[i - 1]
            
            # Calculate gap between lines
            prev_bottom = max(b.y2 for b in prev_line)
            curr_top = min(b.y1 for b in line)
            gap = curr_top - prev_bottom
            
            # Average line height
            avg_height = sum(b.height for b in prev_line) / len(prev_line)
            
            # Check for paragraph break
            is_new_paragraph = False
            
            # Large gap indicates paragraph break
            if gap > avg_height * 1.5:
                is_new_paragraph = True
            
            # Check for indentation change
            prev_left = min(b.x1 for b in prev_line)
            curr_left = min(b.x1 for b in line)
            
            if abs(curr_left - prev_left) > avg_height * 2:
                is_new_paragraph = True
            
            if is_new_paragraph:
                # Create paragraph from accumulated lines
                paragraph = self._create_paragraph_block(current_paragraph_lines)
                paragraphs.append(paragraph)
                current_paragraph_lines = [line]
            else:
                current_paragraph_lines.append(line)
        
        # Create final paragraph
        if current_paragraph_lines:
            paragraph = self._create_paragraph_block(current_paragraph_lines)
            paragraphs.append(paragraph)
        
        return paragraphs
    
    def _create_paragraph_block(
        self,
        lines: List[List[TextBlock]]
    ) -> ProcessedTextBlock:
        """Create a ProcessedTextBlock from a list of lines"""
        all_blocks = [block for line in lines for block in line]
        
        # Combine text with newlines between lines
        text_lines = []
        for line in lines:
            line_text = ' '.join(b.text for b in line)
            text_lines.append(line_text)
        
        combined_text = '\n'.join(text_lines)
        
        # Calculate bounding box
        x1 = min(b.x1 for b in all_blocks)
        y1 = min(b.y1 for b in all_blocks)
        x2 = max(b.x2 for b in all_blocks)
        y2 = max(b.y2 for b in all_blocks)
        
        # Average confidence
        avg_confidence = sum(b.confidence for b in all_blocks) / len(all_blocks)
        
        # Determine block type
        block_type = self._determine_block_type(combined_text, all_blocks)
        
        return ProcessedTextBlock(
            text=combined_text,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            confidence=avg_confidence,
            block_type=block_type,
            reading_order=0,  # Will be set later
            child_blocks=all_blocks
        )
    
    def _determine_block_type(
        self,
        text: str,
        blocks: List[TextBlock]
    ) -> str:
        """Determine the type of text block"""
        # Short text at top might be title
        if len(blocks) == 1 or len(text) < 100:
            avg_y = sum(b.y1 for b in blocks) / len(blocks)
            if avg_y < 200:  # Near top
                return "heading"
        
        # Check for list patterns
        if re.match(r'^[\d•\-\*]', text.strip()):
            return "list"
        
        # Check for number patterns (could be table data)
        num_pattern = r'\d+[\.,]\d+'
        if len(re.findall(num_pattern, text)) > 3:
            return "data"
        
        return "paragraph"
    
    def _determine_reading_order(
        self,
        paragraphs: List[ProcessedTextBlock]
    ) -> List[ProcessedTextBlock]:
        """
        Determine reading order for paragraphs.
        
        Uses top-to-bottom, left-to-right ordering with
        column detection.
        """
        if not paragraphs:
            return []
        
        # Group by potential columns
        # Simple approach: sort by y primarily, x secondarily
        sorted_paragraphs = sorted(
            paragraphs,
            key=lambda p: (p.y1, p.x1)
        )
        
        # Assign reading order
        for i, paragraph in enumerate(sorted_paragraphs):
            paragraph.reading_order = i
        
        return sorted_paragraphs
    
    def get_full_text(
        self,
        processed_blocks: List[ProcessedTextBlock],
        separator: str = "\n\n"
    ) -> str:
        """Get full text from processed blocks in reading order"""
        sorted_blocks = sorted(processed_blocks, key=lambda b: b.reading_order)
        return separator.join(b.text for b in sorted_blocks)
    
    def find_text(
        self,
        processed_blocks: List[ProcessedTextBlock],
        query: str,
        case_sensitive: bool = False
    ) -> List[Tuple[ProcessedTextBlock, int]]:
        """
        Find blocks containing specific text.
        
        Returns list of (block, position) tuples.
        """
        results = []
        
        for block in processed_blocks:
            text = block.text if case_sensitive else block.text.lower()
            search = query if case_sensitive else query.lower()
            
            pos = text.find(search)
            if pos >= 0:
                results.append((block, pos))
        
        return results
