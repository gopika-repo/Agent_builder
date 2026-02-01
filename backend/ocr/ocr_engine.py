"""
OCR Engine Module

Provides text extraction capabilities using Tesseract (primary)
and EasyOCR (fallback) with coordinate-aware output.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

try:
    import pytesseract
    from pytesseract import Output
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

import cv2
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class TextBlock:
    """Represents a detected text block with position"""
    text: str
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    block_type: str = "text"  # text, word, line, paragraph
    language: str = "en"
    page_number: int = 0
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "width": self.width,
            "height": self.height,
            "confidence": self.confidence,
            "block_type": self.block_type,
            "language": self.language,
            "page_number": self.page_number
        }


@dataclass
class OCRResult:
    """Result of OCR processing"""
    page_number: int
    image_width: int
    image_height: int
    text_blocks: List[TextBlock] = field(default_factory=list)
    full_text: str = ""
    processing_time_ms: float = 0.0
    engine_used: str = ""
    language: str = "en"
    
    @property
    def word_count(self) -> int:
        return len(self.full_text.split())
    
    @property
    def average_confidence(self) -> float:
        if not self.text_blocks:
            return 0.0
        return sum(b.confidence for b in self.text_blocks) / len(self.text_blocks)
    
    def get_text_in_region(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float
    ) -> List[TextBlock]:
        """Get text blocks within a specific region"""
        results = []
        for block in self.text_blocks:
            # Check if block overlaps with region
            if (block.x1 < x2 and block.x2 > x1 and
                block.y1 < y2 and block.y2 > y1):
                results.append(block)
        return results
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "text_blocks": [b.to_dict() for b in self.text_blocks],
            "full_text": self.full_text,
            "word_count": self.word_count,
            "average_confidence": self.average_confidence,
            "processing_time_ms": self.processing_time_ms,
            "engine_used": self.engine_used,
            "language": self.language
        }


class OCREngine(ABC):
    """Abstract base class for OCR engines"""
    
    @abstractmethod
    def extract_text(
        self,
        image: np.ndarray,
        page_number: int = 0,
        language: str = "en"
    ) -> OCRResult:
        """Extract text from an image"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the OCR engine is available"""
        pass


class TesseractEngine(OCREngine):
    """
    Tesseract OCR Engine.
    
    Primary OCR engine with support for multiple languages
    and detailed bounding box extraction.
    """
    
    def __init__(
        self,
        tesseract_cmd: Optional[str] = None,
        config: str = "--oem 3 --psm 3"
    ):
        """
        Initialize Tesseract engine.
        
        Args:
            tesseract_cmd: Path to tesseract executable
            config: Tesseract configuration string
        """
        self.config = config
        
        if tesseract_cmd and TESSERACT_AVAILABLE:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    
    def is_available(self) -> bool:
        """Check if Tesseract is installed and working"""
        if not TESSERACT_AVAILABLE:
            return False
        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False
    
    def extract_text(
        self,
        image: np.ndarray,
        page_number: int = 0,
        language: str = "eng"
    ) -> OCRResult:
        """
        Extract text using Tesseract.
        
        Args:
            image: Input image as numpy array
            page_number: Page number for multi-page documents
            language: Tesseract language code
            
        Returns:
            OCRResult with extracted text and positions
        """
        import time
        start_time = time.time()
        
        height, width = image.shape[:2]
        
        if not TESSERACT_AVAILABLE:
            logger.error("Tesseract not available")
            return OCRResult(
                page_number=page_number,
                image_width=width,
                image_height=height,
                engine_used="tesseract-unavailable"
            )
        
        # Convert BGR to RGB if needed
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        try:
            # Get detailed data with bounding boxes
            data = pytesseract.image_to_data(
                image_rgb,
                lang=language,
                config=self.config,
                output_type=Output.DICT
            )
            
            text_blocks = []
            full_text_parts = []
            
            n_boxes = len(data['text'])
            for i in range(n_boxes):
                text = data['text'][i]
                conf = int(data['conf'][i])
                
                # Skip empty text or low confidence
                if not text.strip() or conf < 0:
                    continue
                
                x = data['left'][i]
                y = data['top'][i]
                w = data['width'][i]
                h = data['height'][i]
                
                block = TextBlock(
                    text=text,
                    x1=float(x),
                    y1=float(y),
                    x2=float(x + w),
                    y2=float(y + h),
                    confidence=conf / 100.0,
                    block_type="word",
                    language=language,
                    page_number=page_number
                )
                
                text_blocks.append(block)
                full_text_parts.append(text)
            
            # Get full text
            full_text = pytesseract.image_to_string(
                image_rgb,
                lang=language,
                config=self.config
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            return OCRResult(
                page_number=page_number,
                image_width=width,
                image_height=height,
                text_blocks=text_blocks,
                full_text=full_text.strip(),
                processing_time_ms=processing_time,
                engine_used="tesseract",
                language=language
            )
            
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return OCRResult(
                page_number=page_number,
                image_width=width,
                image_height=height,
                processing_time_ms=(time.time() - start_time) * 1000,
                engine_used="tesseract-error"
            )
    
    def extract_from_region(
        self,
        image: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        page_number: int = 0,
        language: str = "eng"
    ) -> OCRResult:
        """Extract text from a specific region of the image"""
        region = image[y1:y2, x1:x2]
        result = self.extract_text(region, page_number, language)
        
        # Adjust coordinates to be relative to full image
        for block in result.text_blocks:
            block.x1 += x1
            block.y1 += y1
            block.x2 += x1
            block.y2 += y1
        
        return result


class EasyOCREngine(OCREngine):
    """
    EasyOCR Engine.
    
    Fallback OCR engine that handles challenging images better
    than Tesseract in some cases.
    """
    
    def __init__(
        self,
        languages: List[str] = None,
        gpu: bool = False
    ):
        """
        Initialize EasyOCR engine.
        
        Args:
            languages: List of language codes
            gpu: Whether to use GPU acceleration
        """
        self.languages = languages or ["en"]
        self.gpu = gpu
        self.reader = None
        
        if EASYOCR_AVAILABLE:
            self._initialize_reader()
    
    def _initialize_reader(self):
        """Initialize the EasyOCR reader"""
        try:
            self.reader = easyocr.Reader(
                self.languages,
                gpu=self.gpu,
                verbose=False
            )
            logger.info(f"EasyOCR initialized with languages: {self.languages}")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            self.reader = None
    
    def is_available(self) -> bool:
        """Check if EasyOCR is available"""
        return EASYOCR_AVAILABLE and self.reader is not None
    
    def extract_text(
        self,
        image: np.ndarray,
        page_number: int = 0,
        language: str = "en"
    ) -> OCRResult:
        """
        Extract text using EasyOCR.
        
        Args:
            image: Input image as numpy array
            page_number: Page number for multi-page documents
            language: Language code (for result metadata)
            
        Returns:
            OCRResult with extracted text and positions
        """
        import time
        start_time = time.time()
        
        height, width = image.shape[:2]
        
        if not self.is_available():
            logger.error("EasyOCR not available")
            return OCRResult(
                page_number=page_number,
                image_width=width,
                image_height=height,
                engine_used="easyocr-unavailable"
            )
        
        try:
            # EasyOCR expects RGB
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # Run OCR
            results = self.reader.readtext(
                image_rgb,
                detail=1,
                paragraph=False
            )
            
            text_blocks = []
            full_text_parts = []
            
            for result in results:
                bbox, text, confidence = result
                
                if not text.strip():
                    continue
                
                # EasyOCR returns polygon, convert to bounding box
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                
                block = TextBlock(
                    text=text,
                    x1=float(min(x_coords)),
                    y1=float(min(y_coords)),
                    x2=float(max(x_coords)),
                    y2=float(max(y_coords)),
                    confidence=float(confidence),
                    block_type="word",
                    language=language,
                    page_number=page_number
                )
                
                text_blocks.append(block)
                full_text_parts.append(text)
            
            processing_time = (time.time() - start_time) * 1000
            
            return OCRResult(
                page_number=page_number,
                image_width=width,
                image_height=height,
                text_blocks=text_blocks,
                full_text=" ".join(full_text_parts),
                processing_time_ms=processing_time,
                engine_used="easyocr",
                language=language
            )
            
        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            return OCRResult(
                page_number=page_number,
                image_width=width,
                image_height=height,
                processing_time_ms=(time.time() - start_time) * 1000,
                engine_used="easyocr-error"
            )


class HybridOCREngine:
    """
    Hybrid OCR engine that uses Tesseract as primary
    and EasyOCR as fallback.
    """
    
    def __init__(
        self,
        tesseract_config: str = "--oem 3 --psm 3",
        easyocr_languages: List[str] = None,
        fallback_threshold: float = 0.3
    ):
        """
        Initialize hybrid engine.
        
        Args:
            tesseract_config: Tesseract configuration
            easyocr_languages: Languages for EasyOCR
            fallback_threshold: Confidence threshold to trigger fallback
        """
        self.tesseract = TesseractEngine(config=tesseract_config)
        self.easyocr = EasyOCREngine(languages=easyocr_languages or ["en"])
        self.fallback_threshold = fallback_threshold
    
    def extract_text(
        self,
        image: np.ndarray,
        page_number: int = 0,
        language: str = "eng"
    ) -> OCRResult:
        """
        Extract text using hybrid approach.
        
        First tries Tesseract, falls back to EasyOCR if confidence is low.
        
        Args:
            image: Input image
            page_number: Page number
            language: Language code
            
        Returns:
            Best OCRResult from available engines
        """
        # Try Tesseract first
        if self.tesseract.is_available():
            result = self.tesseract.extract_text(image, page_number, language)
            
            # Check if result is good enough
            if (result.text_blocks and 
                result.average_confidence >= self.fallback_threshold):
                logger.debug(f"Using Tesseract result (confidence: {result.average_confidence:.2f})")
                return result
            
            logger.debug(f"Tesseract confidence low ({result.average_confidence:.2f}), trying EasyOCR")
        
        # Fallback to EasyOCR
        if self.easyocr.is_available():
            easyocr_result = self.easyocr.extract_text(image, page_number, language)
            
            # Compare results if we have both
            if self.tesseract.is_available():
                # Use the one with better confidence/more text
                tesseract_result = result
                if (easyocr_result.average_confidence > tesseract_result.average_confidence or
                    len(easyocr_result.full_text) > len(tesseract_result.full_text) * 1.2):
                    logger.debug("Using EasyOCR result")
                    return easyocr_result
                else:
                    logger.debug("Keeping Tesseract result")
                    return tesseract_result
            
            return easyocr_result
        
        # Return whatever we have
        if self.tesseract.is_available():
            return result
        
        # No engines available
        height, width = image.shape[:2]
        logger.error("No OCR engines available")
        return OCRResult(
            page_number=page_number,
            image_width=width,
            image_height=height,
            engine_used="none-available"
        )
    
    def extract_batch(
        self,
        images: List[np.ndarray],
        language: str = "eng"
    ) -> List[OCRResult]:
        """Extract text from multiple images"""
        return [
            self.extract_text(image, page_number=i, language=language)
            for i, image in enumerate(images)
        ]
