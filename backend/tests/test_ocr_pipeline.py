"""
OCR Pipeline Tests

Tests for Tesseract, EasyOCR, and Hybrid OCR engines.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Import modules to test
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestTextBlock:
    """Test TextBlock dataclass"""
    
    def test_text_block_creation(self):
        """Test creating a TextBlock"""
        from ocr.ocr_engine import TextBlock
        
        block = TextBlock(
            text="Annual Report",
            x1=50.0, y1=30.0, x2=200.0, y2=60.0,
            confidence=0.95,
            block_type="word",
            language="en",
            page_number=0
        )
        
        assert block.text == "Annual Report"
        assert block.confidence == 0.95
        assert block.width == 150.0
        assert block.height == 30.0
    
    def test_text_block_center(self):
        """Test center calculation"""
        from ocr.ocr_engine import TextBlock
        
        block = TextBlock(
            text="Test",
            x1=0.0, y1=0.0, x2=100.0, y2=100.0,
            confidence=0.9,
            block_type="word"
        )
        
        center = block.center
        assert center == (50.0, 50.0)
    
    def test_text_block_to_dict(self):
        """Test serialization to dict"""
        from ocr.ocr_engine import TextBlock
        
        block = TextBlock(
            text="Revenue",
            x1=50.0, y1=100.0, x2=150.0, y2=130.0,
            confidence=0.88,
            block_type="word",
            language="en",
            page_number=1
        )
        
        data = block.to_dict()
        
        assert data["text"] == "Revenue"
        assert data["confidence"] == 0.88
        assert data["page_number"] == 1
        assert "width" in data
        assert "height" in data


class TestOCRResult:
    """Test OCRResult dataclass"""
    
    def test_empty_result(self):
        """Test empty OCR result"""
        from ocr.ocr_engine import OCRResult
        
        result = OCRResult(
            page_number=0,
            image_width=800,
            image_height=1000,
            engine_used="tesseract"
        )
        
        assert result.word_count == 0
        assert result.average_confidence == 0.0
    
    def test_result_with_blocks(self, mock_ocr_result):
        """Test OCR result with text blocks"""
        assert mock_ocr_result.word_count > 0
        assert mock_ocr_result.average_confidence > 0
        assert len(mock_ocr_result.text_blocks) == 2
    
    def test_get_text_in_region(self, mock_ocr_result):
        """Test finding text in a specific region"""
        # Region that contains the first block
        blocks = mock_ocr_result.get_text_in_region(40, 20, 250, 70)
        assert len(blocks) >= 1
        assert any("Annual" in b.text for b in blocks)
        
        # Region with no text
        empty_blocks = mock_ocr_result.get_text_in_region(500, 500, 600, 600)
        assert len(empty_blocks) == 0


class TestTesseractEngine:
    """Test Tesseract OCR Engine"""
    
    def test_availability_check(self):
        """Test Tesseract availability check"""
        from ocr.ocr_engine import TesseractEngine
        
        engine = TesseractEngine()
        # Result depends on system installation
        availability = engine.is_available()
        assert isinstance(availability, bool)
    
    def test_extract_text_with_image(self, sample_image):
        """Test text extraction from image"""
        from ocr.ocr_engine import TesseractEngine
        
        engine = TesseractEngine()
        if not engine.is_available():
            pytest.skip("Tesseract not installed")
        
        result = engine.extract_text(sample_image, page_number=0)
        
        assert result.page_number == 0
        assert result.image_width == sample_image.shape[1]
        assert result.image_height == sample_image.shape[0]
        assert result.engine_used == "tesseract"
    
    def test_extract_from_region(self, sample_image):
        """Test extraction from specific region"""
        from ocr.ocr_engine import TesseractEngine
        
        engine = TesseractEngine()
        if not engine.is_available():
            pytest.skip("Tesseract not installed")
        
        result = engine.extract_from_region(
            sample_image,
            x1=50, y1=30, x2=750, y2=100,
            page_number=0
        )
        
        # Coordinates should be adjusted
        for block in result.text_blocks:
            assert block.x1 >= 50
            assert block.y1 >= 30


class TestEasyOCREngine:
    """Test EasyOCR Engine"""
    
    def test_initialization(self):
        """Test EasyOCR initialization"""
        from ocr.ocr_engine import EasyOCREngine, EASYOCR_AVAILABLE
        
        if not EASYOCR_AVAILABLE:
            pytest.skip("EasyOCR not installed")
        
        engine = EasyOCREngine(languages=["en"], gpu=False)
        assert engine.languages == ["en"]
    
    def test_availability_check(self):
        """Test EasyOCR availability"""
        from ocr.ocr_engine import EasyOCREngine
        
        engine = EasyOCREngine(gpu=False)
        availability = engine.is_available()
        assert isinstance(availability, bool)


class TestHybridOCREngine:
    """Test Hybrid OCR Engine"""
    
    def test_initialization(self):
        """Test hybrid engine initialization"""
        from ocr.ocr_engine import HybridOCREngine
        
        engine = HybridOCREngine(
            fallback_threshold=0.3
        )
        
        assert engine.fallback_threshold == 0.3
    
    def test_extract_text_prefers_tesseract(self, sample_image):
        """Test that Tesseract is preferred"""
        from ocr.ocr_engine import HybridOCREngine
        
        engine = HybridOCREngine()
        result = engine.extract_text(sample_image, page_number=0)
        
        # Should return some result
        assert result.page_number == 0
        assert result.image_width > 0
    
    def test_batch_extraction(self, sample_image):
        """Test batch text extraction"""
        from ocr.ocr_engine import HybridOCREngine
        
        engine = HybridOCREngine()
        images = [sample_image, sample_image]
        
        results = engine.extract_batch(images)
        
        assert len(results) == 2
        assert results[0].page_number == 0
        assert results[1].page_number == 1


class TestTextProcessor:
    """Test Text Processor"""
    
    def test_normalize_text(self):
        """Test text normalization"""
        from ocr.text_processor import TextProcessor
        
        processor = TextProcessor()
        
        # Test whitespace normalization
        assert processor._normalize_text("  hello   world  ") == "hello world"
        
        # Test ligature handling
        assert processor._normalize_text("ﬁnance") == "finance"
        assert processor._normalize_text("ﬂow") == "flow"
    
    def test_determine_block_type(self):
        """Test block type determination"""
        from ocr.text_processor import TextProcessor
        from ocr.ocr_engine import TextBlock
        
        processor = TextProcessor()
        
        # Short text at top should be heading
        blocks = [TextBlock(
            text="Title",
            x1=50, y1=50, x2=200, y2=80,
            confidence=0.9
        )]
        
        block_type = processor._determine_block_type("Title", blocks)
        assert block_type == "heading"
        
        # List pattern
        list_type = processor._determine_block_type("1. First item", blocks)
        assert list_type == "list"
    
    def test_get_full_text(self):
        """Test full text extraction in reading order"""
        from ocr.text_processor import TextProcessor, ProcessedTextBlock
        from ocr.ocr_engine import TextBlock
        
        processor = TextProcessor()
        
        blocks = [
            ProcessedTextBlock(
                text="Second paragraph",
                x1=50, y1=200, x2=750, y2=300,
                confidence=0.9,
                block_type="paragraph",
                reading_order=1,
                child_blocks=[]
            ),
            ProcessedTextBlock(
                text="First paragraph",
                x1=50, y1=50, x2=750, y2=150,
                confidence=0.9,
                block_type="paragraph",
                reading_order=0,
                child_blocks=[]
            ),
        ]
        
        full_text = processor.get_full_text(blocks)
        
        # Should be in reading order
        assert full_text.index("First") < full_text.index("Second")
