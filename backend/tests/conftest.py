"""
Pytest Configuration and Fixtures

Provides shared fixtures for testing document processing pipeline.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import os

# Add backend to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_image():
    """Generate a sample document-like image for testing"""
    # Create a simple grayscale image with text-like patterns
    height, width = 1000, 800
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Add some black rectangles to simulate text blocks
    import cv2
    
    # Header area
    cv2.rectangle(image, (50, 30), (750, 80), (0, 0, 0), 2)
    
    # Paragraph blocks
    for y in range(150, 600, 100):
        cv2.rectangle(image, (50, y), (750, y + 60), (200, 200, 200), -1)
        
    # Table-like structure
    cv2.rectangle(image, (50, 650), (750, 850), (0, 0, 0), 2)
    for y in range(650, 850, 50):
        cv2.line(image, (50, y), (750, y), (0, 0, 0), 1)
    for x in range(50, 750, 175):
        cv2.line(image, (x, 650), (x, 850), (0, 0, 0), 1)
    
    return image


@pytest.fixture
def sample_text():
    """Sample document text for testing"""
    return """
    ANNUAL FINANCIAL REPORT 2024
    
    Executive Summary
    
    This document presents the annual financial results for fiscal year 2024.
    Revenue increased by 15% to $500 million. Net profit margin improved to 12%.
    
    Key Highlights:
    - Total Revenue: $500M
    - Operating Income: $75M
    - Net Profit: $60M
    - Employee Count: 2,500
    
    Financial Performance Table:
    
    | Quarter | Revenue | Expenses | Profit |
    |---------|---------|----------|--------|
    | Q1      | $120M   | $100M    | $20M   |
    | Q2      | $125M   | $105M    | $20M   |
    | Q3      | $130M   | $110M    | $20M   |
    | Q4      | $125M   | $105M    | $20M   |
    """


@pytest.fixture
def sample_table_data():
    """Sample table structure for testing"""
    return {
        "id": "table_1",
        "page_number": 1,
        "headers": ["Quarter", "Revenue", "Expenses", "Profit"],
        "rows": [
            ["Q1", "$120M", "$100M", "$20M"],
            ["Q2", "$125M", "$105M", "$20M"],
            ["Q3", "$130M", "$110M", "$20M"],
            ["Q4", "$125M", "$105M", "$20M"],
        ],
        "num_rows": 4,
        "num_cols": 4
    }


@pytest.fixture
def sample_document_state():
    """Sample document state for testing agents"""
    return {
        "document_id": "test-doc-001",
        "filename": "test_document.pdf",
        "file_type": "pdf",
        "status": "pending",
        "current_agent": None,
        "pages": [
            {
                "page_number": 0,
                "width": 800,
                "height": 1000
            }
        ],
        "vision_results": {},
        "ocr_results": {},
        "layout_results": {},
        "reasoning_results": {},
        "fused_output": {},
        "validation_results": {},
        "errors": [],
        "config": {}
    }


@pytest.fixture
def temp_upload_dir():
    """Create temporary upload directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_ocr_result():
    """Mock OCR result for testing"""
    from ocr.ocr_engine import OCRResult, TextBlock
    
    return OCRResult(
        page_number=0,
        image_width=800,
        image_height=1000,
        text_blocks=[
            TextBlock(
                text="Annual Report",
                x1=50.0, y1=30.0, x2=200.0, y2=60.0,
                confidence=0.95,
                block_type="word",
                language="en",
                page_number=0
            ),
            TextBlock(
                text="Revenue",
                x1=50.0, y1=150.0, x2=150.0, y2=180.0,
                confidence=0.90,
                block_type="word",
                language="en",
                page_number=0
            ),
        ],
        full_text="Annual Report\nRevenue increased by 15%",
        processing_time_ms=150.0,
        engine_used="tesseract",
        language="en"
    )


@pytest.fixture
def mock_detection_result():
    """Mock CV detection result for testing"""
    from cv.detector import DetectionResult, BoundingBox
    
    return DetectionResult(
        page_number=0,
        image_width=800,
        image_height=1000,
        detections=[
            BoundingBox(
                x1=50.0, y1=30.0, x2=750.0, y2=80.0,
                confidence=0.92,
                label="header",
                class_id=4,
                page_number=0
            ),
            BoundingBox(
                x1=50.0, y1=650.0, x2=750.0, y2=850.0,
                confidence=0.88,
                label="table",
                class_id=0,
                page_number=0
            ),
        ],
        processing_time_ms=50.0,
        model_version="opencv-fallback"
    )
