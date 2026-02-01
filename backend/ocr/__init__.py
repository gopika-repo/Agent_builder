"""
OCR Module
Text extraction using Tesseract and EasyOCR
"""

from .ocr_engine import OCREngine, TesseractEngine, EasyOCREngine, OCRResult, TextBlock
from .text_processor import TextProcessor

__all__ = [
    "OCREngine",
    "TesseractEngine", 
    "EasyOCREngine",
    "OCRResult",
    "TextBlock",
    "TextProcessor"
]
