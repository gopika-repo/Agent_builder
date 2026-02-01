"""
OCR Agent

Extracts text from documents using Tesseract (primary) and EasyOCR (fallback).
Provides coordinate-aware text extraction with confidence scores.
"""

import logging
from typing import Dict, Any, List
import time
import numpy as np

from .state import (
    DocumentState, ProcessingStatus,
    OCROutput, TextBlockData
)
from ..ocr.ocr_engine import HybridOCREngine, OCRResult

logger = logging.getLogger(__name__)


class OCRAgent:
    """
    OCR Agent for text extraction.
    
    Uses Tesseract as primary engine with EasyOCR fallback.
    Produces structured text blocks with position coordinates.
    """
    
    def __init__(
        self,
        fallback_threshold: float = 0.3,
        languages: List[str] = None
    ):
        """
        Initialize OCR Agent.
        
        Args:
            fallback_threshold: Confidence threshold to trigger EasyOCR
            languages: Languages to use for OCR
        """
        self.languages = languages or ["en"]
        self.ocr_engine = HybridOCREngine(
            fallback_threshold=fallback_threshold,
            easyocr_languages=self.languages
        )
    
    def __call__(self, state: DocumentState) -> DocumentState:
        """
        Process document through OCR agent.
        
        Args:
            state: Current document state
            
        Returns:
            Updated state with OCR results
        """
        logger.info(f"OCR Agent processing document: {state.get('document_id')}")
        start_time = time.time()
        
        state["status"] = ProcessingStatus.OCR_PROCESSING.value
        state["current_agent"] = "ocr"
        
        try:
            pages = state.get("pages", [])
            vision_results = state.get("vision_results", {})
            
            all_text_blocks = []
            page_texts = {}
            full_text_parts = []
            engine_used = "hybrid"
            
            # Process each page
            for page_data in pages:
                page_number = page_data.get("page_number", 0)
                
                logger.debug(f"OCR processing page {page_number}")
                
                # In real implementation:
                # image = load_image_from_state(page_data)
                # result = self.ocr_engine.extract_text(image, page_number)
                
                # For demo, create sample OCR output
                width = page_data.get("width", 2480)
                height = page_data.get("height", 3508)
                
                sample_result = self._create_sample_ocr(page_number, width, height)
                
                for block in sample_result.text_blocks:
                    text_block = TextBlockData(
                        text=block.text,
                        x1=block.x1,
                        y1=block.y1,
                        x2=block.x2,
                        y2=block.y2,
                        confidence=block.confidence,
                        page_number=page_number
                    )
                    all_text_blocks.append(text_block)
                
                page_texts[page_number] = sample_result.full_text
                full_text_parts.append(sample_result.full_text)
                engine_used = sample_result.engine_used
            
            # Also extract text from detected regions (tables, figures)
            table_detections = vision_results.get("tables", [])
            for table in table_detections:
                # In real implementation, would OCR the table region specifically
                pass
            
            processing_time = (time.time() - start_time) * 1000
            
            ocr_output = OCROutput(
                text_blocks=all_text_blocks,
                full_text="\n\n".join(full_text_parts),
                page_texts=page_texts,
                processing_time_ms=processing_time,
                engine_used=engine_used
            )
            
            state["ocr_results"] = ocr_output.to_dict()
            
            logger.info(
                f"OCR Agent completed: {len(all_text_blocks)} text blocks, "
                f"{len(ocr_output.full_text.split())} words in {processing_time:.2f}ms"
            )
            
        except Exception as e:
            logger.error(f"OCR Agent error: {e}")
            state["errors"].append(f"OCR processing failed: {str(e)}")
            state["ocr_results"] = OCROutput().to_dict()
        
        return state
    
    def _create_sample_ocr(
        self,
        page_number: int,
        width: int,
        height: int
    ) -> OCRResult:
        """Create sample OCR output for demonstration"""
        from ..ocr.ocr_engine import OCRResult, TextBlock
        
        sample_text = """
        QUARTERLY FINANCIAL REPORT
        Q4 2025
        
        Executive Summary
        
        This quarterly report presents the financial performance and key metrics
        for the fourth quarter of 2025. Our company has demonstrated strong growth
        across all major business segments, with revenue increasing by 15% year-over-year.
        
        Key Highlights:
        • Total Revenue: $2.5 billion
        • Net Income: $450 million
        • Operating Margin: 18%
        • Customer Base: 50 million active users
        
        Financial Performance
        
        Revenue for Q4 2025 reached $2.5 billion, representing a 15% increase
        compared to the same period last year. This growth was driven by expansion
        in our enterprise segment and improved customer retention rates.
        
        The company's operating expenses remained well-controlled, with total
        operating costs of $2.05 billion. This resulted in an operating margin
        of 18%, up from 16% in the previous quarter.
        """
        
        # Create text blocks simulating word-level OCR output
        text_blocks = []
        words = sample_text.split()
        
        # Simulate word positions
        x = width * 0.1
        y = height * 0.05
        line_height = 30
        word_spacing = 15
        
        for i, word in enumerate(words):
            word_width = len(word) * 12  # Approximate
            
            if x + word_width > width * 0.9:
                x = width * 0.1
                y += line_height
            
            block = TextBlock(
                text=word,
                x1=x,
                y1=y,
                x2=x + word_width,
                y2=y + line_height - 5,
                confidence=0.85 + (i % 10) * 0.01,
                block_type="word",
                language="en",
                page_number=page_number
            )
            text_blocks.append(block)
            
            x += word_width + word_spacing
        
        return OCRResult(
            page_number=page_number,
            image_width=width,
            image_height=height,
            text_blocks=text_blocks,
            full_text=sample_text.strip(),
            processing_time_ms=150.0,
            engine_used="tesseract",
            language="en"
        )
    
    def process_image(
        self,
        image: np.ndarray,
        page_number: int = 0,
        language: str = "eng"
    ) -> OCROutput:
        """
        Process a single image.
        
        Args:
            image: Input image as numpy array
            page_number: Page number
            language: OCR language
            
        Returns:
            OCROutput with extracted text
        """
        result = self.ocr_engine.extract_text(image, page_number, language)
        
        text_blocks = [
            TextBlockData(
                text=b.text,
                x1=b.x1,
                y1=b.y1,
                x2=b.x2,
                y2=b.y2,
                confidence=b.confidence,
                page_number=page_number
            )
            for b in result.text_blocks
        ]
        
        return OCROutput(
            text_blocks=text_blocks,
            full_text=result.full_text,
            page_texts={page_number: result.full_text},
            processing_time_ms=result.processing_time_ms,
            engine_used=result.engine_used
        )


def ocr_agent_node(state: DocumentState) -> DocumentState:
    """LangGraph node function for OCR Agent"""
    agent = OCRAgent()
    return agent(state)
