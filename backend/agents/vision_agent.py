"""
Vision Agent

Detects document layout elements using YOLO and OpenCV:
- Tables
- Figures/Images
- Charts/Graphs
- Signatures
- Headers/Footers
"""

import logging
from typing import Dict, Any, List
import time
import numpy as np

from .state import (
    DocumentState, ProcessingStatus, 
    VisionOutput, Detection
)
from ..cv.detector import DocumentLayoutDetector, BoundingBox
from ..cv.preprocessor import ImagePreprocessor

logger = logging.getLogger(__name__)


class VisionAgent:
    """
    Vision Agent for document layout detection.
    
    Uses YOLO + OpenCV to detect document elements and
    output bounding boxes with confidence scores.
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.5,
        device: str = "cpu"
    ):
        """
        Initialize Vision Agent.
        
        Args:
            confidence_threshold: Minimum detection confidence
            device: Device for inference ('cpu' or 'cuda')
        """
        self.detector = DocumentLayoutDetector(
            confidence_threshold=confidence_threshold,
            device=device
        )
        self.preprocessor = ImagePreprocessor()
        self.confidence_threshold = confidence_threshold
    
    def __call__(self, state: DocumentState) -> DocumentState:
        """
        Process document through vision agent.
        
        Args:
            state: Current document state
            
        Returns:
            Updated state with vision results
        """
        logger.info(f"Vision Agent processing document: {state.get('document_id')}")
        start_time = time.time()
        
        state["status"] = ProcessingStatus.VISION_PROCESSING.value
        state["current_agent"] = "vision"
        
        try:
            # Get page images from state
            pages = state.get("pages", [])
            
            if not pages:
                logger.warning("No pages to process")
                state["errors"].append("No pages found in document")
                state["vision_results"] = VisionOutput().to_dict()
                return state
            
            all_detections = []
            tables = []
            figures = []
            charts = []
            signatures = []
            
            # Process each page
            for page_data in pages:
                page_number = page_data.get("page_number", 0)
                
                # In real implementation, image would be stored separately
                # For now, we'll create a placeholder detection
                # In production, images are passed via file paths or stored in state
                
                logger.debug(f"Processing page {page_number}")
                
                # Simulate detection result for demo
                # In real implementation:
                # image = load_image_from_state(page_data)
                # result = self.detector.detect(image, page_number)
                
                # For now, create mock detections based on page dimensions
                width = page_data.get("width", 2480)  # A4 at 300 DPI
                height = page_data.get("height", 3508)
                
                # Create sample detections for demonstration
                sample_detections = self._create_sample_detections(
                    page_number, width, height
                )
                
                for det in sample_detections:
                    all_detections.append(det)
                    
                    if det.label == "table":
                        tables.append(det)
                    elif det.label == "figure":
                        figures.append(det)
                    elif det.label == "chart":
                        charts.append(det)
                    elif det.label == "signature":
                        signatures.append(det)
            
            processing_time = (time.time() - start_time) * 1000
            
            vision_output = VisionOutput(
                detections=all_detections,
                tables=tables,
                figures=figures,
                charts=charts,
                signatures=signatures,
                processing_time_ms=processing_time
            )
            
            state["vision_results"] = vision_output.to_dict()
            
            logger.info(
                f"Vision Agent completed: {len(all_detections)} detections "
                f"({len(tables)} tables, {len(figures)} figures, "
                f"{len(charts)} charts, {len(signatures)} signatures) "
                f"in {processing_time:.2f}ms"
            )
            
        except Exception as e:
            logger.error(f"Vision Agent error: {e}")
            state["errors"].append(f"Vision processing failed: {str(e)}")
            state["vision_results"] = VisionOutput().to_dict()
        
        return state
    
    def _create_sample_detections(
        self,
        page_number: int,
        width: int,
        height: int
    ) -> List[Detection]:
        """Create sample detections for demonstration"""
        detections = []
        
        # Sample table detection
        detections.append(Detection(
            x1=width * 0.1,
            y1=height * 0.3,
            x2=width * 0.9,
            y2=height * 0.5,
            label="table",
            confidence=0.85,
            page_number=page_number
        ))
        
        # Sample figure detection
        detections.append(Detection(
            x1=width * 0.2,
            y1=height * 0.55,
            x2=width * 0.8,
            y2=height * 0.75,
            label="figure",
            confidence=0.78,
            page_number=page_number
        ))
        
        # Sample header
        detections.append(Detection(
            x1=width * 0.1,
            y1=height * 0.02,
            x2=width * 0.9,
            y2=height * 0.08,
            label="header",
            confidence=0.92,
            page_number=page_number
        ))
        
        return detections
    
    def process_image(
        self,
        image: np.ndarray,
        page_number: int = 0
    ) -> VisionOutput:
        """
        Process a single image.
        
        Args:
            image: Input image as numpy array
            page_number: Page number
            
        Returns:
            VisionOutput with detections
        """
        start_time = time.time()
        
        # Preprocess image
        preprocessed = self.preprocessor.preprocess(image)
        
        # Run detection
        result = self.detector.detect(preprocessed, page_number)
        
        # Convert to Detection objects
        detections = []
        tables = []
        figures = []
        charts = []
        signatures = []
        
        for bbox in result.detections:
            detection = Detection(
                x1=bbox.x1,
                y1=bbox.y1,
                x2=bbox.x2,
                y2=bbox.y2,
                label=bbox.label,
                confidence=bbox.confidence,
                page_number=page_number
            )
            
            detections.append(detection)
            
            if bbox.label == "table":
                tables.append(detection)
            elif bbox.label == "figure":
                figures.append(detection)
            elif bbox.label == "chart":
                charts.append(detection)
            elif bbox.label == "signature":
                signatures.append(detection)
        
        processing_time = (time.time() - start_time) * 1000
        
        return VisionOutput(
            detections=detections,
            tables=tables,
            figures=figures,
            charts=charts,
            signatures=signatures,
            processing_time_ms=processing_time
        )


def vision_agent_node(state: DocumentState) -> DocumentState:
    """LangGraph node function for Vision Agent"""
    agent = VisionAgent()
    return agent(state)
