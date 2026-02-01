"""
Visual Confidence Heatmap Generator

Generates visual overlays showing:
- Per-region confidence scores
- Color-coded heatmaps (green=high, red=low)
- Interactive region highlighting
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import io
import base64

logger = logging.getLogger(__name__)

try:
    import numpy as np
    import cv2
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False


@dataclass
class ConfidenceRegion:
    """A region with confidence information"""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    label: str
    source: str  # ocr, vision, fusion
    page_number: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "confidence": round(self.confidence, 3),
            "label": self.label,
            "source": self.source,
            "page_number": self.page_number
        }


@dataclass
class HeatmapResult:
    """Result of heatmap generation"""
    page_number: int
    width: int
    height: int
    regions: List[ConfidenceRegion]
    image_base64: Optional[str] = None  # Base64 encoded heatmap overlay
    average_confidence: float = 0.0
    low_confidence_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "width": self.width,
            "height": self.height,
            "regions": [r.to_dict() for r in self.regions],
            "image_base64": self.image_base64,
            "average_confidence": round(self.average_confidence, 3),
            "low_confidence_count": self.low_confidence_count
        }


class ConfidenceHeatmapGenerator:
    """
    Generates visual confidence heatmaps for documents.
    
    Features:
    - Color-coded overlays (green=high, red=low)
    - Per-region confidence visualization
    - Interactive highlighting data for frontend
    - Aggregate statistics
    """
    
    def __init__(
        self,
        low_threshold: float = 0.6,
        high_threshold: float = 0.8,
        alpha: float = 0.4
    ):
        """
        Initialize heatmap generator.
        
        Args:
            low_threshold: Below this is red (low confidence)
            high_threshold: Above this is green (high confidence)
            alpha: Transparency of overlay (0-1)
        """
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.alpha = alpha
    
    def generate(
        self,
        page_image: 'np.ndarray',
        regions: List[ConfidenceRegion],
        page_number: int = 0
    ) -> HeatmapResult:
        """
        Generate confidence heatmap for a page.
        
        Args:
            page_image: Original page image (numpy array)
            regions: List of confidence regions
            page_number: Page number
            
        Returns:
            HeatmapResult with visualization data
        """
        if not CV_AVAILABLE:
            return self._generate_without_cv(regions, page_number)
        
        height, width = page_image.shape[:2]
        
        # Create overlay
        overlay = page_image.copy()
        
        low_count = 0
        total_confidence = 0.0
        
        for region in regions:
            # Get color based on confidence
            color = self._confidence_to_color(region.confidence)
            
            # Draw filled rectangle
            x1, y1 = int(region.x1), int(region.y1)
            x2, y2 = int(region.x2), int(region.y2)
            
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            
            # Track statistics
            total_confidence += region.confidence
            if region.confidence < self.low_threshold:
                low_count += 1
        
        # Blend overlay with original
        result = cv2.addWeighted(overlay, self.alpha, page_image, 1 - self.alpha, 0)
        
        # Draw borders and labels
        for region in regions:
            x1, y1 = int(region.x1), int(region.y1)
            x2, y2 = int(region.x2), int(region.y2)
            color = self._confidence_to_color(region.confidence)
            
            # Draw border
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            
            # Draw confidence label
            label = f"{region.label}: {region.confidence:.0%}"
            cv2.putText(
                result, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )
        
        # Encode to base64
        _, buffer = cv2.imencode('.png', result)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        avg_confidence = total_confidence / len(regions) if regions else 0.0
        
        return HeatmapResult(
            page_number=page_number,
            width=width,
            height=height,
            regions=regions,
            image_base64=image_base64,
            average_confidence=avg_confidence,
            low_confidence_count=low_count
        )
    
    def generate_from_results(
        self,
        page_image: 'np.ndarray',
        vision_results: Dict[str, Any],
        ocr_results: Dict[str, Any],
        validation_results: Dict[str, Any],
        page_number: int = 0
    ) -> HeatmapResult:
        """
        Generate heatmap from pipeline results.
        
        Args:
            page_image: Original page image
            vision_results: Vision agent output
            ocr_results: OCR agent output
            validation_results: Validation agent output
            page_number: Page number
            
        Returns:
            HeatmapResult with combined confidence data
        """
        regions = []
        
        # Extract vision detections
        for detection in vision_results.get("detections", []):
            if detection.get("page_number", 0) == page_number:
                regions.append(ConfidenceRegion(
                    x1=detection.get("x1", 0),
                    y1=detection.get("y1", 0),
                    x2=detection.get("x2", 0),
                    y2=detection.get("y2", 0),
                    confidence=detection.get("confidence", 0),
                    label=detection.get("label", "unknown"),
                    source="vision",
                    page_number=page_number
                ))
        
        # Extract OCR blocks
        for block in ocr_results.get("text_blocks", []):
            if block.get("page_number", 0) == page_number:
                regions.append(ConfidenceRegion(
                    x1=block.get("x1", 0),
                    y1=block.get("y1", 0),
                    x2=block.get("x2", 0),
                    y2=block.get("y2", 0),
                    confidence=block.get("confidence", 0),
                    label="text",
                    source="ocr",
                    page_number=page_number
                ))
        
        # Add validation scores
        for field in validation_results.get("field_scores", []):
            if field.get("page_number", 0) == page_number and "bbox" in field:
                bbox = field["bbox"]
                regions.append(ConfidenceRegion(
                    x1=bbox[0],
                    y1=bbox[1],
                    x2=bbox[2],
                    y2=bbox[3],
                    confidence=field.get("score", 0),
                    label=field.get("field_name", "field"),
                    source="validation",
                    page_number=page_number
                ))
        
        return self.generate(page_image, regions, page_number)
    
    def _confidence_to_color(self, confidence: float) -> Tuple[int, int, int]:
        """
        Convert confidence to BGR color.
        
        Returns:
            BGR color tuple (for OpenCV)
        """
        if confidence >= self.high_threshold:
            # Green
            return (0, 200, 0)
        elif confidence >= self.low_threshold:
            # Yellow to Orange (interpolate)
            ratio = (confidence - self.low_threshold) / (self.high_threshold - self.low_threshold)
            green = int(200 * ratio)
            return (0, green, 200)
        else:
            # Red
            return (0, 0, 200)
    
    def _generate_without_cv(
        self,
        regions: List[ConfidenceRegion],
        page_number: int
    ) -> HeatmapResult:
        """Generate result without OpenCV (metadata only)"""
        low_count = sum(1 for r in regions if r.confidence < self.low_threshold)
        avg_confidence = sum(r.confidence for r in regions) / len(regions) if regions else 0.0
        
        return HeatmapResult(
            page_number=page_number,
            width=0,
            height=0,
            regions=regions,
            image_base64=None,
            average_confidence=avg_confidence,
            low_confidence_count=low_count
        )
    
    def get_color_legend(self) -> Dict[str, Any]:
        """Get the color legend for the heatmap"""
        return {
            "low": {
                "threshold": f"< {self.low_threshold:.0%}",
                "color": "#FF0000",
                "meaning": "Low confidence - requires review"
            },
            "medium": {
                "threshold": f"{self.low_threshold:.0%} - {self.high_threshold:.0%}",
                "color": "#FFA500",
                "meaning": "Medium confidence - may need verification"
            },
            "high": {
                "threshold": f"> {self.high_threshold:.0%}",
                "color": "#00FF00",
                "meaning": "High confidence - auto-accepted"
            }
        }


def generate_confidence_css_overlay(regions: List[ConfidenceRegion]) -> str:
    """
    Generate CSS for frontend confidence overlay.
    
    This creates styled div elements that can be positioned
    over the document image in the frontend.
    """
    css_parts = []
    
    for i, region in enumerate(regions):
        # Determine color
        if region.confidence >= 0.8:
            color = "rgba(0, 200, 0, 0.3)"
            border = "rgb(0, 200, 0)"
        elif region.confidence >= 0.6:
            color = "rgba(255, 165, 0, 0.3)"
            border = "rgb(255, 165, 0)"
        else:
            color = "rgba(255, 0, 0, 0.3)"
            border = "rgb(255, 0, 0)"
        
        css_parts.append(f"""
.confidence-region-{i} {{
    position: absolute;
    left: {region.x1}px;
    top: {region.y1}px;
    width: {region.x2 - region.x1}px;
    height: {region.y2 - region.y1}px;
    background-color: {color};
    border: 2px solid {border};
    pointer-events: auto;
    cursor: pointer;
}}
.confidence-region-{i}:hover::after {{
    content: '{region.label}: {region.confidence:.0%}';
    position: absolute;
    top: -20px;
    left: 0;
    background: black;
    color: white;
    padding: 2px 5px;
    font-size: 12px;
    border-radius: 3px;
}}
""")
    
    return "\n".join(css_parts)


def generate_confidence_react_data(regions: List[ConfidenceRegion]) -> List[Dict[str, Any]]:
    """
    Generate data structure for React frontend rendering.
    
    Returns:
        List of region data for frontend components
    """
    return [
        {
            "id": f"region-{i}",
            "x": region.x1,
            "y": region.y1,
            "width": region.x2 - region.x1,
            "height": region.y2 - region.y1,
            "confidence": region.confidence,
            "label": region.label,
            "source": region.source,
            "page": region.page_number,
            "color": (
                "#00C800" if region.confidence >= 0.8
                else "#FFA500" if region.confidence >= 0.6
                else "#FF0000"
            ),
            "opacity": 0.3,
            "requiresReview": region.confidence < 0.6
        }
        for i, region in enumerate(regions)
    ]
