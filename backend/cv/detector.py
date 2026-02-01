"""
Document Layout Detector using YOLO

This module provides document layout detection capabilities using YOLO
to identify tables, figures, charts, signatures, and other document elements.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    
import cv2

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Represents a detected region with bounding box"""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    label: str
    class_id: int
    page_number: int = 0
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "width": self.width,
            "height": self.height,
            "confidence": self.confidence,
            "label": self.label,
            "class_id": self.class_id,
            "page_number": self.page_number
        }
    
    def contains_point(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2
    
    def intersection_over_union(self, other: 'BoundingBox') -> float:
        """Calculate IoU with another bounding box"""
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        union = self.area + other.area - intersection
        
        return intersection / union if union > 0 else 0.0


@dataclass
class DetectionResult:
    """Result of document layout detection"""
    page_number: int
    image_width: int
    image_height: int
    detections: List[BoundingBox] = field(default_factory=list)
    processing_time_ms: float = 0.0
    model_version: str = ""
    
    def get_by_label(self, label: str) -> List[BoundingBox]:
        """Get all detections of a specific type"""
        return [d for d in self.detections if d.label == label]
    
    @property
    def tables(self) -> List[BoundingBox]:
        return self.get_by_label("table")
    
    @property
    def figures(self) -> List[BoundingBox]:
        return self.get_by_label("figure")
    
    @property
    def charts(self) -> List[BoundingBox]:
        return self.get_by_label("chart")
    
    @property
    def signatures(self) -> List[BoundingBox]:
        return self.get_by_label("signature")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "detections": [d.to_dict() for d in self.detections],
            "processing_time_ms": self.processing_time_ms,
            "model_version": self.model_version,
            "summary": {
                "total_detections": len(self.detections),
                "tables": len(self.tables),
                "figures": len(self.figures),
                "charts": len(self.charts),
                "signatures": len(self.signatures)
            }
        }


class DocumentLayoutDetector:
    """
    YOLO-based document layout detection.
    
    Detects:
    - Tables
    - Figures/Images
    - Charts/Graphs
    - Signatures
    - Headers/Footers
    - Text blocks
    """
    
    # Document layout class mapping
    CLASS_NAMES = {
        0: "table",
        1: "figure",
        2: "chart",
        3: "signature",
        4: "header",
        5: "footer",
        6: "paragraph",
        7: "list",
        8: "title",
        9: "caption"
    }
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "cpu"
    ):
        """
        Initialize the document layout detector.
        
        Args:
            model_path: Path to YOLO model weights. If None, uses pre-trained.
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model = None
        self.model_path = model_path
        
        if YOLO_AVAILABLE:
            self._load_model(model_path)
        else:
            logger.warning("YOLO not available. Using fallback detection.")
    
    def _load_model(self, model_path: Optional[str] = None):
        """Load YOLO model"""
        try:
            if model_path and Path(model_path).exists():
                self.model = YOLO(model_path)
                logger.info(f"Loaded custom YOLO model from {model_path}")
            else:
                # Use pre-trained YOLOv8 nano as base
                # In production, this would be a document-specific model
                self.model = YOLO("yolov8n.pt")
                logger.info("Loaded pre-trained YOLOv8n model")
            
            self.model.to(self.device)
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.model = None
    
    def detect(
        self,
        image: np.ndarray,
        page_number: int = 0
    ) -> DetectionResult:
        """
        Detect document layout elements in an image.
        
        Args:
            image: Input image as numpy array (BGR or RGB)
            page_number: Page number for multi-page documents
            
        Returns:
            DetectionResult with all detected elements
        """
        import time
        start_time = time.time()
        
        height, width = image.shape[:2]
        detections = []
        
        if self.model is not None and YOLO_AVAILABLE:
            detections = self._yolo_detect(image, page_number)
        else:
            # Fallback to OpenCV-based detection
            detections = self._fallback_detect(image, page_number)
        
        processing_time = (time.time() - start_time) * 1000
        
        return DetectionResult(
            page_number=page_number,
            image_width=width,
            image_height=height,
            detections=detections,
            processing_time_ms=processing_time,
            model_version="yolov8n" if self.model else "opencv-fallback"
        )
    
    def _yolo_detect(
        self,
        image: np.ndarray,
        page_number: int
    ) -> List[BoundingBox]:
        """Run YOLO detection"""
        detections = []
        
        try:
            results = self.model(
                image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i in range(len(boxes)):
                        box = boxes[i]
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Map to document element classes
                        label = self._map_class_to_document_element(class_id)
                        
                        detections.append(BoundingBox(
                            x1=float(x1),
                            y1=float(y1),
                            x2=float(x2),
                            y2=float(y2),
                            confidence=confidence,
                            label=label,
                            class_id=class_id,
                            page_number=page_number
                        ))
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            detections = self._fallback_detect(image, page_number)
        
        return detections
    
    def _map_class_to_document_element(self, class_id: int) -> str:
        """Map YOLO class ID to document element type"""
        # For pre-trained YOLO, we map common objects to document elements
        # In production, use a document-specific model with proper classes
        yolo_to_doc = {
            # COCO classes that might appear in documents
            0: "figure",    # person in photo
            56: "table",    # chair (placeholder)
            57: "table",    # couch (placeholder)
            58: "figure",   # potted plant
            59: "figure",   # bed
            60: "table",    # dining table
            62: "figure",   # tv
            63: "figure",   # laptop
            64: "figure",   # mouse
            72: "figure",   # refrigerator
            73: "figure",   # book -> could be figure
        }
        
        if class_id in self.CLASS_NAMES:
            return self.CLASS_NAMES[class_id]
        
        return yolo_to_doc.get(class_id, "figure")
    
    def _fallback_detect(
        self,
        image: np.ndarray,
        page_number: int
    ) -> List[BoundingBox]:
        """
        Fallback detection using OpenCV when YOLO is not available.
        Uses contour detection and heuristics.
        """
        detections = []
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        height, width = gray.shape
        
        # Detect tables using line detection
        table_boxes = self._detect_tables_opencv(gray, page_number)
        detections.extend(table_boxes)
        
        # Detect figures using contour analysis
        figure_boxes = self._detect_figures_opencv(gray, page_number)
        detections.extend(figure_boxes)
        
        # Detect text regions
        text_boxes = self._detect_text_regions_opencv(gray, page_number)
        detections.extend(text_boxes)
        
        # Apply NMS to remove overlapping detections
        detections = self._apply_nms(detections)
        
        return detections
    
    def _detect_tables_opencv(
        self,
        gray: np.ndarray,
        page_number: int
    ) -> List[BoundingBox]:
        """Detect tables using horizontal and vertical line detection"""
        tables = []
        height, width = gray.shape
        
        # Detect lines using morphological operations
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        
        # Binary threshold
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Detect horizontal lines
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Detect vertical lines
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
        
        # Combine
        table_mask = cv2.add(horizontal, vertical)
        
        # Find contours of potential tables
        contours, _ = cv2.findContours(
            table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size - tables should be reasonably large
            if w > width * 0.1 and h > height * 0.05:
                # Check if it has grid structure
                roi = table_mask[y:y+h, x:x+w]
                line_density = np.sum(roi > 0) / (w * h)
                
                if line_density > 0.01:  # Has sufficient line structure
                    tables.append(BoundingBox(
                        x1=float(x),
                        y1=float(y),
                        x2=float(x + w),
                        y2=float(y + h),
                        confidence=min(0.9, line_density * 10),
                        label="table",
                        class_id=0,
                        page_number=page_number
                    ))
        
        return tables
    
    def _detect_figures_opencv(
        self,
        gray: np.ndarray,
        page_number: int
    ) -> List[BoundingBox]:
        """Detect figures/images using contour analysis"""
        figures = []
        height, width = gray.shape
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate to connect nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # Figures should be reasonably sized
            if (w > width * 0.1 and h > height * 0.1 and 
                area > (width * height) * 0.02):
                
                # Check aspect ratio - figures usually aren't too extreme
                aspect_ratio = w / h if h > 0 else 0
                if 0.2 < aspect_ratio < 5:
                    # Check if region has image-like properties
                    roi = gray[y:y+h, x:x+w]
                    std_dev = np.std(roi)
                    
                    if std_dev > 30:  # Has variation, likely an image
                        figures.append(BoundingBox(
                            x1=float(x),
                            y1=float(y),
                            x2=float(x + w),
                            y2=float(y + h),
                            confidence=min(0.8, std_dev / 100),
                            label="figure",
                            class_id=1,
                            page_number=page_number
                        ))
        
        return figures
    
    def _detect_text_regions_opencv(
        self,
        gray: np.ndarray,
        page_number: int
    ) -> List[BoundingBox]:
        """Detect text regions using MSER or morphological operations"""
        text_regions = []
        height, width = gray.shape
        
        # Binary threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological operations to connect text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        dilated = cv2.dilate(binary, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Text blocks have specific characteristics
            if w > width * 0.05 and h > 10 and h < height * 0.3:
                aspect_ratio = w / h if h > 0 else 0
                
                # Text blocks are typically wider than tall
                if aspect_ratio > 1:
                    # Check if it's at the top (header) or bottom (footer)
                    if y < height * 0.1:
                        label = "header"
                    elif y > height * 0.9:
                        label = "footer"
                    else:
                        label = "paragraph"
                    
                    text_regions.append(BoundingBox(
                        x1=float(x),
                        y1=float(y),
                        x2=float(x + w),
                        y2=float(y + h),
                        confidence=0.7,
                        label=label,
                        class_id=6 if label == "paragraph" else (4 if label == "header" else 5),
                        page_number=page_number
                    ))
        
        return text_regions
    
    def _apply_nms(
        self,
        detections: List[BoundingBox],
        iou_threshold: float = 0.5
    ) -> List[BoundingBox]:
        """Apply Non-Maximum Suppression to remove overlapping detections"""
        if not detections:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
        
        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)
            
            # Remove overlapping boxes
            detections = [
                d for d in detections
                if best.intersection_over_union(d) < iou_threshold
            ]
        
        return keep
    
    def detect_batch(
        self,
        images: List[np.ndarray]
    ) -> List[DetectionResult]:
        """
        Detect layout in multiple images.
        
        Args:
            images: List of images as numpy arrays
            
        Returns:
            List of DetectionResults, one per image
        """
        return [
            self.detect(image, page_number=i)
            for i, image in enumerate(images)
        ]
    
    def visualize(
        self,
        image: np.ndarray,
        result: DetectionResult,
        show_labels: bool = True,
        show_confidence: bool = True
    ) -> np.ndarray:
        """
        Draw detection results on an image.
        
        Args:
            image: Original image
            result: Detection result to visualize
            show_labels: Whether to show class labels
            show_confidence: Whether to show confidence scores
            
        Returns:
            Image with detections drawn
        """
        output = image.copy()
        
        # Color map for different classes
        colors = {
            "table": (0, 255, 0),       # Green
            "figure": (255, 0, 0),      # Blue
            "chart": (0, 255, 255),     # Yellow
            "signature": (255, 0, 255), # Magenta
            "header": (128, 128, 0),    # Olive
            "footer": (0, 128, 128),    # Teal
            "paragraph": (128, 0, 128), # Purple
            "list": (0, 128, 0),        # Dark Green
            "title": (255, 128, 0),     # Orange
            "caption": (128, 255, 0)    # Lime
        }
        
        for detection in result.detections:
            color = colors.get(detection.label, (200, 200, 200))
            
            # Draw rectangle
            pt1 = (int(detection.x1), int(detection.y1))
            pt2 = (int(detection.x2), int(detection.y2))
            cv2.rectangle(output, pt1, pt2, color, 2)
            
            # Draw label
            if show_labels:
                label_text = detection.label
                if show_confidence:
                    label_text += f" {detection.confidence:.2f}"
                
                (text_width, text_height), _ = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                
                cv2.rectangle(
                    output,
                    (pt1[0], pt1[1] - text_height - 5),
                    (pt1[0] + text_width, pt1[1]),
                    color,
                    -1
                )
                
                cv2.putText(
                    output,
                    label_text,
                    (pt1[0], pt1[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
        
        return output
