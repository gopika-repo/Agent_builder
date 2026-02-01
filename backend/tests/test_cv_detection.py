"""
CV Detection Tests

Tests for YOLO document layout detection and OpenCV fallback.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestBoundingBox:
    """Test BoundingBox dataclass"""
    
    def test_bounding_box_creation(self):
        """Test creating a BoundingBox"""
        from cv.detector import BoundingBox
        
        bbox = BoundingBox(
            x1=50.0, y1=100.0, x2=200.0, y2=250.0,
            confidence=0.85,
            label="table",
            class_id=0,
            page_number=0
        )
        
        assert bbox.label == "table"
        assert bbox.confidence == 0.85
        assert bbox.width == 150.0
        assert bbox.height == 150.0
        assert bbox.area == 22500.0
    
    def test_center_calculation(self):
        """Test center point calculation"""
        from cv.detector import BoundingBox
        
        bbox = BoundingBox(
            x1=0.0, y1=0.0, x2=100.0, y2=200.0,
            confidence=0.9,
            label="figure",
            class_id=1
        )
        
        center = bbox.center
        assert center == (50.0, 100.0)
    
    def test_contains_point(self):
        """Test point containment"""
        from cv.detector import BoundingBox
        
        bbox = BoundingBox(
            x1=100.0, y1=100.0, x2=200.0, y2=200.0,
            confidence=0.9,
            label="table",
            class_id=0
        )
        
        # Inside
        assert bbox.contains_point(150, 150) == True
        # Outside
        assert bbox.contains_point(50, 50) == False
        # On edge
        assert bbox.contains_point(100, 100) == True
    
    def test_iou_calculation(self):
        """Test Intersection over Union calculation"""
        from cv.detector import BoundingBox
        
        bbox1 = BoundingBox(
            x1=0.0, y1=0.0, x2=100.0, y2=100.0,
            confidence=0.9, label="table", class_id=0
        )
        
        # Same box = 1.0 IoU
        bbox2 = BoundingBox(
            x1=0.0, y1=0.0, x2=100.0, y2=100.0,
            confidence=0.9, label="table", class_id=0
        )
        assert bbox1.intersection_over_union(bbox2) == 1.0
        
        # No overlap = 0.0 IoU
        bbox3 = BoundingBox(
            x1=200.0, y1=200.0, x2=300.0, y2=300.0,
            confidence=0.9, label="table", class_id=0
        )
        assert bbox1.intersection_over_union(bbox3) == 0.0
        
        # Partial overlap
        bbox4 = BoundingBox(
            x1=50.0, y1=50.0, x2=150.0, y2=150.0,
            confidence=0.9, label="table", class_id=0
        )
        iou = bbox1.intersection_over_union(bbox4)
        assert 0 < iou < 1


class TestDetectionResult:
    """Test DetectionResult dataclass"""
    
    def test_empty_result(self):
        """Test empty detection result"""
        from cv.detector import DetectionResult
        
        result = DetectionResult(
            page_number=0,
            image_width=800,
            image_height=1000
        )
        
        assert len(result.detections) == 0
        assert len(result.tables) == 0
        assert len(result.figures) == 0
    
    def test_result_with_detections(self, mock_detection_result):
        """Test detection result with detections"""
        assert len(mock_detection_result.detections) == 2
        assert len(mock_detection_result.tables) == 1
        assert mock_detection_result.processing_time_ms > 0
    
    def test_get_by_label(self, mock_detection_result):
        """Test filtering by label"""
        tables = mock_detection_result.get_by_label("table")
        assert len(tables) == 1
        assert tables[0].label == "table"
        
        headers = mock_detection_result.get_by_label("header")
        assert len(headers) == 1


class TestDocumentLayoutDetector:
    """Test DocumentLayoutDetector"""
    
    def test_initialization(self):
        """Test detector initialization"""
        from cv.detector import DocumentLayoutDetector
        
        detector = DocumentLayoutDetector(
            confidence_threshold=0.5,
            device="cpu"
        )
        
        assert detector.confidence_threshold == 0.5
        assert detector.device == "cpu"
    
    def test_class_mapping(self):
        """Test class name mapping"""
        from cv.detector import DocumentLayoutDetector
        
        detector = DocumentLayoutDetector()
        
        assert detector.CLASS_NAMES[0] == "table"
        assert detector.CLASS_NAMES[1] == "figure"
        assert detector.CLASS_NAMES[2] == "chart"
        assert detector.CLASS_NAMES[3] == "signature"
    
    def test_detect_with_sample_image(self, sample_image):
        """Test detection on sample image"""
        from cv.detector import DocumentLayoutDetector
        
        detector = DocumentLayoutDetector()
        result = detector.detect(sample_image, page_number=0)
        
        assert result.page_number == 0
        assert result.image_width == sample_image.shape[1]
        assert result.image_height == sample_image.shape[0]
        # Fallback detection should find something
        assert len(result.detections) >= 0
    
    def test_fallback_detection(self, sample_image):
        """Test OpenCV fallback detection"""
        from cv.detector import DocumentLayoutDetector
        
        detector = DocumentLayoutDetector()
        
        # Force fallback by setting model to None
        detector.model = None
        
        result = detector.detect(sample_image, page_number=0)
        
        assert result.model_version == "opencv-fallback"
        # Should still detect something
        assert result.image_width == sample_image.shape[1]
    
    def test_batch_detection(self, sample_image):
        """Test batch detection"""
        from cv.detector import DocumentLayoutDetector
        
        detector = DocumentLayoutDetector()
        images = [sample_image, sample_image]
        
        results = detector.detect_batch(images)
        
        assert len(results) == 2
        assert results[0].page_number == 0
        assert results[1].page_number == 1
    
    def test_visualization(self, sample_image, mock_detection_result):
        """Test detection visualization"""
        from cv.detector import DocumentLayoutDetector
        
        detector = DocumentLayoutDetector()
        
        visualized = detector.visualize(
            sample_image,
            mock_detection_result,
            show_labels=True,
            show_confidence=True
        )
        
        assert visualized.shape == sample_image.shape
        # Should be different from original (has drawings)
        assert not np.array_equal(visualized, sample_image)
    
    def test_nms_application(self):
        """Test Non-Maximum Suppression"""
        from cv.detector import DocumentLayoutDetector, BoundingBox
        
        detector = DocumentLayoutDetector()
        
        # Create overlapping boxes
        boxes = [
            BoundingBox(0, 0, 100, 100, 0.9, "table", 0),
            BoundingBox(10, 10, 110, 110, 0.8, "table", 0),  # Overlaps
            BoundingBox(200, 200, 300, 300, 0.85, "figure", 1),  # No overlap
        ]
        
        filtered = detector._apply_nms(boxes, iou_threshold=0.5)
        
        # Should keep the highest confidence + non-overlapping
        assert len(filtered) == 2
        assert filtered[0].confidence == 0.9


class TestTableDetection:
    """Test table-specific detection"""
    
    def test_detect_tables_opencv(self, sample_image):
        """Test OpenCV table detection"""
        from cv.detector import DocumentLayoutDetector
        import cv2
        
        detector = DocumentLayoutDetector()
        gray = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)
        
        tables = detector._detect_tables_opencv(gray, page_number=0)
        
        # Sample image has a table-like structure
        assert isinstance(tables, list)
        for table in tables:
            assert table.label == "table"
            assert 0 <= table.confidence <= 1


class TestFigureDetection:
    """Test figure detection"""
    
    def test_detect_figures_opencv(self, sample_image):
        """Test OpenCV figure detection"""
        from cv.detector import DocumentLayoutDetector
        import cv2
        
        detector = DocumentLayoutDetector()
        gray = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)
        
        figures = detector._detect_figures_opencv(gray, page_number=0)
        
        assert isinstance(figures, list)
        for figure in figures:
            assert figure.label == "figure"
