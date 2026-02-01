"""
Fusion Agent

Combines outputs from Vision, OCR, and Layout agents into a unified representation.
Resolves conflicts between modalities and produces structured JSON output.
"""

import logging
from typing import Dict, Any, List, Optional
import time
import uuid

from .state import (
    DocumentState, ProcessingStatus,
    FusedDocument, FusedElement, TableData
)

logger = logging.getLogger(__name__)


class FusionAgent:
    """
    Fusion Agent for combining multi-modal outputs.
    
    Integrates:
    - Vision detection results
    - OCR text extraction
    - Layout structure analysis
    
    Produces unified structured representation with conflict resolution.
    """
    
    def __init__(
        self,
        vision_weight: float = 0.3,
        ocr_weight: float = 0.4,
        layout_weight: float = 0.3
    ):
        """
        Initialize Fusion Agent.
        
        Args:
            vision_weight: Weight for vision confidence
            ocr_weight: Weight for OCR confidence
            layout_weight: Weight for layout confidence
        """
        self.vision_weight = vision_weight
        self.ocr_weight = ocr_weight
        self.layout_weight = layout_weight
    
    def __call__(self, state: DocumentState) -> DocumentState:
        """
        Process document through fusion agent.
        
        Args:
            state: Current document state
            
        Returns:
            Updated state with fused output
        """
        logger.info(f"Fusion Agent processing: {state.get('document_id')}")
        start_time = time.time()
        
        state["status"] = ProcessingStatus.FUSION.value
        state["current_agent"] = "fusion"
        
        try:
            vision_results = state.get("vision_results", {})
            ocr_results = state.get("ocr_results", {})
            layout_graph = state.get("layout_graph", {})
            text_analysis = state.get("text_analysis", {})
            
            # Fuse elements from all modalities
            elements = self._fuse_elements(
                vision_results, ocr_results, layout_graph
            )
            
            # Extract and structure tables
            tables = self._extract_tables(
                vision_results, ocr_results, layout_graph
            )
            
            # Build document structure
            structure = self._build_structure(layout_graph, text_analysis)
            
            # Compile metadata
            metadata = self._compile_metadata(
                state, vision_results, ocr_results, text_analysis
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            fused_document = FusedDocument(
                elements=elements,
                tables=tables,
                full_text=ocr_results.get("full_text", ""),
                structure=structure,
                metadata=metadata,
                processing_time_ms=processing_time
            )
            
            state["fused_output"] = fused_document.to_dict()
            
            logger.info(
                f"Fusion Agent completed: {len(elements)} elements, "
                f"{len(tables)} tables in {processing_time:.2f}ms"
            )
            
        except Exception as e:
            logger.error(f"Fusion Agent error: {e}")
            state["errors"].append(f"Fusion failed: {str(e)}")
            state["fused_output"] = FusedDocument().to_dict()
        
        return state
    
    def _fuse_elements(
        self,
        vision_results: Dict,
        ocr_results: Dict,
        layout_graph: Dict
    ) -> List[FusedElement]:
        """Fuse elements from all modalities"""
        elements = []
        
        # Get all regions from layout graph
        layout_nodes = layout_graph.get("nodes", [])
        vision_detections = vision_results.get("detections", [])
        ocr_blocks = ocr_results.get("text_blocks", [])
        
        # Process layout nodes as primary source
        for node in layout_nodes:
            element_id = node.get("id", str(uuid.uuid4())[:8])
            element_type = node.get("type", "text")
            
            # Find matching vision detection
            vision_match = self._find_matching_detection(
                node, vision_detections
            )
            
            # Find matching OCR blocks
            ocr_matches = self._find_matching_ocr(node, ocr_blocks)
            
            # Calculate confidences
            vision_conf = vision_match.get("confidence", 0.0) if vision_match else 0.0
            ocr_conf = (
                sum(b.get("confidence", 0) for b in ocr_matches) / len(ocr_matches)
                if ocr_matches else 0.0
            )
            layout_conf = 0.8  # Layout nodes have inherent confidence
            
            # Get content
            content = node.get("content", "")
            if not content and ocr_matches:
                content = " ".join(b.get("text", "") for b in ocr_matches)
            
            # Determine sources
            sources = ["layout"]
            if vision_match:
                sources.append("vision")
            if ocr_matches:
                sources.append("ocr")
            
            elements.append(FusedElement(
                id=element_id,
                type=element_type,
                content=content,
                page_number=node.get("page_number", 0),
                x1=node.get("x1", 0),
                y1=node.get("y1", 0),
                x2=node.get("x2", 0),
                y2=node.get("y2", 0),
                vision_confidence=vision_conf,
                ocr_confidence=ocr_conf,
                layout_confidence=layout_conf,
                sources=sources
            ))
        
        # Add any vision detections not matched to layout
        for detection in vision_detections:
            if not self._is_detection_covered(detection, elements):
                element_id = str(uuid.uuid4())[:8]
                
                elements.append(FusedElement(
                    id=element_id,
                    type=detection.get("label", "unknown"),
                    content="",
                    page_number=detection.get("page_number", 0),
                    x1=detection.get("x1", 0),
                    y1=detection.get("y1", 0),
                    x2=detection.get("x2", 0),
                    y2=detection.get("y2", 0),
                    vision_confidence=detection.get("confidence", 0),
                    ocr_confidence=0.0,
                    layout_confidence=0.0,
                    sources=["vision"]
                ))
        
        return elements
    
    def _find_matching_detection(
        self,
        node: Dict,
        detections: List[Dict]
    ) -> Optional[Dict]:
        """Find vision detection matching a layout node"""
        node_box = (
            node.get("x1", 0),
            node.get("y1", 0),
            node.get("x2", 0),
            node.get("y2", 0)
        )
        
        best_match = None
        best_iou = 0.0
        
        for detection in detections:
            if detection.get("page_number") != node.get("page_number"):
                continue
            
            det_box = (
                detection.get("x1", 0),
                detection.get("y1", 0),
                detection.get("x2", 0),
                detection.get("y2", 0)
            )
            
            iou = self._calculate_iou(node_box, det_box)
            
            if iou > best_iou and iou > 0.3:
                best_iou = iou
                best_match = detection
        
        return best_match
    
    def _find_matching_ocr(
        self,
        node: Dict,
        ocr_blocks: List[Dict]
    ) -> List[Dict]:
        """Find OCR blocks within a layout node"""
        matches = []
        
        for block in ocr_blocks:
            if block.get("page_number") != node.get("page_number"):
                continue
            
            # Check if block center is within node
            block_cx = (block.get("x1", 0) + block.get("x2", 0)) / 2
            block_cy = (block.get("y1", 0) + block.get("y2", 0)) / 2
            
            if (node.get("x1", 0) <= block_cx <= node.get("x2", 0) and
                node.get("y1", 0) <= block_cy <= node.get("y2", 0)):
                matches.append(block)
        
        return matches
    
    def _calculate_iou(
        self,
        box1: tuple,
        box2: tuple
    ) -> float:
        """Calculate Intersection over Union"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _is_detection_covered(
        self,
        detection: Dict,
        elements: List[FusedElement]
    ) -> bool:
        """Check if detection is already covered by an element"""
        det_box = (
            detection.get("x1", 0),
            detection.get("y1", 0),
            detection.get("x2", 0),
            detection.get("y2", 0)
        )
        
        for element in elements:
            if element.page_number != detection.get("page_number"):
                continue
            
            elem_box = (element.x1, element.y1, element.x2, element.y2)
            iou = self._calculate_iou(det_box, elem_box)
            
            if iou > 0.5:
                return True
        
        return False
    
    def _extract_tables(
        self,
        vision_results: Dict,
        ocr_results: Dict,
        layout_graph: Dict
    ) -> List[TableData]:
        """Extract and structure tables"""
        tables = []
        
        table_detections = vision_results.get("tables", [])
        ocr_blocks = ocr_results.get("text_blocks", [])
        
        for i, table_det in enumerate(table_detections):
            table_id = f"table_{i}"
            
            # Get OCR text within table bounds
            table_text_blocks = [
                b for b in ocr_blocks
                if (b.get("page_number") == table_det.get("page_number") and
                    table_det.get("x1", 0) <= b.get("x1", 0) <= table_det.get("x2", 0) and
                    table_det.get("y1", 0) <= b.get("y1", 0) <= table_det.get("y2", 0))
            ]
            
            # Simple table structure extraction
            # In production, would use more sophisticated table parsing
            rows, headers = self._parse_table_structure(
                table_text_blocks, table_det
            )
            
            tables.append(TableData(
                id=table_id,
                page_number=table_det.get("page_number", 0),
                headers=headers,
                rows=rows,
                x1=table_det.get("x1", 0),
                y1=table_det.get("y1", 0),
                x2=table_det.get("x2", 0),
                y2=table_det.get("y2", 0),
                confidence=table_det.get("confidence", 0)
            ))
        
        return tables
    
    def _parse_table_structure(
        self,
        text_blocks: List[Dict],
        table_bounds: Dict
    ) -> tuple:
        """Parse table structure from text blocks"""
        if not text_blocks:
            return [], []
        
        # Sort by row (y position) then column (x position)
        sorted_blocks = sorted(
            text_blocks,
            key=lambda b: (b.get("y1", 0), b.get("x1", 0))
        )
        
        # Group into rows
        rows = []
        current_row = []
        current_y = sorted_blocks[0].get("y1", 0)
        row_threshold = 20
        
        for block in sorted_blocks:
            if abs(block.get("y1", 0) - current_y) > row_threshold:
                if current_row:
                    rows.append([b.get("text", "") for b in current_row])
                current_row = [block]
                current_y = block.get("y1", 0)
            else:
                current_row.append(block)
        
        if current_row:
            rows.append([b.get("text", "") for b in current_row])
        
        # First row as headers (simple heuristic)
        headers = rows[0] if rows else []
        data_rows = rows[1:] if len(rows) > 1 else []
        
        return data_rows, headers
    
    def _build_structure(
        self,
        layout_graph: Dict,
        text_analysis: Dict
    ) -> Dict[str, Any]:
        """Build document structure"""
        return {
            "document_type": text_analysis.get("document_type", "unknown"),
            "page_count": len(layout_graph.get("page_structure", {})),
            "sections": self._extract_sections(layout_graph),
            "reading_order": layout_graph.get("reading_order", []),
            "topics": text_analysis.get("topics", [])
        }
    
    def _extract_sections(self, layout_graph: Dict) -> List[Dict]:
        """Extract document sections from layout"""
        sections = []
        nodes = layout_graph.get("nodes", [])
        
        current_section = None
        
        for node in nodes:
            if node.get("type") in ["heading", "title"]:
                if current_section:
                    sections.append(current_section)
                
                current_section = {
                    "title": node.get("content", "")[:100],
                    "level": 1 if node.get("type") == "title" else 2,
                    "page": node.get("page_number", 0),
                    "elements": []
                }
            elif current_section:
                current_section["elements"].append(node.get("id"))
        
        if current_section:
            sections.append(current_section)
        
        return sections
    
    def _compile_metadata(
        self,
        state: DocumentState,
        vision_results: Dict,
        ocr_results: Dict,
        text_analysis: Dict
    ) -> Dict[str, Any]:
        """Compile document metadata"""
        return {
            "document_id": state.get("document_id", ""),
            "filename": state.get("filename", ""),
            "file_type": state.get("file_type", ""),
            "page_count": state.get("page_count", 0),
            "word_count": ocr_results.get("word_count", 0),
            "language": text_analysis.get("language", "en"),
            "document_type": text_analysis.get("document_type", "unknown"),
            "has_tables": len(vision_results.get("tables", [])) > 0,
            "has_figures": len(vision_results.get("figures", [])) > 0,
            "has_charts": len(vision_results.get("charts", [])) > 0,
            "has_signatures": len(vision_results.get("signatures", [])) > 0
        }


def fusion_agent_node(state: DocumentState) -> DocumentState:
    """LangGraph node function for Fusion Agent"""
    agent = FusionAgent()
    return agent(state)
