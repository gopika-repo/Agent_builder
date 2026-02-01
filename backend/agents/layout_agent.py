"""
Layout Agent

Analyzes spatial relationships in documents to build a structural graph.
Groups content into headers, paragraphs, tables, captions, etc.
"""

import logging
from typing import Dict, Any, List, Tuple
import time
from dataclasses import dataclass
import uuid

from .state import (
    DocumentState, ProcessingStatus,
    LayoutGraph, LayoutNode
)

logger = logging.getLogger(__name__)


@dataclass
class SpatialRegion:
    """A region with spatial properties"""
    x1: float
    y1: float
    x2: float
    y2: float
    content: str
    region_type: str
    confidence: float
    page_number: int
    
    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2
    
    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1


class LayoutAgent:
    """
    Layout Agent for document structure analysis.
    
    Analyzes spatial relationships between detected elements
    and builds a document layout graph with reading order.
    """
    
    def __init__(
        self,
        column_threshold: float = 0.3,
        paragraph_gap_ratio: float = 1.5
    ):
        """
        Initialize Layout Agent.
        
        Args:
            column_threshold: Threshold for detecting columns
            paragraph_gap_ratio: Gap ratio for paragraph detection
        """
        self.column_threshold = column_threshold
        self.paragraph_gap_ratio = paragraph_gap_ratio
    
    def __call__(self, state: DocumentState) -> DocumentState:
        """
        Process document through layout agent.
        
        Args:
            state: Current document state
            
        Returns:
            Updated state with layout graph
        """
        logger.info(f"Layout Agent processing document: {state.get('document_id')}")
        start_time = time.time()
        
        state["status"] = ProcessingStatus.LAYOUT_ANALYSIS.value
        state["current_agent"] = "layout"
        
        try:
            vision_results = state.get("vision_results", {})
            ocr_results = state.get("ocr_results", {})
            pages = state.get("pages", [])
            
            # Build regions from vision and OCR outputs
            regions = self._build_regions(vision_results, ocr_results)
            
            # Analyze layout structure
            nodes, edges = self._analyze_structure(regions, pages)
            
            # Determine reading order
            reading_order = self._determine_reading_order(nodes)
            
            # Group by page
            page_structure = self._group_by_page(nodes)
            
            layout_graph = LayoutGraph(
                nodes=nodes,
                edges=edges,
                page_structure=page_structure,
                reading_order=reading_order
            )
            
            state["layout_graph"] = layout_graph.to_dict()
            
            processing_time = (time.time() - start_time) * 1000
            
            logger.info(
                f"Layout Agent completed: {len(nodes)} nodes, "
                f"{len(edges)} edges in {processing_time:.2f}ms"
            )
            
        except Exception as e:
            logger.error(f"Layout Agent error: {e}")
            state["errors"].append(f"Layout analysis failed: {str(e)}")
            state["layout_graph"] = LayoutGraph().to_dict()
        
        return state
    
    def _build_regions(
        self,
        vision_results: Dict,
        ocr_results: Dict
    ) -> List[SpatialRegion]:
        """Build spatial regions from vision and OCR outputs"""
        regions = []
        
        # Add regions from vision detections
        for detection in vision_results.get("detections", []):
            regions.append(SpatialRegion(
                x1=detection.get("x1", 0),
                y1=detection.get("y1", 0),
                x2=detection.get("x2", 0),
                y2=detection.get("y2", 0),
                content="",
                region_type=detection.get("label", "unknown"),
                confidence=detection.get("confidence", 0),
                page_number=detection.get("page_number", 0)
            ))
        
        # Add regions from OCR text blocks
        text_blocks = ocr_results.get("text_blocks", [])
        
        # Group nearby text blocks into paragraphs
        paragraph_regions = self._group_text_into_paragraphs(text_blocks)
        regions.extend(paragraph_regions)
        
        return regions
    
    def _group_text_into_paragraphs(
        self,
        text_blocks: List[Dict]
    ) -> List[SpatialRegion]:
        """Group text blocks into paragraph regions"""
        if not text_blocks:
            return []
        
        # Sort by position
        sorted_blocks = sorted(text_blocks, key=lambda b: (b.get("y1", 0), b.get("x1", 0)))
        
        paragraphs = []
        current_paragraph = [sorted_blocks[0]]
        
        for block in sorted_blocks[1:]:
            prev = current_paragraph[-1]
            
            # Check if on same line or nearby
            y_diff = block.get("y1", 0) - prev.get("y2", 0)
            avg_height = (prev.get("y2", 0) - prev.get("y1", 0))
            
            if y_diff < avg_height * self.paragraph_gap_ratio:
                current_paragraph.append(block)
            else:
                # Create paragraph from accumulated blocks
                if current_paragraph:
                    paragraph = self._create_paragraph_region(current_paragraph)
                    paragraphs.append(paragraph)
                current_paragraph = [block]
        
        # Don't forget the last paragraph
        if current_paragraph:
            paragraph = self._create_paragraph_region(current_paragraph)
            paragraphs.append(paragraph)
        
        return paragraphs
    
    def _create_paragraph_region(self, blocks: List[Dict]) -> SpatialRegion:
        """Create a paragraph region from text blocks"""
        if not blocks:
            return None
        
        x1 = min(b.get("x1", 0) for b in blocks)
        y1 = min(b.get("y1", 0) for b in blocks)
        x2 = max(b.get("x2", 0) for b in blocks)
        y2 = max(b.get("y2", 0) for b in blocks)
        
        content = " ".join(b.get("text", "") for b in blocks)
        avg_confidence = sum(b.get("confidence", 0) for b in blocks) / len(blocks)
        page = blocks[0].get("page_number", 0)
        
        # Determine region type based on position and content
        region_type = self._classify_text_region(content, y1, x1, x2)
        
        return SpatialRegion(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            content=content,
            region_type=region_type,
            confidence=avg_confidence,
            page_number=page
        )
    
    def _classify_text_region(
        self,
        content: str,
        y: float,
        x1: float,
        x2: float
    ) -> str:
        """Classify a text region based on content and position"""
        # Simple heuristics for classification
        content_lower = content.lower().strip()
        
        # Check for heading patterns
        if len(content) < 100:
            if content.isupper() or content_lower.startswith(("chapter", "section")):
                return "heading"
            if y < 200:  # Near top
                return "title"
        
        # Check for list items
        if content_lower.startswith(("•", "-", "*", "1.", "2.", "a.", "b.")):
            return "list"
        
        return "paragraph"
    
    def _analyze_structure(
        self,
        regions: List[SpatialRegion],
        pages: List[Dict]
    ) -> Tuple[List[LayoutNode], List[tuple]]:
        """Analyze document structure and build graph"""
        nodes = []
        edges = []
        
        for region in regions:
            node_id = str(uuid.uuid4())[:8]
            
            node = LayoutNode(
                id=node_id,
                type=region.region_type,
                content=region.content,
                x1=region.x1,
                y1=region.y1,
                x2=region.x2,
                y2=region.y2,
                page_number=region.page_number
            )
            nodes.append(node)
        
        # Build edges based on reading flow
        sorted_nodes = sorted(nodes, key=lambda n: (n.page_number, n.y1, n.x1))
        
        for i in range(len(sorted_nodes) - 1):
            current = sorted_nodes[i]
            next_node = sorted_nodes[i + 1]
            
            # Determine relationship type
            if current.page_number == next_node.page_number:
                if abs(current.y2 - next_node.y1) < 50:
                    relationship = "follows"
                else:
                    relationship = "precedes"
            else:
                relationship = "next_page"
            
            edges.append((current.id, next_node.id, relationship))
        
        # Add containment relationships (e.g., caption under figure)
        for i, node in enumerate(nodes):
            if node.type == "caption":
                # Find nearest figure/table above
                for other in nodes:
                    if other.type in ["figure", "table", "chart"]:
                        if (other.page_number == node.page_number and
                            other.y2 < node.y1 and
                            abs(other.center_x - (node.x1 + node.x2) / 2) < 100):
                            node.parent = other.id
                            other.children.append(node.id)
                            edges.append((other.id, node.id, "has_caption"))
                            break
        
        return nodes, edges
    
    def _determine_reading_order(self, nodes: List[LayoutNode]) -> List[str]:
        """Determine reading order for nodes"""
        # Sort by page, then by vertical position, then by horizontal
        sorted_nodes = sorted(
            nodes,
            key=lambda n: (n.page_number, n.y1, n.x1)
        )
        
        return [node.id for node in sorted_nodes]
    
    def _group_by_page(self, nodes: List[LayoutNode]) -> Dict[int, List[str]]:
        """Group node IDs by page number"""
        page_structure = {}
        
        for node in nodes:
            page = node.page_number
            if page not in page_structure:
                page_structure[page] = []
            page_structure[page].append(node.id)
        
        return page_structure


def layout_agent_node(state: DocumentState) -> DocumentState:
    """LangGraph node function for Layout Agent"""
    agent = LayoutAgent()
    return agent(state)
