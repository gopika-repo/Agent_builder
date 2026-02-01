"""
Shared State Schema for LangGraph Agents

Defines the state that flows between all 6 agents in the document processing pipeline.
"""

from typing import TypedDict, List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class ProcessingStatus(str, Enum):
    """Status of document processing"""
    PENDING = "pending"
    VISION_PROCESSING = "vision_processing"
    OCR_PROCESSING = "ocr_processing"
    LAYOUT_ANALYSIS = "layout_analysis"
    TEXT_REASONING = "text_reasoning"
    FUSION = "fusion"
    VALIDATION = "validation"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PageImage:
    """Represents a single page image"""
    page_number: int
    image: np.ndarray
    width: int
    height: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "width": self.width,
            "height": self.height
        }


@dataclass
class Detection:
    """A detected region from vision agent"""
    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    confidence: float
    page_number: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "label": self.label,
            "confidence": self.confidence,
            "page_number": self.page_number
        }


@dataclass
class VisionOutput:
    """Output from Vision Agent"""
    detections: List[Detection] = field(default_factory=list)
    tables: List[Detection] = field(default_factory=list)
    figures: List[Detection] = field(default_factory=list)
    charts: List[Detection] = field(default_factory=list)
    signatures: List[Detection] = field(default_factory=list)
    processing_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "detections": [d.to_dict() for d in self.detections],
            "tables": [t.to_dict() for t in self.tables],
            "figures": [f.to_dict() for f in self.figures],
            "charts": [c.to_dict() for c in self.charts],
            "signatures": [s.to_dict() for s in self.signatures],
            "processing_time_ms": self.processing_time_ms,
            "summary": {
                "total": len(self.detections),
                "tables": len(self.tables),
                "figures": len(self.figures),
                "charts": len(self.charts),
                "signatures": len(self.signatures)
            }
        }


@dataclass
class TextBlockData:
    """A text block from OCR"""
    text: str
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    page_number: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "confidence": self.confidence,
            "page_number": self.page_number
        }


@dataclass
class OCROutput:
    """Output from OCR Agent"""
    text_blocks: List[TextBlockData] = field(default_factory=list)
    full_text: str = ""
    page_texts: Dict[int, str] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    engine_used: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text_blocks": [b.to_dict() for b in self.text_blocks],
            "full_text": self.full_text,
            "page_texts": self.page_texts,
            "processing_time_ms": self.processing_time_ms,
            "engine_used": self.engine_used,
            "word_count": len(self.full_text.split())
        }


@dataclass
class LayoutNode:
    """A node in the document layout graph"""
    id: str
    type: str  # header, paragraph, table, figure, list, etc.
    content: str
    x1: float
    y1: float
    x2: float
    y2: float
    page_number: int
    children: List[str] = field(default_factory=list)
    parent: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "content": self.content[:200] + "..." if len(self.content) > 200 else self.content,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "page_number": self.page_number,
            "children": self.children,
            "parent": self.parent
        }


@dataclass
class LayoutGraph:
    """Document layout structure"""
    nodes: List[LayoutNode] = field(default_factory=list)
    edges: List[tuple] = field(default_factory=list)  # (from_id, to_id, relationship)
    page_structure: Dict[int, List[str]] = field(default_factory=dict)  # page -> node ids
    reading_order: List[str] = field(default_factory=list)  # node ids in reading order
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [{"from": e[0], "to": e[1], "rel": e[2]} for e in self.edges],
            "page_structure": self.page_structure,
            "reading_order": self.reading_order
        }


@dataclass
class Entity:
    """An extracted entity"""
    text: str
    type: str  # person, organization, date, money, etc.
    confidence: float
    source_page: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "type": self.type,
            "confidence": self.confidence,
            "source_page": self.source_page
        }


@dataclass
class TextAnalysis:
    """Output from Text Reasoning Agent"""
    summary: str = ""
    summary_eli5: str = ""  # ELI5 version
    summary_expert: str = ""  # Expert version
    key_points: List[str] = field(default_factory=list)
    entities: List[Entity] = field(default_factory=list)
    document_type: str = ""
    language: str = "en"
    topics: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "summary_eli5": self.summary_eli5,
            "summary_expert": self.summary_expert,
            "key_points": self.key_points,
            "entities": [e.to_dict() for e in self.entities],
            "document_type": self.document_type,
            "language": self.language,
            "topics": self.topics,
            "processing_time_ms": self.processing_time_ms
        }


@dataclass
class TableData:
    """Structured table data"""
    id: str
    page_number: int
    headers: List[str] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)
    x1: float = 0
    y1: float = 0
    x2: float = 0
    y2: float = 0
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "page_number": self.page_number,
            "headers": self.headers,
            "rows": self.rows,
            "num_rows": len(self.rows),
            "num_cols": len(self.headers) if self.headers else (len(self.rows[0]) if self.rows else 0),
            "confidence": self.confidence
        }


@dataclass
class FusedElement:
    """A fused element combining multiple modality outputs"""
    id: str
    type: str  # text, table, figure, chart, signature
    content: Any  # Text string or structured data
    page_number: int
    x1: float
    y1: float
    x2: float
    y2: float
    vision_confidence: float = 0.0
    ocr_confidence: float = 0.0
    layout_confidence: float = 0.0
    sources: List[str] = field(default_factory=list)  # Which agents contributed
    
    def to_dict(self) -> Dict[str, Any]:
        content_preview = self.content
        if isinstance(self.content, str) and len(self.content) > 200:
            content_preview = self.content[:200] + "..."
        
        return {
            "id": self.id,
            "type": self.type,
            "content": content_preview,
            "page_number": self.page_number,
            "bbox": [self.x1, self.y1, self.x2, self.y2],
            "confidences": {
                "vision": self.vision_confidence,
                "ocr": self.ocr_confidence,
                "layout": self.layout_confidence
            },
            "sources": self.sources
        }


@dataclass
class FusedDocument:
    """Complete fused document representation"""
    elements: List[FusedElement] = field(default_factory=list)
    tables: List[TableData] = field(default_factory=list)
    full_text: str = ""
    structure: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "elements": [e.to_dict() for e in self.elements],
            "tables": [t.to_dict() for t in self.tables],
            "full_text": self.full_text,
            "structure": self.structure,
            "metadata": self.metadata,
            "processing_time_ms": self.processing_time_ms
        }


@dataclass
class FieldConfidence:
    """Confidence score for a specific field"""
    field_id: str
    field_name: str
    value: Any
    ocr_confidence: float
    vision_confidence: float
    llm_confidence: float
    combined_confidence: float
    needs_review: bool
    review_reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_id": self.field_id,
            "field_name": self.field_name,
            "value": self.value,
            "confidences": {
                "ocr": self.ocr_confidence,
                "vision": self.vision_confidence,
                "llm": self.llm_confidence,
                "combined": self.combined_confidence
            },
            "needs_review": self.needs_review,
            "review_reason": self.review_reason
        }


@dataclass
class ConfidenceReport:
    """Complete confidence report from Validation Agent"""
    overall_confidence: float = 0.0
    field_scores: List[FieldConfidence] = field(default_factory=list)
    items_needing_review: List[str] = field(default_factory=list)
    modality_agreement: float = 0.0
    processing_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_confidence": self.overall_confidence,
            "field_scores": [f.to_dict() for f in self.field_scores],
            "items_needing_review": self.items_needing_review,
            "modality_agreement": self.modality_agreement,
            "processing_time_ms": self.processing_time_ms
        }


class DocumentState(TypedDict, total=False):
    """
    Shared state for the document processing pipeline.
    
    This state flows through all 6 agents in the LangGraph workflow.
    """
    # Document identification
    document_id: str
    filename: str
    file_type: str
    
    # Raw input
    pages: List[Dict]  # List of PageImage as dicts
    page_count: int
    
    # Agent outputs
    vision_results: Dict  # VisionOutput as dict
    ocr_results: Dict  # OCROutput as dict
    layout_graph: Dict  # LayoutGraph as dict
    text_analysis: Dict  # TextAnalysis as dict
    fused_output: Dict  # FusedDocument as dict
    confidence_scores: Dict  # ConfidenceReport as dict
    
    # Processing metadata
    status: str
    current_agent: str
    errors: List[str]
    warnings: List[str]
    processing_start_time: str
    processing_end_time: str
    total_processing_time_ms: float
    
    # Configuration
    config: Dict[str, Any]


def create_initial_state(
    document_id: str,
    filename: str,
    file_type: str,
    pages: List[PageImage],
    config: Optional[Dict[str, Any]] = None
) -> DocumentState:
    """Create initial state for document processing"""
    return DocumentState(
        document_id=document_id,
        filename=filename,
        file_type=file_type,
        pages=[p.to_dict() for p in pages],
        page_count=len(pages),
        vision_results={},
        ocr_results={},
        layout_graph={},
        text_analysis={},
        fused_output={},
        confidence_scores={},
        status=ProcessingStatus.PENDING.value,
        current_agent="",
        errors=[],
        warnings=[],
        processing_start_time="",
        processing_end_time="",
        total_processing_time_ms=0.0,
        config=config or {}
    )
