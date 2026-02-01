"""
Advanced Features API Routes

Endpoints for:
- Multi-document comparison
- Table reasoning
- Conflict resolution
- Confidence heatmaps
- Cross-modal retrieval
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/advanced", tags=["advanced"])


# === Request/Response Models ===

class MultiDocumentQuery(BaseModel):
    """Query across multiple documents"""
    query: str = Field(..., description="Natural language query")
    document_ids: List[str] = Field(..., description="List of document IDs to search")
    compare: bool = Field(False, description="Enable cross-document comparison")
    mode: str = Field("standard", description="Response mode: standard, eli5, expert")


class TableReasoningQuery(BaseModel):
    """Query for table reasoning"""
    document_id: str
    table_id: str
    query: str = Field(..., description="Question about the table")


class CrossModalQuery(BaseModel):
    """Cross-modal retrieval query"""
    query: str
    document_ids: List[str]
    modalities: List[str] = Field(["text", "table", "image"], description="Modalities to search")
    rerank: bool = Field(True, description="Enable LLM-based re-ranking")


class ConflictResolutionRequest(BaseModel):
    """Request conflict analysis"""
    document_id: str


class HeatmapRequest(BaseModel):
    """Request confidence heatmap"""
    document_id: str
    page_number: int = 0


# === Multi-Document Reasoning ===

@router.post("/multi-document/query")
async def query_multiple_documents(request: MultiDocumentQuery) -> Dict[str, Any]:
    """
    Query across multiple documents with optional comparison.
    
    This enables questions like:
    - "Compare revenue between Report A and Report B"
    - "What are the common themes across all documents?"
    """
    try:
        from ..rag.cross_modal_retriever import HybridCrossModalRetriever
        
        retriever = HybridCrossModalRetriever()
        result = retriever.retrieve_multi_document(
            query=request.query,
            document_ids=request.document_ids,
            compare=request.compare
        )
        
        return {
            "success": True,
            "query": request.query,
            "documents": request.document_ids,
            "result": result,
            "mode": request.mode
        }
        
    except Exception as e:
        logger.error(f"Multi-document query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/multi-document/compare/{doc_id_1}/{doc_id_2}")
async def compare_documents(
    doc_id_1: str,
    doc_id_2: str,
    aspect: Optional[str] = Query(None, description="Specific aspect to compare")
) -> Dict[str, Any]:
    """
    Compare two documents directly.
    """
    try:
        # Get document summaries
        query = aspect or "key findings and numbers"
        
        from ..rag.cross_modal_retriever import HybridCrossModalRetriever
        
        retriever = HybridCrossModalRetriever()
        
        doc1_results = retriever.retrieve(
            query=query,
            document_ids=[doc_id_1]
        )
        
        doc2_results = retriever.retrieve(
            query=query,
            document_ids=[doc_id_2]
        )
        
        return {
            "success": True,
            "comparison": {
                "document_1": {
                    "id": doc_id_1,
                    "findings": doc1_results.to_dict()
                },
                "document_2": {
                    "id": doc_id_2,
                    "findings": doc2_results.to_dict()
                }
            },
            "aspect": aspect or "general"
        }
        
    except Exception as e:
        logger.error(f"Document comparison failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# === Table Reasoning ===

@router.post("/table/reason")
async def reason_about_table(request: TableReasoningQuery) -> Dict[str, Any]:
    """
    Perform intelligent reasoning on a table.
    
    Supports:
    - "What is the maximum revenue?"
    - "Is there an increasing trend?"
    - "Compare Q1 and Q4"
    """
    try:
        from ..agents.table_reasoning_agent import TableReasoningAgent
        
        # Get table data (mock for now, would fetch from storage)
        table_data = await _get_table_data(request.document_id, request.table_id)
        
        agent = TableReasoningAgent()
        result = agent.analyze(
            table_data=table_data,
            query=request.query,
            table_id=request.table_id
        )
        
        return {
            "success": True,
            "document_id": request.document_id,
            "table_id": request.table_id,
            "query": request.query,
            "result": result.to_dict()
        }
        
    except Exception as e:
        logger.error(f"Table reasoning failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/table/{document_id}/list")
async def list_tables(document_id: str) -> Dict[str, Any]:
    """List all tables in a document."""
    # Would fetch from document storage
    return {
        "document_id": document_id,
        "tables": []  # Populated from document results
    }


# === Cross-Modal Retrieval ===

@router.post("/retrieval/cross-modal")
async def cross_modal_retrieval(request: CrossModalQuery) -> Dict[str, Any]:
    """
    Perform hybrid cross-modal retrieval.
    
    Searches across text, tables, and images with LLM re-ranking.
    """
    try:
        from ..rag.cross_modal_retriever import HybridCrossModalRetriever
        
        retriever = HybridCrossModalRetriever()
        result = retriever.retrieve(
            query=request.query,
            document_ids=request.document_ids,
            k_final=10
        )
        
        return {
            "success": True,
            "query": request.query,
            "result": result.to_dict(),
            "modalities_searched": request.modalities,
            "reranked": request.rerank
        }
        
    except Exception as e:
        logger.error(f"Cross-modal retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# === Conflict Resolution ===

@router.post("/conflicts/analyze")
async def analyze_conflicts(request: ConflictResolutionRequest) -> Dict[str, Any]:
    """
    Analyze document for conflicts between modalities.
    
    Detects and resolves disagreements between OCR, Vision, and LLM.
    """
    try:
        from ..agents.conflict_resolution import ConflictResolutionEngine
        
        # Get document results (would fetch from storage)
        ocr_results = await _get_document_ocr(request.document_id)
        vision_results = await _get_document_vision(request.document_id)
        fused_output = await _get_document_fused(request.document_id)
        
        engine = ConflictResolutionEngine()
        report = engine.analyze(
            document_id=request.document_id,
            ocr_results=ocr_results,
            vision_results=vision_results,
            fused_output=fused_output
        )
        
        summary = engine.generate_summary(report)
        
        return {
            "success": True,
            "document_id": request.document_id,
            "report": report.to_dict(),
            "summary": summary
        }
        
    except Exception as e:
        logger.error(f"Conflict analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conflicts/{document_id}/summary")
async def get_conflict_summary(document_id: str) -> Dict[str, Any]:
    """Get conflict resolution summary for a document."""
    try:
        from ..agents.conflict_resolution import ConflictResolutionEngine
        
        # Get cached conflict report
        # In production, this would be stored during processing
        return {
            "document_id": document_id,
            "total_conflicts": 0,
            "resolved": 0,
            "pending_review": 0,
            "summary": "No conflicts found or document not yet analyzed."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === Confidence Heatmap ===

@router.post("/heatmap/generate")
async def generate_heatmap(request: HeatmapRequest) -> Dict[str, Any]:
    """
    Generate confidence heatmap for a document page.
    
    Returns color-coded visualization data for frontend.
    """
    try:
        from ..agents.confidence_heatmap import (
            ConfidenceHeatmapGenerator,
            generate_confidence_react_data
        )
        
        # Get document results
        vision_results = await _get_document_vision(request.document_id)
        ocr_results = await _get_document_ocr(request.document_id)
        validation_results = await _get_document_validation(request.document_id)
        
        generator = ConfidenceHeatmapGenerator()
        
        # For API, return React-compatible data (no image processing)
        from ..agents.confidence_heatmap import ConfidenceRegion
        
        regions = []
        
        # Extract regions from results
        for detection in vision_results.get("detections", []):
            if detection.get("page_number", 0) == request.page_number:
                regions.append(ConfidenceRegion(
                    x1=detection.get("x1", 0),
                    y1=detection.get("y1", 0),
                    x2=detection.get("x2", 0),
                    y2=detection.get("y2", 0),
                    confidence=detection.get("confidence", 0),
                    label=detection.get("label", "unknown"),
                    source="vision",
                    page_number=request.page_number
                ))
        
        for block in ocr_results.get("text_blocks", []):
            if block.get("page_number", 0) == request.page_number:
                regions.append(ConfidenceRegion(
                    x1=block.get("x1", 0),
                    y1=block.get("y1", 0),
                    x2=block.get("x2", 0),
                    y2=block.get("y2", 0),
                    confidence=block.get("confidence", 0),
                    label="text",
                    source="ocr",
                    page_number=request.page_number
                ))
        
        react_data = generate_confidence_react_data(regions)
        legend = generator.get_color_legend()
        
        return {
            "success": True,
            "document_id": request.document_id,
            "page_number": request.page_number,
            "regions": react_data,
            "legend": legend,
            "statistics": {
                "total_regions": len(regions),
                "low_confidence": sum(1 for r in regions if r.confidence < 0.6),
                "average_confidence": sum(r.confidence for r in regions) / len(regions) if regions else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Heatmap generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/heatmap/{document_id}/legend")
async def get_heatmap_legend(document_id: str) -> Dict[str, Any]:
    """Get the color legend for confidence heatmaps."""
    from ..agents.confidence_heatmap import ConfidenceHeatmapGenerator
    
    generator = ConfidenceHeatmapGenerator()
    return generator.get_color_legend()


# === Pipeline Health ===

@router.get("/health/pipeline")
async def get_pipeline_health() -> Dict[str, Any]:
    """
    Get self-healing pipeline health status.
    """
    try:
        from ..agents.self_healing import SelfHealingPipeline
        
        # Get global pipeline instance (would be singleton in production)
        pipeline = SelfHealingPipeline()
        health = pipeline.get_health()
        
        return {
            "success": True,
            "health": health.to_dict(),
            "message": "Pipeline is healthy" if health.is_healthy else "Some components degraded"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Could not retrieve pipeline health"
        }


@router.post("/health/reset")
async def reset_pipeline_health() -> Dict[str, Any]:
    """
    Reset pipeline health status after manual intervention.
    """
    try:
        from ..agents.self_healing import SelfHealingPipeline
        
        pipeline = SelfHealingPipeline()
        pipeline.reset_health()
        
        return {
            "success": True,
            "message": "Pipeline health reset successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === Helper Functions ===

async def _get_table_data(document_id: str, table_id: str) -> Dict[str, Any]:
    """Get table data from storage (placeholder)."""
    # In production, fetch from document storage
    return {
        "id": table_id,
        "headers": ["Quarter", "Revenue", "Expenses", "Profit"],
        "rows": [
            ["Q1", "120", "100", "20"],
            ["Q2", "125", "105", "20"],
            ["Q3", "130", "110", "20"],
            ["Q4", "125", "105", "20"]
        ]
    }


async def _get_document_ocr(document_id: str) -> Dict[str, Any]:
    """Get OCR results from storage (placeholder)."""
    return {"text_blocks": [], "full_text": ""}


async def _get_document_vision(document_id: str) -> Dict[str, Any]:
    """Get vision results from storage (placeholder)."""
    return {"detections": [], "tables": [], "figures": []}


async def _get_document_fused(document_id: str) -> Dict[str, Any]:
    """Get fused output from storage (placeholder)."""
    return {"elements": [], "tables": [], "entities": []}


async def _get_document_validation(document_id: str) -> Dict[str, Any]:
    """Get validation results from storage (placeholder)."""
    return {"field_scores": [], "overall_confidence": 0.0}
