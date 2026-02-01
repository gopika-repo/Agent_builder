"""
Chat Routes

Endpoints for querying documents with natural language.
Supports ELI5 vs Expert mode.
"""

import logging
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ...rag.rag_pipeline import RAGPipeline
from .documents import documents_store

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatQuery(BaseModel):
    """Chat query model"""
    query: str = Field(..., description="Natural language query")
    mode: str = Field(
        default="standard",
        description="Response mode: 'standard', 'eli5', or 'expert'"
    )
    include_sources: bool = Field(
        default=True,
        description="Include source references in response"
    )


class ChatResponse(BaseModel):
    """Chat response model"""
    query: str
    answer: str
    mode: str
    sources: Optional[List[dict]] = None


class CompareExplanationResponse(BaseModel):
    """Compare explanations response"""
    query: str
    eli5: str
    expert: str
    sources: Optional[List[dict]] = None


@router.post("/{document_id}", response_model=ChatResponse)
async def chat_with_document(
    document_id: str,
    query: ChatQuery
):
    """
    Query a document using natural language.
    
    Modes:
    - **standard**: Balanced response
    - **eli5**: Simple explanation (Explain Like I'm 5)
    - **expert**: Technical analysis with citations
    """
    # Verify document exists and is processed
    doc = documents_store.get(document_id)
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if doc["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Document not ready. Status: {doc['status']}"
        )
    
    # Validate mode
    valid_modes = ["standard", "eli5", "expert"]
    if query.mode not in valid_modes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode. Must be one of: {valid_modes}"
        )
    
    try:
        rag = RAGPipeline()
        result = rag.query(
            query=query.query,
            document_id=document_id,
            mode=query.mode,
            include_sources=query.include_sources
        )
        
        return ChatResponse(
            query=query.query,
            answer=result["answer"],
            mode=query.mode,
            sources=result.get("sources") if query.include_sources else None
        )
        
    except Exception as e:
        logger.error(f"Chat query failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}"
        )


@router.post("/{document_id}/explain", response_model=CompareExplanationResponse)
async def compare_explanations(
    document_id: str,
    query: str = Query(..., description="Query to explain in both modes")
):
    """
    Get both ELI5 and Expert explanations for comparison.
    
    This is the WOW feature - same document content explained at two levels.
    """
    # Verify document exists and is processed
    doc = documents_store.get(document_id)
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if doc["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Document not ready. Status: {doc['status']}"
        )
    
    try:
        rag = RAGPipeline()
        result = rag.compare_explanations(
            query=query,
            document_id=document_id
        )
        
        return CompareExplanationResponse(
            query=query,
            eli5=result["eli5"],
            expert=result["expert"],
            sources=result.get("sources")
        )
        
    except Exception as e:
        logger.error(f"Compare explanations failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Comparison failed: {str(e)}"
        )


@router.get("/{document_id}/suggest")
async def get_suggested_questions(
    document_id: str,
    count: int = Query(5, ge=1, le=10)
):
    """
    Get suggested questions for a document.
    
    Based on document content and key topics.
    """
    doc = documents_store.get(document_id)
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if doc["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Document not ready. Status: {doc['status']}"
        )
    
    results = doc.get("results", {})
    text_analysis = results.get("text_analysis", {})
    fused_output = results.get("fused_output", {})
    
    # Generate suggested questions based on document content
    suggestions = []
    
    # Based on document type
    doc_type = text_analysis.get("document_type", "")
    if doc_type == "financial_report":
        suggestions.extend([
            "What are the key financial metrics in this report?",
            "How does revenue compare to the previous period?",
            "What are the main risk factors mentioned?"
        ])
    elif doc_type == "legal_contract":
        suggestions.extend([
            "What are the main obligations of each party?",
            "What is the termination clause?",
            "What liabilities are mentioned?"
        ])
    
    # Based on key points
    key_points = text_analysis.get("key_points", [])
    if key_points:
        suggestions.append(f"Can you explain: {key_points[0]}?")
    
    # Based on entities
    entities = text_analysis.get("entities", [])
    if entities:
        for entity in entities[:2]:
            suggestions.append(f"What does the document say about {entity.get('text', '')}?")
    
    # Based on tables
    tables = fused_output.get("tables", [])
    if tables:
        suggestions.append("What information is shown in the tables?")
    
    # Generic fallbacks
    suggestions.extend([
        "What is this document about?",
        "What are the main conclusions?",
        "Summarize the key points."
    ])
    
    return {
        "document_id": document_id,
        "suggestions": suggestions[:count]
    }


@router.get("/{document_id}/search")
async def search_document(
    document_id: str,
    query: str = Query(..., description="Search query"),
    modalities: str = Query("text,tables,images", description="Comma-separated modalities"),
    limit: int = Query(10, ge=1, le=50)
):
    """
    Search document content across modalities.
    
    Returns matching content with relevance scores.
    """
    doc = documents_store.get(document_id)
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if doc["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Document not ready. Status: {doc['status']}"
        )
    
    try:
        from ...rag.retriever import MultiModalRetriever
        
        retriever = MultiModalRetriever()
        modality_list = [m.strip() for m in modalities.split(",")]
        
        results = retriever.retrieve(
            query=query,
            document_id=document_id,
            modalities=modality_list,
            limit=limit
        )
        
        return {
            "document_id": document_id,
            "query": query,
            "results": [r.to_dict() for r in results]
        }
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )
