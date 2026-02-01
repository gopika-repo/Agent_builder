"""
Human Review Routes

Endpoints for human review workflow:
- Get flagged items
- Submit corrections
- Provide feedback
"""

import logging
from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from .documents import documents_store

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory corrections store
corrections_store = {}


class FlaggedItem(BaseModel):
    """A flagged item requiring review"""
    field_id: str
    field_name: str
    current_value: str
    confidence: float
    review_reason: str
    page_number: Optional[int] = None
    bbox: Optional[List[float]] = None


class Correction(BaseModel):
    """A correction submitted by human reviewer"""
    field_id: str
    original_value: str
    corrected_value: str
    correction_type: str = Field(
        default="edit",
        description="Type of correction: 'edit', 'confirm', 'reject'"
    )
    notes: Optional[str] = None


class CorrectionResponse(BaseModel):
    """Response after submitting correction"""
    field_id: str
    status: str
    message: str


class ReviewSummary(BaseModel):
    """Summary of review status"""
    document_id: str
    total_flagged: int
    reviewed: int
    pending: int
    corrections_made: int


@router.get("/{document_id}/flags")
async def get_flagged_items(
    document_id: str,
    page: Optional[int] = Query(None, description="Filter by page number"),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0)
):
    """
    Get items flagged for human review.
    
    Returns fields with low confidence that need verification.
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
    confidence_scores = results.get("confidence_scores", {})
    field_scores = confidence_scores.get("field_scores", [])
    
    flagged_items = []
    
    for field in field_scores:
        if field.get("needs_review", False):
            # Apply filters
            conf = field.get("confidences", {}).get("combined", 0)
            if conf < min_confidence:
                continue
            
            item = FlaggedItem(
                field_id=field.get("field_id", ""),
                field_name=field.get("field_name", ""),
                current_value=str(field.get("value", ""))[:500],
                confidence=conf,
                review_reason=field.get("review_reason", "Low confidence"),
                page_number=field.get("page_number"),
                bbox=field.get("bbox")
            )
            
            # Check if already corrected
            correction_key = f"{document_id}:{field.get('field_id')}"
            if correction_key in corrections_store:
                continue
            
            flagged_items.append(item)
    
    return {
        "document_id": document_id,
        "flagged_count": len(flagged_items),
        "items": [item.dict() for item in flagged_items]
    }


@router.put("/{document_id}/correct", response_model=CorrectionResponse)
async def submit_correction(
    document_id: str,
    correction: Correction
):
    """
    Submit a correction for a flagged field.
    
    Correction types:
    - **edit**: Change the value
    - **confirm**: Mark as correct despite low confidence
    - **reject**: Mark as incorrect/remove
    """
    doc = documents_store.get(document_id)
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Store correction
    correction_key = f"{document_id}:{correction.field_id}"
    corrections_store[correction_key] = {
        "document_id": document_id,
        "field_id": correction.field_id,
        "original_value": correction.original_value,
        "corrected_value": correction.corrected_value,
        "correction_type": correction.correction_type,
        "notes": correction.notes,
        "timestamp": datetime.utcnow().isoformat(),
        "applied": False
    }
    
    logger.info(f"Correction submitted: {correction_key}")
    
    return CorrectionResponse(
        field_id=correction.field_id,
        status="accepted",
        message=f"Correction recorded for field {correction.field_id}"
    )


@router.get("/{document_id}/corrections")
async def get_corrections(document_id: str):
    """Get all corrections for a document"""
    doc = documents_store.get(document_id)
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc_corrections = [
        v for k, v in corrections_store.items()
        if k.startswith(f"{document_id}:")
    ]
    
    return {
        "document_id": document_id,
        "correction_count": len(doc_corrections),
        "corrections": doc_corrections
    }


@router.get("/{document_id}/summary", response_model=ReviewSummary)
async def get_review_summary(document_id: str):
    """Get summary of review status for a document"""
    doc = documents_store.get(document_id)
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    results = doc.get("results", {})
    confidence_scores = results.get("confidence_scores", {})
    items_needing_review = confidence_scores.get("items_needing_review", [])
    
    # Count corrections
    doc_corrections = [
        k for k in corrections_store.keys()
        if k.startswith(f"{document_id}:")
    ]
    
    return ReviewSummary(
        document_id=document_id,
        total_flagged=len(items_needing_review),
        reviewed=len(doc_corrections),
        pending=len(items_needing_review) - len(doc_corrections),
        corrections_made=len([
            c for c in doc_corrections
            if corrections_store[c].get("correction_type") == "edit"
        ])
    )


@router.post("/{document_id}/apply-corrections")
async def apply_corrections(document_id: str):
    """
    Apply all pending corrections to the document.
    
    Updates the fused output with corrected values.
    """
    doc = documents_store.get(document_id)
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Get pending corrections
    doc_corrections = [
        (k, v) for k, v in corrections_store.items()
        if k.startswith(f"{document_id}:") and not v.get("applied")
    ]
    
    if not doc_corrections:
        return {
            "message": "No pending corrections to apply",
            "applied_count": 0
        }
    
    applied_count = 0
    
    # Apply corrections to fused output
    results = doc.get("results", {})
    fused_output = results.get("fused_output", {})
    elements = fused_output.get("elements", [])
    
    for correction_key, correction in doc_corrections:
        field_id = correction["field_id"]
        
        # Find and update element
        for element in elements:
            if element.get("id") == field_id:
                if correction["correction_type"] == "edit":
                    element["content"] = correction["corrected_value"]
                    element["human_corrected"] = True
                elif correction["correction_type"] == "confirm":
                    element["human_verified"] = True
                elif correction["correction_type"] == "reject":
                    element["rejected"] = True
                
                corrections_store[correction_key]["applied"] = True
                applied_count += 1
                break
    
    return {
        "message": f"Applied {applied_count} corrections",
        "applied_count": applied_count
    }


@router.get("/{document_id}/confidence-heatmap")
async def get_confidence_heatmap(
    document_id: str,
    page: int = Query(0, ge=0, description="Page number")
):
    """
    Get confidence heatmap data for visualization.
    
    Returns regions with their confidence scores for overlay display.
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
    fused_output = results.get("fused_output", {})
    elements = fused_output.get("elements", [])
    
    # Filter by page
    page_elements = [
        e for e in elements
        if e.get("page_number", 0) == page
    ]
    
    heatmap_data = []
    
    for element in page_elements:
        confidences = element.get("confidences", {})
        combined = (
            confidences.get("vision", 0) * 0.3 +
            confidences.get("ocr", 0) * 0.3 +
            confidences.get("layout", 0) * 0.4
        )
        
        bbox = element.get("bbox", [0, 0, 0, 0])
        
        heatmap_data.append({
            "id": element.get("id", ""),
            "type": element.get("type", "unknown"),
            "x1": bbox[0] if len(bbox) > 0 else 0,
            "y1": bbox[1] if len(bbox) > 1 else 0,
            "x2": bbox[2] if len(bbox) > 2 else 0,
            "y2": bbox[3] if len(bbox) > 3 else 0,
            "confidence": combined,
            "needs_review": combined < 0.7
        })
    
    return {
        "document_id": document_id,
        "page": page,
        "regions": heatmap_data
    }
