"""
Document Routes

Endpoints for document upload, processing, and result retrieval.
"""

import logging
from typing import Optional, List
from pathlib import Path
import uuid
import aiofiles
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ...config import get_settings
from ...agents.workflow import run_document_pipeline
from ...agents.state import ProcessingStatus
from ...rag.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory document store (use database in production)
documents_store = {}


class DocumentResponse(BaseModel):
    """Document response model"""
    document_id: str
    filename: str
    status: str
    created_at: str
    page_count: Optional[int] = None
    message: str


class DocumentStatus(BaseModel):
    """Document status model"""
    document_id: str
    status: str
    current_agent: Optional[str] = None
    progress: float
    errors: List[str] = []


class ProcessingResult(BaseModel):
    """Processing result model"""
    document_id: str
    status: str
    vision_results: Optional[dict] = None
    ocr_results: Optional[dict] = None
    layout_graph: Optional[dict] = None
    text_analysis: Optional[dict] = None
    fused_output: Optional[dict] = None
    confidence_scores: Optional[dict] = None
    processing_time_ms: float


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload a document for processing.
    
    Accepts PDF, PNG, JPG, JPEG, TIFF files.
    Starts async processing pipeline.
    """
    settings = get_settings()
    
    # Validate file type
    allowed_extensions = settings.supported_formats_list
    file_ext = Path(file.filename).suffix.lower().lstrip(".")
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {allowed_extensions}"
        )
    
    # Check file size
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    max_size = settings.max_file_size_mb * 1024 * 1024
    if file_size > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {settings.max_file_size_mb}MB"
        )
    
    # Generate document ID
    document_id = str(uuid.uuid4())
    
    # Save file
    upload_path = Path(settings.upload_dir) / f"{document_id}.{file_ext}"
    
    async with aiofiles.open(upload_path, "wb") as f:
        content = await file.read()
        await f.write(content)
    
    # Store document metadata
    documents_store[document_id] = {
        "document_id": document_id,
        "filename": file.filename,
        "file_type": file_ext,
        "file_path": str(upload_path),
        "status": ProcessingStatus.PENDING.value,
        "created_at": datetime.utcnow().isoformat(),
        "current_agent": None,
        "results": None,
        "errors": []
    }
    
    # Start background processing
    background_tasks.add_task(process_document, document_id)
    
    logger.info(f"Document uploaded: {document_id} ({file.filename})")
    
    return DocumentResponse(
        document_id=document_id,
        filename=file.filename,
        status=ProcessingStatus.PENDING.value,
        created_at=documents_store[document_id]["created_at"],
        message="Document uploaded successfully. Processing started."
    )


async def process_document(document_id: str):
    """Background task to process document through pipeline"""
    try:
        doc = documents_store.get(document_id)
        if not doc:
            logger.error(f"Document not found: {document_id}")
            return
        
        file_path = doc["file_path"]
        file_type = doc["file_type"]
        filename = doc["filename"]
        
        # Load pages from file
        pages = await load_document_pages(file_path, file_type)
        
        doc["page_count"] = len(pages)
        doc["status"] = ProcessingStatus.VISION_PROCESSING.value
        
        # Run pipeline
        result = await run_document_pipeline(
            document_id=document_id,
            filename=filename,
            file_type=file_type,
            pages=pages,
            progress_callback=lambda p: update_progress(document_id, p)
        )
        
        # Store results
        doc["results"] = result
        doc["status"] = result.get("status", ProcessingStatus.COMPLETED.value)
        doc["errors"] = result.get("errors", [])
        
        # Index in RAG if successful
        if doc["status"] == ProcessingStatus.COMPLETED.value:
            try:
                rag = RAGPipeline()
                rag.index_document(document_id, result.get("fused_output", {}))
            except Exception as e:
                logger.error(f"RAG indexing failed: {e}")
        
        logger.info(f"Document processing completed: {document_id}")
        
    except Exception as e:
        logger.error(f"Document processing failed: {document_id} - {e}")
        if document_id in documents_store:
            documents_store[document_id]["status"] = ProcessingStatus.FAILED.value
            documents_store[document_id]["errors"].append(str(e))


async def load_document_pages(file_path: str, file_type: str) -> list:
    """Load document pages from file"""
    from ...cv.preprocessor import ImagePreprocessor
    
    preprocessor = ImagePreprocessor()
    
    if file_type == "pdf":
        # Convert PDF to images
        pages = preprocessor.pdf_to_images(file_path)
        return [
            {
                "page_number": i,
                "image": page,
                "width": page.shape[1],
                "height": page.shape[0]
            }
            for i, page in enumerate(pages)
        ]
    else:
        # Load single image
        import cv2
        image = cv2.imread(file_path)
        return [{
            "page_number": 0,
            "image": image,
            "width": image.shape[1],
            "height": image.shape[0]
        }]


def update_progress(document_id: str, progress_data: dict):
    """Update document processing progress"""
    if document_id in documents_store:
        documents_store[document_id]["current_agent"] = progress_data.get("current_agent")
        documents_store[document_id]["status"] = progress_data.get("status")


@router.get("/{document_id}", response_model=DocumentStatus)
async def get_document_status(document_id: str):
    """Get document processing status"""
    doc = documents_store.get(document_id)
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Calculate progress
    status = doc["status"]
    progress_map = {
        ProcessingStatus.PENDING.value: 0.0,
        ProcessingStatus.VISION_PROCESSING.value: 0.15,
        ProcessingStatus.OCR_PROCESSING.value: 0.30,
        ProcessingStatus.LAYOUT_ANALYSIS.value: 0.45,
        ProcessingStatus.TEXT_REASONING.value: 0.60,
        ProcessingStatus.FUSION.value: 0.75,
        ProcessingStatus.VALIDATION.value: 0.90,
        ProcessingStatus.COMPLETED.value: 1.0,
        ProcessingStatus.FAILED.value: 0.0
    }
    
    return DocumentStatus(
        document_id=document_id,
        status=status,
        current_agent=doc.get("current_agent"),
        progress=progress_map.get(status, 0.0),
        errors=doc.get("errors", [])
    )


@router.get("/{document_id}/results")
async def get_document_results(document_id: str):
    """Get full processing results for a document"""
    doc = documents_store.get(document_id)
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if doc["status"] != ProcessingStatus.COMPLETED.value:
        raise HTTPException(
            status_code=400,
            detail=f"Document processing not complete. Status: {doc['status']}"
        )
    
    results = doc.get("results", {})
    
    return {
        "document_id": document_id,
        "filename": doc["filename"],
        "status": doc["status"],
        "page_count": doc.get("page_count", 0),
        "vision_results": results.get("vision_results", {}),
        "ocr_results": results.get("ocr_results", {}),
        "layout_graph": results.get("layout_graph", {}),
        "text_analysis": results.get("text_analysis", {}),
        "fused_output": results.get("fused_output", {}),
        "confidence_scores": results.get("confidence_scores", {}),
        "processing_time_ms": results.get("total_processing_time_ms", 0)
    }


@router.get("/{document_id}/summary")
async def get_document_summary(document_id: str):
    """Get document summary with key information"""
    doc = documents_store.get(document_id)
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if doc["status"] != ProcessingStatus.COMPLETED.value:
        raise HTTPException(
            status_code=400,
            detail=f"Document processing not complete. Status: {doc['status']}"
        )
    
    results = doc.get("results", {})
    text_analysis = results.get("text_analysis", {})
    fused_output = results.get("fused_output", {})
    confidence = results.get("confidence_scores", {})
    
    return {
        "document_id": document_id,
        "filename": doc["filename"],
        "document_type": text_analysis.get("document_type", "unknown"),
        "summary": text_analysis.get("summary", ""),
        "key_points": text_analysis.get("key_points", []),
        "entities": text_analysis.get("entities", []),
        "topics": text_analysis.get("topics", []),
        "table_count": len(fused_output.get("tables", [])),
        "figure_count": len([
            e for e in fused_output.get("elements", [])
            if e.get("type") in ["figure", "image", "chart"]
        ]),
        "overall_confidence": confidence.get("overall_confidence", 0),
        "needs_review": len(confidence.get("items_needing_review", [])) > 0
    }


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its data"""
    doc = documents_store.get(document_id)
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Delete file
    file_path = doc.get("file_path")
    if file_path and Path(file_path).exists():
        Path(file_path).unlink()
    
    # Delete from vector store
    try:
        from ...rag.vector_store import QdrantVectorStore
        store = QdrantVectorStore()
        store.delete_document(document_id)
    except Exception as e:
        logger.warning(f"Failed to delete from vector store: {e}")
    
    # Remove from memory
    del documents_store[document_id]
    
    return {"message": f"Document {document_id} deleted successfully"}


@router.get("/")
async def list_documents(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(20, ge=1, le=100)
):
    """List all documents"""
    docs = list(documents_store.values())
    
    if status:
        docs = [d for d in docs if d["status"] == status]
    
    docs = sorted(docs, key=lambda x: x["created_at"], reverse=True)[:limit]
    
    return {
        "total": len(documents_store),
        "documents": [
            {
                "document_id": d["document_id"],
                "filename": d["filename"],
                "status": d["status"],
                "created_at": d["created_at"],
                "page_count": d.get("page_count")
            }
            for d in docs
        ]
    }
