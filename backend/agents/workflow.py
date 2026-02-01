"""
LangGraph Workflow Definition

Orchestrates the 6-agent document processing pipeline with:
- Explicit state management
- Defined transitions
- Error recovery paths
"""

import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime
import time

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import DocumentState, ProcessingStatus, create_initial_state, PageImage
from .vision_agent import VisionAgent, vision_agent_node
from .ocr_agent import OCRAgent, ocr_agent_node
from .layout_agent import LayoutAgent, layout_agent_node
from .text_reasoning_agent import TextReasoningAgent, text_reasoning_agent_node
from .fusion_agent import FusionAgent, fusion_agent_node
from .validation_agent import ValidationAgent, validation_agent_node

logger = logging.getLogger(__name__)


def should_continue_after_vision(state: DocumentState) -> str:
    """Determine next step after vision agent"""
    if state.get("errors"):
        # Check severity of errors
        if any("critical" in e.lower() for e in state.get("errors", [])):
            return "error_handler"
    
    vision_results = state.get("vision_results", {})
    if not vision_results.get("detections"):
        logger.warning("No detections from vision agent, continuing anyway")
    
    return "ocr"


def should_continue_after_ocr(state: DocumentState) -> str:
    """Determine next step after OCR agent"""
    if state.get("errors"):
        if any("critical" in e.lower() for e in state.get("errors", [])):
            return "error_handler"
    
    ocr_results = state.get("ocr_results", {})
    if not ocr_results.get("full_text"):
        logger.warning("No text extracted from OCR, continuing to layout")
    
    return "layout"


def should_continue_after_layout(state: DocumentState) -> str:
    """Determine next step after layout agent"""
    if state.get("errors"):
        if any("critical" in e.lower() for e in state.get("errors", [])):
            return "error_handler"
    
    return "reasoning"


def should_continue_after_reasoning(state: DocumentState) -> str:
    """Determine next step after text reasoning agent"""
    if state.get("errors"):
        if any("critical" in e.lower() for e in state.get("errors", [])):
            return "error_handler"
    
    return "fusion"


def should_continue_after_fusion(state: DocumentState) -> str:
    """Determine next step after fusion agent"""
    if state.get("errors"):
        if any("critical" in e.lower() for e in state.get("errors", [])):
            return "error_handler"
    
    return "validation"


def should_continue_after_validation(state: DocumentState) -> str:
    """Determine next step after validation agent"""
    # Validation is the final step in normal flow
    return END


def error_handler_node(state: DocumentState) -> DocumentState:
    """Handle errors in the pipeline"""
    logger.error(f"Error handler invoked for document {state.get('document_id')}")
    
    state["status"] = ProcessingStatus.FAILED.value
    state["processing_end_time"] = datetime.utcnow().isoformat()
    
    # Calculate total processing time
    if state.get("processing_start_time"):
        try:
            start = datetime.fromisoformat(state["processing_start_time"])
            end = datetime.fromisoformat(state["processing_end_time"])
            state["total_processing_time_ms"] = (end - start).total_seconds() * 1000
        except:
            pass
    
    return state


def create_workflow() -> StateGraph:
    """
    Create the LangGraph workflow for document processing.
    
    Flow:
    vision -> ocr -> layout -> reasoning -> fusion -> validation -> END
    
    With error recovery paths at each step.
    
    Returns:
        Compiled StateGraph workflow
    """
    # Create the graph
    workflow = StateGraph(DocumentState)
    
    # Add nodes
    workflow.add_node("vision", vision_agent_node)
    workflow.add_node("ocr", ocr_agent_node)
    workflow.add_node("layout", layout_agent_node)
    workflow.add_node("reasoning", text_reasoning_agent_node)
    workflow.add_node("fusion", fusion_agent_node)
    workflow.add_node("validation", validation_agent_node)
    workflow.add_node("error_handler", error_handler_node)
    
    # Set entry point
    workflow.set_entry_point("vision")
    
    # Add conditional edges with error recovery
    workflow.add_conditional_edges(
        "vision",
        should_continue_after_vision,
        {
            "ocr": "ocr",
            "error_handler": "error_handler"
        }
    )
    
    workflow.add_conditional_edges(
        "ocr",
        should_continue_after_ocr,
        {
            "layout": "layout",
            "error_handler": "error_handler"
        }
    )
    
    workflow.add_conditional_edges(
        "layout",
        should_continue_after_layout,
        {
            "reasoning": "reasoning",
            "error_handler": "error_handler"
        }
    )
    
    workflow.add_conditional_edges(
        "reasoning",
        should_continue_after_reasoning,
        {
            "fusion": "fusion",
            "error_handler": "error_handler"
        }
    )
    
    workflow.add_conditional_edges(
        "fusion",
        should_continue_after_fusion,
        {
            "validation": "validation",
            "error_handler": "error_handler"
        }
    )
    
    workflow.add_conditional_edges(
        "validation",
        should_continue_after_validation,
        {
            END: END
        }
    )
    
    # Error handler always ends
    workflow.add_edge("error_handler", END)
    
    return workflow


def compile_workflow(checkpointer: Optional[MemorySaver] = None):
    """
    Compile the workflow with optional checkpointing.
    
    Args:
        checkpointer: Optional memory saver for state persistence
        
    Returns:
        Compiled workflow
    """
    workflow = create_workflow()
    
    if checkpointer:
        return workflow.compile(checkpointer=checkpointer)
    
    return workflow.compile()


async def run_document_pipeline(
    document_id: str,
    filename: str,
    file_type: str,
    pages: list,
    config: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable] = None
) -> DocumentState:
    """
    Run the complete document processing pipeline.
    
    Args:
        document_id: Unique document identifier
        filename: Original filename
        file_type: File type (pdf, png, etc.)
        pages: List of page images/data
        config: Optional configuration overrides
        progress_callback: Optional callback for progress updates
        
    Returns:
        Final DocumentState with all processing results
    """
    logger.info(f"Starting pipeline for document: {document_id}")
    
    start_time = time.time()
    
    # Convert pages to PageImage objects
    page_objects = []
    for i, page in enumerate(pages):
        if isinstance(page, dict):
            page_objects.append(PageImage(
                page_number=page.get("page_number", i),
                image=page.get("image"),
                width=page.get("width", 0),
                height=page.get("height", 0)
            ))
        else:
            # Assume it's a numpy array
            page_objects.append(PageImage(
                page_number=i,
                image=page,
                width=page.shape[1] if hasattr(page, 'shape') else 0,
                height=page.shape[0] if hasattr(page, 'shape') else 0
            ))
    
    # Create initial state
    initial_state = create_initial_state(
        document_id=document_id,
        filename=filename,
        file_type=file_type,
        pages=page_objects,
        config=config
    )
    
    initial_state["processing_start_time"] = datetime.utcnow().isoformat()
    
    # Compile workflow
    app = compile_workflow()
    
    # Run the workflow
    final_state = None
    
    try:
        # Execute workflow
        for event in app.stream(initial_state):
            # Get current node and state
            for node_name, node_state in event.items():
                logger.debug(f"Completed node: {node_name}")
                
                if progress_callback:
                    await progress_callback({
                        "document_id": document_id,
                        "current_agent": node_name,
                        "status": node_state.get("status", "processing")
                    })
                
                final_state = node_state
        
        # Set completion time
        if final_state:
            final_state["processing_end_time"] = datetime.utcnow().isoformat()
            final_state["total_processing_time_ms"] = (time.time() - start_time) * 1000
        
        logger.info(
            f"Pipeline completed for {document_id} "
            f"in {(time.time() - start_time) * 1000:.2f}ms"
        )
        
    except Exception as e:
        logger.error(f"Pipeline failed for {document_id}: {e}")
        
        if final_state:
            final_state["errors"].append(f"Pipeline error: {str(e)}")
            final_state["status"] = ProcessingStatus.FAILED.value
        else:
            final_state = initial_state
            final_state["errors"] = [f"Pipeline error: {str(e)}"]
            final_state["status"] = ProcessingStatus.FAILED.value
    
    return final_state


def run_document_pipeline_sync(
    document_id: str,
    filename: str,
    file_type: str,
    pages: list,
    config: Optional[Dict[str, Any]] = None
) -> DocumentState:
    """
    Synchronous version of run_document_pipeline.
    
    Args:
        document_id: Unique document identifier
        filename: Original filename
        file_type: File type (pdf, png, etc.)
        pages: List of page images/data
        config: Optional configuration overrides
        
    Returns:
        Final DocumentState with all processing results
    """
    import asyncio
    
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            run_document_pipeline(
                document_id=document_id,
                filename=filename,
                file_type=file_type,
                pages=pages,
                config=config
            )
        )
    finally:
        loop.close()
