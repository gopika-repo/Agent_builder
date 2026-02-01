"""
WebSocket Handler

Real-time updates for document processing progress.
"""

import logging
from typing import Dict, Set
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)
router = APIRouter()


class ConnectionManager:
    """Manage WebSocket connections"""
    
    def __init__(self):
        # document_id -> set of websockets
        self.active_connections: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, document_id: str):
        """Accept and register a new connection"""
        await websocket.accept()
        
        if document_id not in self.active_connections:
            self.active_connections[document_id] = set()
        
        self.active_connections[document_id].add(websocket)
        logger.info(f"WebSocket connected for document: {document_id}")
    
    def disconnect(self, websocket: WebSocket, document_id: str):
        """Remove a connection"""
        if document_id in self.active_connections:
            self.active_connections[document_id].discard(websocket)
            
            if not self.active_connections[document_id]:
                del self.active_connections[document_id]
        
        logger.info(f"WebSocket disconnected for document: {document_id}")
    
    async def broadcast(self, document_id: str, message: dict):
        """Broadcast message to all connections for a document"""
        if document_id not in self.active_connections:
            return
        
        disconnected = set()
        
        for websocket in self.active_connections[document_id]:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Failed to send message: {e}")
                disconnected.add(websocket)
        
        # Clean up disconnected
        for ws in disconnected:
            self.active_connections[document_id].discard(ws)


# Global connection manager
manager = ConnectionManager()


@router.websocket("/{document_id}")
async def websocket_endpoint(websocket: WebSocket, document_id: str):
    """
    WebSocket endpoint for real-time processing updates.
    
    Clients receive:
    - Processing progress updates
    - Agent transitions
    - Completion/error notifications
    """
    await manager.connect(websocket, document_id)
    
    try:
        while True:
            # Wait for messages from client (heartbeat/commands)
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                
                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                
                elif message.get("type") == "status":
                    # Get current status
                    from .routes.documents import documents_store
                    
                    doc = documents_store.get(document_id)
                    if doc:
                        await websocket.send_json({
                            "type": "status",
                            "document_id": document_id,
                            "status": doc.get("status"),
                            "current_agent": doc.get("current_agent")
                        })
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Document not found"
                        })
                        
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON"
                })
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, document_id)


async def send_progress_update(
    document_id: str,
    status: str,
    current_agent: str = None,
    progress: float = 0.0,
    message: str = None
):
    """
    Send progress update to connected clients.
    
    Called by document processing pipeline.
    """
    await manager.broadcast(document_id, {
        "type": "progress",
        "document_id": document_id,
        "status": status,
        "current_agent": current_agent,
        "progress": progress,
        "message": message
    })


async def send_completion(
    document_id: str,
    success: bool,
    message: str = None
):
    """Send completion notification"""
    await manager.broadcast(document_id, {
        "type": "complete" if success else "error",
        "document_id": document_id,
        "success": success,
        "message": message
    })


async def send_confidence_update(
    document_id: str,
    field_id: str,
    confidence: float,
    needs_review: bool
):
    """Send live confidence update"""
    await manager.broadcast(document_id, {
        "type": "confidence",
        "document_id": document_id,
        "field_id": field_id,
        "confidence": confidence,
        "needs_review": needs_review
    })
