"""
FastAPI Application Entry Point

Main application setup with CORS, routes, and lifecycle events.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ..config import get_settings
from .routes import documents, chat, review, advanced
from .websocket import router as websocket_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting Document Intelligence Platform API")
    settings = get_settings()
    
    # Ensure directories exist
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.processed_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Upload directory: {settings.upload_dir}")
    logger.info(f"Processed directory: {settings.processed_dir}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Document Intelligence Platform API")


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    settings = get_settings()
    
    app = FastAPI(
        title="Multi-Modal Document Intelligence Platform",
        description="""
        A production-ready document intelligence system featuring:
        
        - 🔍 **6-Agent LangGraph Pipeline**: Vision → OCR → Layout → Reasoning → Fusion → Validation
        - 📚 **Multi-Modal RAG**: Text, Table, and Image retrieval with cross-modal search
        - 🎯 **Confidence Scoring**: Field-level confidence with human review flagging
        - 💡 **ELI5 vs Expert Mode**: Two explanation levels for any query
        - 🖼️ **Visual Review**: Bounding box overlay and confidence heatmaps
        """,
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(
        documents.router,
        prefix="/api/documents",
        tags=["Documents"]
    )
    
    app.include_router(
        chat.router,
        prefix="/api/chat",
        tags=["Chat & Query"]
    )
    
    app.include_router(
        review.router,
        prefix="/api/review",
        tags=["Human Review"]
    )
    
    app.include_router(
        websocket_router,
        prefix="/ws",
        tags=["WebSocket"]
    )
    
    app.include_router(
        advanced.router,
        tags=["Advanced Features"]
    )
    
    # Health check endpoint
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "version": "1.0.0",
            "service": "document-intelligence-api"
        }
    
    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root():
        """API root endpoint"""
        return {
            "message": "Multi-Modal Document Intelligence Platform API",
            "docs": "/docs",
            "health": "/health"
        }
    
    return app


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    uvicorn.run(
        "backend.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )
