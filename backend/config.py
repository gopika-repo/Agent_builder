"""
Configuration settings for the Document Intelligence Platform
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Optional
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # LLM Configuration
    groq_api_key: Optional[str] = Field(default=None, env="GROQ_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    llm_provider: str = Field(default="groq", env="LLM_PROVIDER")
    llm_model: str = Field(default="llama-3.3-70b-versatile", env="LLM_MODEL")
    
    # Qdrant Configuration
    qdrant_host: str = Field(default="localhost", env="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, env="QDRANT_PORT")
    qdrant_api_key: Optional[str] = Field(default=None, env="QDRANT_API_KEY")
    
    # Embedding Models
    text_embedding_model: str = Field(
        default="all-MiniLM-L6-v2", 
        env="TEXT_EMBEDDING_MODEL"
    )
    image_embedding_model: str = Field(
        default="openai/clip-vit-base-patch32",
        env="IMAGE_EMBEDDING_MODEL"
    )
    
    # Computer Vision (YOLO) Configuration
    yolo_model_path: str = Field(
        default="yolov8n.pt",
        env="YOLO_MODEL_PATH"
    )
    yolo_confidence_threshold: float = Field(
        default=0.5,
        env="YOLO_CONFIDENCE_THRESHOLD"
    )
    yolo_iou_threshold: float = Field(
        default=0.45,
        env="YOLO_IOU_THRESHOLD"
    )
    
    # OCR Configuration
    ocr_engine: str = Field(
        default="hybrid",
        env="OCR_ENGINE"  # Options: tesseract, easyocr, hybrid
    )
    ocr_languages: str = Field(
        default="en",
        env="OCR_LANGUAGES"  # Comma-separated language codes
    )
    
    # Document Processing
    max_file_size_mb: int = Field(default=50, env="MAX_FILE_SIZE_MB")
    supported_formats: str = Field(
        default="pdf,png,jpg,jpeg,tiff",
        env="SUPPORTED_FORMATS"
    )
    
    # Confidence Thresholds
    low_confidence_threshold: float = Field(
        default=0.6,
        env="LOW_CONFIDENCE_THRESHOLD"
    )
    human_review_threshold: float = Field(
        default=0.7,
        env="HUMAN_REVIEW_THRESHOLD"
    )
    
    # Server Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    debug: bool = Field(default=True, env="DEBUG")
    
    # CORS
    cors_origins: str = Field(
        default="http://localhost:5173,http://localhost:3000",
        env="CORS_ORIGINS"
    )
    
    # Storage
    upload_dir: str = Field(default="./uploads", env="UPLOAD_DIR")
    processed_dir: str = Field(default="./processed", env="PROCESSED_DIR")
    
    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.cors_origins.split(",")]
    
    @property
    def supported_formats_list(self) -> List[str]:
        return [fmt.strip() for fmt in self.supported_formats.split(",")]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Collection names for Qdrant
QDRANT_COLLECTIONS = {
    "text": "document_text_chunks",
    "tables": "document_tables",
    "images": "document_images"
}

# YOLO model configuration (uses settings when available)
def get_yolo_config():
    """Get YOLO configuration from settings"""
    settings = get_settings()
    return {
        "model_path": settings.yolo_model_path,
        "confidence_threshold": settings.yolo_confidence_threshold,
        "iou_threshold": settings.yolo_iou_threshold,
        "classes": [
            "table",
            "figure", 
            "chart",
            "signature",
            "header",
            "footer",
            "paragraph",
            "list",
            "title"
        ]
    }

# Default YOLO config for backwards compatibility
YOLO_CONFIG = {
    "model_path": "yolov8n.pt",
    "confidence_threshold": 0.5,
    "iou_threshold": 0.45,
    "classes": [
        "table",
        "figure", 
        "chart",
        "signature",
        "header",
        "footer",
        "paragraph",
        "list",
        "title"
    ]
}

# Agent configuration
AGENT_CONFIG = {
    "max_retries": 3,
    "timeout_seconds": 120,
    "enable_recovery": True
}
