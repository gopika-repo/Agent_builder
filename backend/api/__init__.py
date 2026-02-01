"""
FastAPI Backend API
Document Intelligence Platform REST API
"""

from .main import app, create_app
from .routes import documents, chat, review

__all__ = ["app", "create_app"]
