"""
Qdrant Vector Store Management

Manages three collections for multi-modal document storage:
- text_chunks: Text embeddings
- tables: Table embeddings
- images: Image/figure embeddings
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue
)

from ..config import get_settings, QDRANT_COLLECTIONS

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A search result from vector store"""
    id: str
    score: float
    payload: Dict[str, Any]
    collection: str


class QdrantVectorStore:
    """
    Qdrant vector database manager for multi-modal document storage.
    
    Manages three separate collections:
    - text_chunks: Text content embeddings
    - tables: Table data embeddings
    - images: Image/figure embeddings
    """
    
    # Embedding dimensions for different models
    DIMENSIONS = {
        "text": 384,      # all-MiniLM-L6-v2
        "image": 512,     # CLIP ViT-B/32
        "table": 384      # Same as text for now
    }
    
    def __init__(
        self,
        host: str = None,
        port: int = None,
        api_key: str = None
    ):
        """
        Initialize Qdrant vector store.
        
        Args:
            host: Qdrant host
            port: Qdrant port
            api_key: Optional API key
        """
        settings = get_settings()
        
        self.host = host or settings.qdrant_host
        self.port = port or settings.qdrant_port
        self.api_key = api_key or settings.qdrant_api_key
        
        self.collections = QDRANT_COLLECTIONS
        
        self._connect()
        self._ensure_collections()
    
    def _connect(self):
        """Connect to Qdrant"""
        try:
            if self.api_key:
                self.client = QdrantClient(
                    host=self.host,
                    port=self.port,
                    api_key=self.api_key
                )
            else:
                self.client = QdrantClient(
                    host=self.host,
                    port=self.port
                )
            
            logger.info(f"Connected to Qdrant at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            # Use in-memory Qdrant for development
            self.client = QdrantClient(":memory:")
            logger.warning("Using in-memory Qdrant (data will not persist)")
    
    def _ensure_collections(self):
        """Ensure all required collections exist"""
        for collection_type, collection_name in self.collections.items():
            try:
                # Check if collection exists
                collections = self.client.get_collections().collections
                exists = any(c.name == collection_name for c in collections)
                
                if not exists:
                    dimension = self.DIMENSIONS.get(collection_type, 384)
                    
                    self.client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(
                            size=dimension,
                            distance=Distance.COSINE
                        )
                    )
                    
                    logger.info(f"Created collection: {collection_name} (dim={dimension})")
                else:
                    logger.debug(f"Collection exists: {collection_name}")
                    
            except Exception as e:
                logger.error(f"Failed to ensure collection {collection_name}: {e}")
    
    def add_text_chunks(
        self,
        document_id: str,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> int:
        """
        Add text chunks to the text collection.
        
        Args:
            document_id: Document ID
            chunks: List of chunk data with text and metadata
            embeddings: Corresponding embeddings
            
        Returns:
            Number of chunks added
        """
        collection_name = self.collections["text"]
        points = []
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = str(uuid.uuid4())
            
            points.append(PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "document_id": document_id,
                    "chunk_id": i,
                    "text": chunk.get("text", ""),
                    "page_number": chunk.get("page_number", 0),
                    "x1": chunk.get("x1", 0),
                    "y1": chunk.get("y1", 0),
                    "x2": chunk.get("x2", 0),
                    "y2": chunk.get("y2", 0),
                    "type": "text"
                }
            ))
        
        self.client.upsert(
            collection_name=collection_name,
            points=points
        )
        
        logger.info(f"Added {len(points)} text chunks for document {document_id}")
        return len(points)
    
    def add_tables(
        self,
        document_id: str,
        tables: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> int:
        """
        Add tables to the tables collection.
        
        Args:
            document_id: Document ID
            tables: List of table data
            embeddings: Corresponding embeddings
            
        Returns:
            Number of tables added
        """
        collection_name = self.collections["tables"]
        points = []
        
        for i, (table, embedding) in enumerate(zip(tables, embeddings)):
            point_id = str(uuid.uuid4())
            
            points.append(PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "document_id": document_id,
                    "table_id": table.get("id", f"table_{i}"),
                    "page_number": table.get("page_number", 0),
                    "headers": table.get("headers", []),
                    "num_rows": table.get("num_rows", 0),
                    "num_cols": table.get("num_cols", 0),
                    "content_preview": str(table.get("rows", [])[:3]),
                    "type": "table"
                }
            ))
        
        self.client.upsert(
            collection_name=collection_name,
            points=points
        )
        
        logger.info(f"Added {len(points)} tables for document {document_id}")
        return len(points)
    
    def add_images(
        self,
        document_id: str,
        images: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> int:
        """
        Add images to the images collection.
        
        Args:
            document_id: Document ID
            images: List of image metadata
            embeddings: Corresponding CLIP embeddings
            
        Returns:
            Number of images added
        """
        collection_name = self.collections["images"]
        points = []
        
        for i, (image, embedding) in enumerate(zip(images, embeddings)):
            point_id = str(uuid.uuid4())
            
            points.append(PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "document_id": document_id,
                    "image_id": image.get("id", f"image_{i}"),
                    "page_number": image.get("page_number", 0),
                    "x1": image.get("x1", 0),
                    "y1": image.get("y1", 0),
                    "x2": image.get("x2", 0),
                    "y2": image.get("y2", 0),
                    "label": image.get("label", "figure"),
                    "caption": image.get("caption", ""),
                    "type": "image"
                }
            ))
        
        self.client.upsert(
            collection_name=collection_name,
            points=points
        )
        
        logger.info(f"Added {len(points)} images for document {document_id}")
        return len(points)
    
    def search_text(
        self,
        query_embedding: List[float],
        document_id: Optional[str] = None,
        limit: int = 10
    ) -> List[SearchResult]:
        """
        Search text chunks.
        
        Args:
            query_embedding: Query embedding
            document_id: Optional filter by document
            limit: Maximum results
            
        Returns:
            List of search results
        """
        collection_name = self.collections["text"]
        
        query_filter = None
        if document_id:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            )
        
        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            query_filter=query_filter,
            limit=limit
        )
        
        return [
            SearchResult(
                id=str(r.id),
                score=r.score,
                payload=r.payload,
                collection="text"
            )
            for r in results
        ]
    
    def search_tables(
        self,
        query_embedding: List[float],
        document_id: Optional[str] = None,
        limit: int = 5
    ) -> List[SearchResult]:
        """Search tables collection"""
        collection_name = self.collections["tables"]
        
        query_filter = None
        if document_id:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            )
        
        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            query_filter=query_filter,
            limit=limit
        )
        
        return [
            SearchResult(
                id=str(r.id),
                score=r.score,
                payload=r.payload,
                collection="tables"
            )
            for r in results
        ]
    
    def search_images(
        self,
        query_embedding: List[float],
        document_id: Optional[str] = None,
        limit: int = 5
    ) -> List[SearchResult]:
        """Search images collection"""
        collection_name = self.collections["images"]
        
        query_filter = None
        if document_id:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            )
        
        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            query_filter=query_filter,
            limit=limit
        )
        
        return [
            SearchResult(
                id=str(r.id),
                score=r.score,
                payload=r.payload,
                collection="images"
            )
            for r in results
        ]
    
    def delete_document(self, document_id: str):
        """Delete all data for a document"""
        for collection_name in self.collections.values():
            try:
                self.client.delete(
                    collection_name=collection_name,
                    points_selector=models.FilterSelector(
                        filter=Filter(
                            must=[
                                FieldCondition(
                                    key="document_id",
                                    match=MatchValue(value=document_id)
                                )
                            ]
                        )
                    )
                )
            except Exception as e:
                logger.error(f"Failed to delete from {collection_name}: {e}")
        
        logger.info(f"Deleted all data for document {document_id}")
    
    def get_document_stats(self, document_id: str) -> Dict[str, int]:
        """Get statistics for a document"""
        stats = {}
        
        for collection_type, collection_name in self.collections.items():
            try:
                count = self.client.count(
                    collection_name=collection_name,
                    count_filter=Filter(
                        must=[
                            FieldCondition(
                                key="document_id",
                                match=MatchValue(value=document_id)
                            )
                        ]
                    )
                )
                stats[collection_type] = count.count
            except:
                stats[collection_type] = 0
        
        return stats
