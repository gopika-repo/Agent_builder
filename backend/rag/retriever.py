"""
Multi-Modal Retriever

Cross-modal retrieval supporting:
- Text to text
- Text to image
- Image to text
- Fusion scoring
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from .vector_store import QdrantVectorStore, SearchResult
from .embeddings import TextEmbedder, ImageEmbedder, TableEmbedder

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A retrieval result with context"""
    content: str
    content_type: str  # text, table, image
    score: float
    page_number: int
    bbox: Optional[Tuple[float, float, float, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "content_type": self.content_type,
            "score": self.score,
            "page_number": self.page_number,
            "bbox": self.bbox,
            "metadata": self.metadata
        }


class MultiModalRetriever:
    """
    Multi-modal retrieval system.
    
    Supports:
    - Text queries → text results
    - Text queries → image results (via CLIP)
    - Text queries → table results
    - Cross-modal fusion scoring
    """
    
    def __init__(
        self,
        vector_store: QdrantVectorStore = None,
        text_embedder: TextEmbedder = None,
        image_embedder: ImageEmbedder = None,
        table_embedder: TableEmbedder = None
    ):
        """
        Initialize multi-modal retriever.
        
        Args:
            vector_store: Qdrant vector store
            text_embedder: Text embedding model
            image_embedder: Image embedding model
            table_embedder: Table embedding model
        """
        self.vector_store = vector_store or QdrantVectorStore()
        self.text_embedder = text_embedder or TextEmbedder()
        self.image_embedder = image_embedder or ImageEmbedder()
        self.table_embedder = table_embedder or TableEmbedder(self.text_embedder)
    
    def retrieve(
        self,
        query: str,
        document_id: Optional[str] = None,
        modalities: List[str] = None,
        limit: int = 10,
        fusion_method: str = "rrf"
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant content across modalities.
        
        Args:
            query: Natural language query
            document_id: Optional filter to specific document
            modalities: Which modalities to search ['text', 'tables', 'images']
            limit: Maximum results per modality
            fusion_method: 'rrf' (reciprocal rank fusion) or 'score'
            
        Returns:
            List of retrieval results ranked by relevance
        """
        modalities = modalities or ["text", "tables", "images"]
        all_results = []
        
        # Generate query embeddings
        text_embedding = self.text_embedder.embed(query)
        
        # For image search, use CLIP text embedding
        if "images" in modalities:
            image_query_embedding = self.image_embedder.embed_text(query)
        
        # Search text
        if "text" in modalities:
            text_results = self.vector_store.search_text(
                query_embedding=text_embedding,
                document_id=document_id,
                limit=limit
            )
            
            for r in text_results:
                all_results.append(RetrievalResult(
                    content=r.payload.get("text", ""),
                    content_type="text",
                    score=r.score,
                    page_number=r.payload.get("page_number", 0),
                    bbox=(
                        r.payload.get("x1", 0),
                        r.payload.get("y1", 0),
                        r.payload.get("x2", 0),
                        r.payload.get("y2", 0)
                    ),
                    metadata=r.payload
                ))
        
        # Search tables
        if "tables" in modalities:
            table_results = self.vector_store.search_tables(
                query_embedding=text_embedding,
                document_id=document_id,
                limit=limit
            )
            
            for r in table_results:
                content = f"Table with headers: {r.payload.get('headers', [])}"
                all_results.append(RetrievalResult(
                    content=content,
                    content_type="table",
                    score=r.score,
                    page_number=r.payload.get("page_number", 0),
                    metadata=r.payload
                ))
        
        # Search images
        if "images" in modalities:
            image_results = self.vector_store.search_images(
                query_embedding=image_query_embedding,
                document_id=document_id,
                limit=limit
            )
            
            for r in image_results:
                content = f"{r.payload.get('label', 'Image')}: {r.payload.get('caption', 'No caption')}"
                all_results.append(RetrievalResult(
                    content=content,
                    content_type="image",
                    score=r.score,
                    page_number=r.payload.get("page_number", 0),
                    bbox=(
                        r.payload.get("x1", 0),
                        r.payload.get("y1", 0),
                        r.payload.get("x2", 0),
                        r.payload.get("y2", 0)
                    ),
                    metadata=r.payload
                ))
        
        # Apply fusion and ranking
        if fusion_method == "rrf":
            ranked_results = self._reciprocal_rank_fusion(all_results)
        else:
            ranked_results = sorted(all_results, key=lambda x: x.score, reverse=True)
        
        return ranked_results[:limit]
    
    def retrieve_text(
        self,
        query: str,
        document_id: Optional[str] = None,
        limit: int = 10
    ) -> List[RetrievalResult]:
        """Retrieve text chunks only"""
        return self.retrieve(
            query=query,
            document_id=document_id,
            modalities=["text"],
            limit=limit
        )
    
    def retrieve_tables(
        self,
        query: str,
        document_id: Optional[str] = None,
        limit: int = 5
    ) -> List[RetrievalResult]:
        """Retrieve tables only"""
        return self.retrieve(
            query=query,
            document_id=document_id,
            modalities=["tables"],
            limit=limit
        )
    
    def retrieve_images(
        self,
        query: str,
        document_id: Optional[str] = None,
        limit: int = 5
    ) -> List[RetrievalResult]:
        """Retrieve images only"""
        return self.retrieve(
            query=query,
            document_id=document_id,
            modalities=["images"],
            limit=limit
        )
    
    def _reciprocal_rank_fusion(
        self,
        results: List[RetrievalResult],
        k: int = 60
    ) -> List[RetrievalResult]:
        """
        Apply Reciprocal Rank Fusion to combine results.
        
        RRF score = sum(1 / (k + rank_i)) for each result set
        
        Args:
            results: All results to fuse
            k: RRF constant (higher = more weight to lower ranks)
            
        Returns:
            Re-ranked results
        """
        # Group by modality
        by_modality = {}
        for r in results:
            if r.content_type not in by_modality:
                by_modality[r.content_type] = []
            by_modality[r.content_type].append(r)
        
        # Sort each modality by score
        for modality in by_modality:
            by_modality[modality].sort(key=lambda x: x.score, reverse=True)
        
        # Calculate RRF scores
        rrf_scores = {}
        
        for modality, modality_results in by_modality.items():
            for rank, result in enumerate(modality_results):
                key = id(result)
                if key not in rrf_scores:
                    rrf_scores[key] = {"result": result, "score": 0}
                rrf_scores[key]["score"] += 1.0 / (k + rank + 1)
        
        # Sort by RRF score
        ranked = sorted(
            rrf_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )
        
        # Update scores and return
        output = []
        for item in ranked:
            result = item["result"]
            result.score = item["score"]
            output.append(result)
        
        return output
    
    def get_context_for_query(
        self,
        query: str,
        document_id: str,
        max_context_length: int = 4000
    ) -> str:
        """
        Get concatenated context for a query.
        
        Args:
            query: User query
            document_id: Document to query
            max_context_length: Maximum context length in characters
            
        Returns:
            Formatted context string
        """
        results = self.retrieve(
            query=query,
            document_id=document_id,
            limit=10
        )
        
        context_parts = []
        current_length = 0
        
        for result in results:
            if result.content_type == "text":
                context = f"[Text, Page {result.page_number}]: {result.content}"
            elif result.content_type == "table":
                context = f"[Table, Page {result.page_number}]: {result.content}"
            else:
                context = f"[Figure, Page {result.page_number}]: {result.content}"
            
            if current_length + len(context) > max_context_length:
                break
            
            context_parts.append(context)
            current_length += len(context) + 2  # +2 for newlines
        
        return "\n\n".join(context_parts)
