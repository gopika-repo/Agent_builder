"""
Advanced Multi-Modal Retriever with Cross-Modal Re-ranking

Features:
- Parallel retrieval from text, table, and image collections
- Cross-modal re-ranking using LLM reasoning
- Evidence grounding with source references
- Multi-document support
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

from .vector_store import QdrantVectorStore
from .embeddings import TextEmbedder, ImageEmbedder, TableEmbedder
from ..config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class GroundedEvidence:
    """Evidence with visual grounding information"""
    content: str
    content_type: str  # text, table, image
    score: float
    page_number: int
    document_id: str
    bbox: Optional[Tuple[float, float, float, float]] = None  # x1, y1, x2, y2
    source_id: str = ""
    reasoning: str = ""  # Why this evidence is relevant
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "content_type": self.content_type,
            "score": round(self.score, 4),
            "page_number": self.page_number,
            "document_id": self.document_id,
            "bbox": self.bbox,
            "source_id": self.source_id,
            "reasoning": self.reasoning
        }


@dataclass
class CrossModalResult:
    """Result of cross-modal retrieval with re-ranking"""
    query: str
    text_results: List[GroundedEvidence] = field(default_factory=list)
    table_results: List[GroundedEvidence] = field(default_factory=list)
    image_results: List[GroundedEvidence] = field(default_factory=list)
    reranked_results: List[GroundedEvidence] = field(default_factory=list)
    reasoning: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "text_results": [r.to_dict() for r in self.text_results],
            "table_results": [r.to_dict() for r in self.table_results],
            "image_results": [r.to_dict() for r in self.image_results],
            "reranked_results": [r.to_dict() for r in self.reranked_results],
            "reasoning": self.reasoning
        }


class HybridCrossModalRetriever:
    """
    Advanced cross-modal retrieval with LLM-based re-ranking.
    
    Features:
    - Parallel retrieval from text, table, and image collections
    - Cross-modal re-ranking using LLM reasoning
    - Multi-document support
    - Evidence grounding with bounding boxes
    """
    
    def __init__(
        self,
        vector_store: QdrantVectorStore = None,
        text_embedder: TextEmbedder = None,
        image_embedder: ImageEmbedder = None,
        table_embedder: TableEmbedder = None,
        k_per_modality: int = 5
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_store: Vector store instance
            text_embedder: Text embedding generator
            image_embedder: Image embedding generator
            table_embedder: Table embedding generator
            k_per_modality: Top-K results to retrieve per modality
        """
        self.vector_store = vector_store or QdrantVectorStore()
        self.text_embedder = text_embedder or TextEmbedder()
        self.image_embedder = image_embedder or ImageEmbedder()
        self.table_embedder = table_embedder or TableEmbedder(self.text_embedder)
        self.k_per_modality = k_per_modality
        self.llm_client = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize LLM client for re-ranking"""
        settings = get_settings()
        try:
            if settings.llm_provider == "groq":
                from groq import Groq
                self.llm_client = Groq(api_key=settings.groq_api_key)
            elif settings.llm_provider == "anthropic":
                import anthropic
                self.llm_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        except Exception as e:
            logger.warning(f"LLM client for re-ranking not available: {e}")
    
    def retrieve(
        self,
        query: str,
        document_ids: List[str] = None,
        k_final: int = 10
    ) -> CrossModalResult:
        """
        Perform hybrid cross-modal retrieval.
        
        Args:
            query: Natural language query
            document_ids: List of document IDs to search (multi-document support)
            k_final: Final number of results after re-ranking
            
        Returns:
            CrossModalResult with all modality results and re-ranked final
        """
        logger.info(f"Hybrid retrieval: '{query}' across {len(document_ids or [])} documents")
        
        # Step 1: Parallel retrieval from all modalities
        text_results = self._retrieve_text(query, document_ids)
        table_results = self._retrieve_tables(query, document_ids)
        image_results = self._retrieve_images(query, document_ids)
        
        logger.debug(f"Retrieved: {len(text_results)} text, {len(table_results)} tables, {len(image_results)} images")
        
        # Step 2: Combine all results
        all_results = text_results + table_results + image_results
        
        # Step 3: Re-rank using LLM reasoning
        reranked, reasoning = self._rerank_with_llm(query, all_results, k_final)
        
        return CrossModalResult(
            query=query,
            text_results=text_results,
            table_results=table_results,
            image_results=image_results,
            reranked_results=reranked,
            reasoning=reasoning
        )
    
    def _retrieve_text(
        self,
        query: str,
        document_ids: List[str] = None
    ) -> List[GroundedEvidence]:
        """Retrieve from text collection"""
        query_embedding = self.text_embedder.embed(query)
        
        results = []
        try:
            hits = self.vector_store.search_text(
                query_embedding,
                document_ids=document_ids,
                limit=self.k_per_modality
            )
            
            for hit in hits:
                results.append(GroundedEvidence(
                    content=hit.get("text", ""),
                    content_type="text",
                    score=hit.get("score", 0.0),
                    page_number=hit.get("page_number", 0),
                    document_id=hit.get("document_id", ""),
                    bbox=(hit.get("x1"), hit.get("y1"), hit.get("x2"), hit.get("y2")),
                    source_id=hit.get("id", "")
                ))
        except Exception as e:
            logger.warning(f"Text retrieval failed: {e}")
        
        return results
    
    def _retrieve_tables(
        self,
        query: str,
        document_ids: List[str] = None
    ) -> List[GroundedEvidence]:
        """Retrieve from table collection"""
        query_embedding = self.table_embedder.text_embedder.embed(query)
        
        results = []
        try:
            hits = self.vector_store.search_tables(
                query_embedding,
                document_ids=document_ids,
                limit=self.k_per_modality
            )
            
            for hit in hits:
                # Format table content for display
                headers = hit.get("headers", [])
                rows = hit.get("rows", [])[:3]  # First 3 rows
                content = self._format_table_content(headers, rows)
                
                results.append(GroundedEvidence(
                    content=content,
                    content_type="table",
                    score=hit.get("score", 0.0),
                    page_number=hit.get("page_number", 0),
                    document_id=hit.get("document_id", ""),
                    bbox=(hit.get("x1"), hit.get("y1"), hit.get("x2"), hit.get("y2")),
                    source_id=hit.get("id", "")
                ))
        except Exception as e:
            logger.warning(f"Table retrieval failed: {e}")
        
        return results
    
    def _retrieve_images(
        self,
        query: str,
        document_ids: List[str] = None
    ) -> List[GroundedEvidence]:
        """Retrieve from image collection using CLIP"""
        # Use CLIP text encoder for cross-modal search
        query_embedding = self.image_embedder.embed_text(query)
        
        results = []
        try:
            hits = self.vector_store.search_images(
                query_embedding,
                document_ids=document_ids,
                limit=self.k_per_modality
            )
            
            for hit in hits:
                results.append(GroundedEvidence(
                    content=hit.get("caption", f"Figure on page {hit.get('page_number', 0)}"),
                    content_type="image",
                    score=hit.get("score", 0.0),
                    page_number=hit.get("page_number", 0),
                    document_id=hit.get("document_id", ""),
                    bbox=(hit.get("x1"), hit.get("y1"), hit.get("x2"), hit.get("y2")),
                    source_id=hit.get("id", "")
                ))
        except Exception as e:
            logger.warning(f"Image retrieval failed: {e}")
        
        return results
    
    def _format_table_content(self, headers: List[str], rows: List[List]) -> str:
        """Format table for display"""
        lines = []
        if headers:
            lines.append(" | ".join(str(h) for h in headers))
            lines.append("-" * len(lines[0]))
        for row in rows:
            lines.append(" | ".join(str(cell) for cell in row))
        return "\n".join(lines)
    
    def _rerank_with_llm(
        self,
        query: str,
        results: List[GroundedEvidence],
        k_final: int
    ) -> Tuple[List[GroundedEvidence], str]:
        """
        Re-rank results using LLM reasoning.
        
        The LLM evaluates which evidence best answers the query
        and provides reasoning for the ranking.
        """
        if not results:
            return [], ""
        
        if not self.llm_client:
            # Fallback: use RRF scoring
            return self._rerank_rrf(results, k_final), "Using Reciprocal Rank Fusion (LLM unavailable)"
        
        # Build context for LLM
        context_parts = []
        for i, result in enumerate(results[:15]):  # Limit to top 15 for context
            context_parts.append(
                f"[{i+1}] Type: {result.content_type}, Page: {result.page_number}, "
                f"Doc: {result.document_id}\nContent: {result.content[:300]}..."
            )
        
        prompt = f"""Given the query and the following retrieved evidence from a document, 
rank the evidence by relevance. Return the indices of the top {k_final} most relevant items in order.

QUERY: {query}

EVIDENCE:
{chr(10).join(context_parts)}

Respond with:
1. RANKING: comma-separated indices (e.g., "3,1,5,2")
2. REASONING: Brief explanation of why the top items are most relevant

Focus on:
- Direct relevance to the query
- Cross-modal evidence (tables supporting text, images illustrating concepts)
- Multi-document connections if present"""

        try:
            settings = get_settings()
            
            if settings.llm_provider == "groq":
                response = self.llm_client.chat.completions.create(
                    model=settings.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=500
                )
                llm_response = response.choices[0].message.content
            else:
                response = self.llm_client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}]
                )
                llm_response = response.content[0].text
            
            # Parse response
            return self._parse_rerank_response(llm_response, results, k_final)
            
        except Exception as e:
            logger.warning(f"LLM re-ranking failed: {e}")
            return self._rerank_rrf(results, k_final), f"Fallback to RRF: {str(e)}"
    
    def _parse_rerank_response(
        self,
        response: str,
        results: List[GroundedEvidence],
        k_final: int
    ) -> Tuple[List[GroundedEvidence], str]:
        """Parse LLM re-ranking response"""
        try:
            lines = response.strip().split("\n")
            ranking_line = ""
            reasoning = ""
            
            for line in lines:
                if "RANKING:" in line.upper():
                    ranking_line = line.split(":", 1)[1].strip()
                elif "REASONING:" in line.upper():
                    reasoning = line.split(":", 1)[1].strip()
            
            if not ranking_line:
                # Try to find numbers
                import re
                numbers = re.findall(r'\d+', response)
                ranking_line = ",".join(numbers[:k_final])
            
            # Parse indices
            indices = [int(x.strip()) - 1 for x in ranking_line.split(",") if x.strip().isdigit()]
            
            reranked = []
            for idx in indices[:k_final]:
                if 0 <= idx < len(results):
                    result = results[idx]
                    result.reasoning = reasoning
                    reranked.append(result)
            
            return reranked, reasoning
            
        except Exception as e:
            logger.warning(f"Failed to parse re-rank response: {e}")
            return self._rerank_rrf(results, k_final), "Parse error, using RRF"
    
    def _rerank_rrf(
        self,
        results: List[GroundedEvidence],
        k_final: int
    ) -> List[GroundedEvidence]:
        """Reciprocal Rank Fusion fallback"""
        # Group by content type
        type_rankings = {"text": [], "table": [], "image": []}
        
        for i, result in enumerate(sorted(results, key=lambda x: x.score, reverse=True)):
            type_rankings[result.content_type].append((i, result))
        
        # Calculate RRF scores
        k = 60
        rrf_scores = {}
        
        for content_type, ranked in type_rankings.items():
            for rank, (original_idx, result) in enumerate(ranked):
                score = 1.0 / (k + rank + 1)
                if original_idx in rrf_scores:
                    rrf_scores[original_idx] += score
                else:
                    rrf_scores[original_idx] = score
        
        # Sort by RRF score
        sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        return [results[i] for i in sorted_indices[:k_final]]
    
    def retrieve_multi_document(
        self,
        query: str,
        document_ids: List[str],
        compare: bool = False
    ) -> Dict[str, Any]:
        """
        Multi-document retrieval with comparison support.
        
        Args:
            query: Natural language query
            document_ids: List of documents to search
            compare: Whether to compare across documents
            
        Returns:
            Results with optional comparison
        """
        result = self.retrieve(query, document_ids)
        
        if compare and len(document_ids) > 1:
            # Group results by document
            by_document = {}
            for evidence in result.reranked_results:
                doc_id = evidence.document_id
                if doc_id not in by_document:
                    by_document[doc_id] = []
                by_document[doc_id].append(evidence)
            
            return {
                "query": query,
                "documents": document_ids,
                "comparison": by_document,
                "merged_results": result.to_dict()
            }
        
        return result.to_dict()
