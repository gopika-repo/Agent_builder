"""
RAG Pipeline

Complete RAG pipeline for document question answering:
- Document indexing
- Query processing
- Response generation with multi-modal context
- ELI5 vs Expert mode
"""

import logging
from typing import Dict, Any, List, Optional
import json

from .vector_store import QdrantVectorStore
from .embeddings import TextEmbedder, ImageEmbedder, TableEmbedder
from .retriever import MultiModalRetriever, RetrievalResult
from ..config import get_settings

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Complete RAG pipeline for document Q&A.
    
    Features:
    - Multi-modal document indexing
    - Cross-modal retrieval
    - LLM-powered response generation
    - ELI5 vs Expert mode explanations
    """
    
    def __init__(
        self,
        vector_store: QdrantVectorStore = None,
        retriever: MultiModalRetriever = None
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            vector_store: Vector store instance
            retriever: Retriever instance
        """
        self.vector_store = vector_store or QdrantVectorStore()
        
        self.text_embedder = TextEmbedder()
        self.image_embedder = ImageEmbedder()
        self.table_embedder = TableEmbedder(self.text_embedder)
        
        self.retriever = retriever or MultiModalRetriever(
            vector_store=self.vector_store,
            text_embedder=self.text_embedder,
            image_embedder=self.image_embedder,
            table_embedder=self.table_embedder
        )
        
        self.llm_client = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize LLM client"""
        settings = get_settings()
        
        try:
            if settings.llm_provider == "groq":
                from groq import Groq
                self.llm_client = Groq(api_key=settings.groq_api_key)
            elif settings.llm_provider == "anthropic":
                import anthropic
                self.llm_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        except Exception as e:
            logger.warning(f"LLM client initialization failed: {e}")
            self.llm_client = None
    
    def index_document(
        self,
        document_id: str,
        fused_output: Dict[str, Any]
    ) -> Dict[str, int]:
        """
        Index a processed document into the vector store.
        
        Args:
            document_id: Document identifier
            fused_output: Fused document output from pipeline
            
        Returns:
            Statistics about indexed content
        """
        stats = {"text_chunks": 0, "tables": 0, "images": 0}
        
        elements = fused_output.get("elements", [])
        tables = fused_output.get("tables", [])
        
        # Index text elements
        text_chunks = []
        text_embeddings = []
        
        for element in elements:
            if element.get("type") in ["paragraph", "text", "heading", "list"]:
                content = element.get("content", "")
                if content and len(content) > 10:
                    text_chunks.append({
                        "text": content,
                        "page_number": element.get("page_number", 0),
                        "x1": element.get("bbox", [0, 0, 0, 0])[0] if element.get("bbox") else 0,
                        "y1": element.get("bbox", [0, 0, 0, 0])[1] if element.get("bbox") else 0,
                        "x2": element.get("bbox", [0, 0, 0, 0])[2] if element.get("bbox") else 0,
                        "y2": element.get("bbox", [0, 0, 0, 0])[3] if element.get("bbox") else 0,
                    })
                    text_embeddings.append(self.text_embedder.embed(content))
        
        if text_chunks:
            stats["text_chunks"] = self.vector_store.add_text_chunks(
                document_id, text_chunks, text_embeddings
            )
        
        # Index tables
        if tables:
            table_data = []
            table_embeddings = []
            
            for table in tables:
                table_data.append({
                    "id": table.get("id", ""),
                    "page_number": table.get("page_number", 0),
                    "headers": table.get("headers", []),
                    "rows": table.get("rows", []),
                    "num_rows": table.get("num_rows", 0),
                    "num_cols": table.get("num_cols", 0)
                })
                table_embeddings.append(self.table_embedder.embed_table(table))
            
            stats["tables"] = self.vector_store.add_tables(
                document_id, table_data, table_embeddings
            )
        
        # Index images/figures
        image_elements = [
            e for e in elements 
            if e.get("type") in ["figure", "image", "chart"]
        ]
        
        if image_elements:
            image_data = []
            image_embeddings = []
            
            for img in image_elements:
                image_data.append({
                    "id": img.get("id", ""),
                    "page_number": img.get("page_number", 0),
                    "label": img.get("type", "figure"),
                    "caption": img.get("content", "")[:200],
                    "x1": img.get("bbox", [0, 0, 0, 0])[0] if img.get("bbox") else 0,
                    "y1": img.get("bbox", [0, 0, 0, 0])[1] if img.get("bbox") else 0,
                    "x2": img.get("bbox", [0, 0, 0, 0])[2] if img.get("bbox") else 0,
                    "y2": img.get("bbox", [0, 0, 0, 0])[3] if img.get("bbox") else 0,
                })
                
                # Use text description for embedding (no actual image available)
                caption = img.get("content", "figure")
                image_embeddings.append(self.image_embedder.embed_text(caption))
            
            stats["images"] = self.vector_store.add_images(
                document_id, image_data, image_embeddings
            )
        
        logger.info(f"Indexed document {document_id}: {stats}")
        return stats
    
    def query(
        self,
        query: str,
        document_id: str,
        mode: str = "standard",
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Query a document using RAG.
        
        Args:
            query: Natural language query
            document_id: Document to query
            mode: 'standard', 'eli5', or 'expert'
            include_sources: Whether to include source references
            
        Returns:
            Response with answer and sources
        """
        # Retrieve relevant context
        results = self.retriever.retrieve(
            query=query,
            document_id=document_id,
            limit=10
        )
        
        # Format context for LLM
        context = self._format_context(results)
        
        # Generate response
        response = self._generate_response(query, context, mode)
        
        output = {
            "answer": response,
            "mode": mode,
            "query": query
        }
        
        if include_sources:
            output["sources"] = [
                {
                    "content": r.content[:200] + "..." if len(r.content) > 200 else r.content,
                    "type": r.content_type,
                    "page": r.page_number,
                    "score": round(r.score, 3)
                }
                for r in results[:5]
            ]
        
        return output
    
    def _format_context(self, results: List[RetrievalResult]) -> str:
        """Format retrieval results as context for LLM"""
        context_parts = []
        
        for i, result in enumerate(results, 1):
            if result.content_type == "text":
                context_parts.append(
                    f"[Source {i} - Text from page {result.page_number}]\n{result.content}"
                )
            elif result.content_type == "table":
                context_parts.append(
                    f"[Source {i} - Table from page {result.page_number}]\n{result.content}"
                )
            elif result.content_type == "image":
                context_parts.append(
                    f"[Source {i} - Figure from page {result.page_number}]\n{result.content}"
                )
        
        return "\n\n".join(context_parts)
    
    def _generate_response(
        self,
        query: str,
        context: str,
        mode: str
    ) -> str:
        """Generate response using LLM"""
        # Build prompt based on mode
        if mode == "eli5":
            system_prompt = """You are a friendly teacher explaining documents to someone with no prior knowledge.
Use simple words, analogies, and everyday examples. Avoid jargon.
If you reference numbers or data, explain what they mean in simple terms.
Your explanation should be easy enough for a 5-year-old to understand the main idea."""
        elif mode == "expert":
            system_prompt = """You are a senior analyst providing expert-level document analysis.
Use precise technical terminology and cite specific data points.
Provide nuanced interpretations and identify implications.
Reference specific sources and be quantitative where possible."""
        else:
            system_prompt = """You are a helpful document assistant.
Answer questions based on the provided document context.
Be accurate, clear, and cite relevant sections when helpful."""
        
        user_prompt = f"""Based on the following document context, answer the question.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
        
        if self.llm_client:
            try:
                settings = get_settings()
                
                if settings.llm_provider == "groq":
                    response = self.llm_client.chat.completions.create(
                        model=settings.llm_model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.3 if mode == "expert" else 0.5,
                        max_tokens=1000
                    )
                    return response.choices[0].message.content
                    
                elif settings.llm_provider == "anthropic":
                    response = self.llm_client.messages.create(
                        model=settings.llm_model,
                        max_tokens=1000,
                        system=system_prompt,
                        messages=[{"role": "user", "content": user_prompt}]
                    )
                    return response.content[0].text
                    
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
        
        # Fallback response
        return self._generate_fallback_response(query, context, mode)
    
    def _generate_fallback_response(
        self,
        query: str,
        context: str,
        mode: str
    ) -> str:
        """Generate fallback response when LLM is unavailable"""
        # Extract relevant snippets from context
        lines = context.split("\n")
        relevant = [l for l in lines if any(
            word.lower() in l.lower() 
            for word in query.split()
        )][:3]
        
        if mode == "eli5":
            prefix = "In simple terms: "
        elif mode == "expert":
            prefix = "Based on the document analysis: "
        else:
            prefix = "Based on the document: "
        
        if relevant:
            return prefix + " ".join(relevant)
        else:
            return prefix + "The document contains relevant information about your query. Key details can be found in the sources provided."
    
    def explain_element(
        self,
        document_id: str,
        element_id: str,
        mode: str = "standard"
    ) -> Dict[str, Any]:
        """
        Explain a specific document element.
        
        Args:
            document_id: Document ID
            element_id: Element to explain
            mode: Explanation mode
            
        Returns:
            Explanation response
        """
        return self.query(
            query=f"Explain the element with ID {element_id} in detail",
            document_id=document_id,
            mode=mode
        )
    
    def compare_explanations(
        self,
        query: str,
        document_id: str
    ) -> Dict[str, Any]:
        """
        Get both ELI5 and Expert explanations for comparison.
        
        Args:
            query: Query to explain
            document_id: Document ID
            
        Returns:
            Both explanation modes
        """
        eli5_response = self.query(query, document_id, mode="eli5")
        expert_response = self.query(query, document_id, mode="expert")
        
        return {
            "query": query,
            "eli5": eli5_response["answer"],
            "expert": expert_response["answer"],
            "sources": eli5_response.get("sources", [])
        }
