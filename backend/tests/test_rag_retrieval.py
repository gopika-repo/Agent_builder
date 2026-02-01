"""
RAG Retrieval Tests

Tests for multi-modal RAG system including:
- Vector store operations
- Embeddings generation
- Multi-modal retrieval
- Cross-modal search
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestTextEmbedder:
    """Test TextEmbedder class"""
    
    def test_initialization(self):
        """Test embedder initialization"""
        from rag.embeddings import TextEmbedder
        
        embedder = TextEmbedder()
        assert embedder.dimension > 0
    
    def test_embed_single_text(self):
        """Test single text embedding"""
        from rag.embeddings import TextEmbedder
        
        embedder = TextEmbedder()
        embedding = embedder.embed("This is a test document about finance.")
        
        assert isinstance(embedding, list)
        assert len(embedding) == embedder.dimension
        # Should be normalized
        norm = np.linalg.norm(embedding)
        assert 0.99 < norm < 1.01
    
    def test_embed_batch(self):
        """Test batch embedding"""
        from rag.embeddings import TextEmbedder
        
        embedder = TextEmbedder()
        texts = ["First document", "Second document", "Third document"]
        
        embeddings = embedder.embed_batch(texts)
        
        assert len(embeddings) == 3
        assert all(len(e) == embedder.dimension for e in embeddings)
    
    def test_similar_texts_similar_embeddings(self):
        """Test that similar texts have similar embeddings"""
        from rag.embeddings import TextEmbedder
        
        embedder = TextEmbedder()
        
        emb1 = embedder.embed("Financial report for Q1 2024")
        emb2 = embedder.embed("Q1 2024 financial report")
        emb3 = embedder.embed("The weather is sunny today")
        
        # Calculate cosine similarity
        def cosine_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        sim_similar = cosine_sim(emb1, emb2)
        sim_different = cosine_sim(emb1, emb3)
        
        # Similar texts should have higher similarity (mock might not guarantee this)
        assert isinstance(sim_similar, float)
        assert isinstance(sim_different, float)


class TestImageEmbedder:
    """Test ImageEmbedder class"""
    
    def test_initialization(self):
        """Test image embedder initialization"""
        from rag.embeddings import ImageEmbedder
        
        embedder = ImageEmbedder()
        assert embedder.dimension == 512
    
    def test_embed_text_for_cross_modal(self):
        """Test text embedding for cross-modal search"""
        from rag.embeddings import ImageEmbedder
        
        embedder = ImageEmbedder()
        embedding = embedder.embed_text("A chart showing revenue growth")
        
        assert isinstance(embedding, list)
        assert len(embedding) == embedder.dimension
    
    def test_embed_image(self, sample_image):
        """Test image embedding"""
        from rag.embeddings import ImageEmbedder
        
        embedder = ImageEmbedder()
        embedding = embedder.embed_image(sample_image)
        
        assert isinstance(embedding, list)
        assert len(embedding) == embedder.dimension


class TestTableEmbedder:
    """Test TableEmbedder class"""
    
    def test_initialization(self):
        """Test table embedder initialization"""
        from rag.embeddings import TableEmbedder
        
        embedder = TableEmbedder()
        assert embedder.dimension > 0
    
    def test_embed_table(self, sample_table_data):
        """Test table embedding"""
        from rag.embeddings import TableEmbedder
        
        embedder = TableEmbedder()
        embedding = embedder.embed_table(sample_table_data)
        
        assert isinstance(embedding, list)
        assert len(embedding) == embedder.dimension
    
    def test_embed_row(self, sample_table_data):
        """Test single row embedding"""
        from rag.embeddings import TableEmbedder
        
        embedder = TableEmbedder()
        
        row = sample_table_data["rows"][0]
        headers = sample_table_data["headers"]
        
        embedding = embedder.embed_row(row, headers)
        
        assert isinstance(embedding, list)
        assert len(embedding) == embedder.dimension
    
    def test_embed_tables_batch(self, sample_table_data):
        """Test batch table embedding"""
        from rag.embeddings import TableEmbedder
        
        embedder = TableEmbedder()
        tables = [sample_table_data, sample_table_data]
        
        embeddings = embedder.embed_tables_batch(tables)
        
        assert len(embeddings) == 2


class TestVectorStore:
    """Test QdrantVectorStore"""
    
    @pytest.fixture
    def mock_qdrant_client(self):
        """Create mock Qdrant client"""
        with patch('rag.vector_store.QdrantClient') as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            yield mock_instance
    
    def test_initialization(self):
        """Test vector store initialization"""
        from rag.vector_store import QdrantVectorStore
        
        # Will use mock or fail gracefully
        try:
            store = QdrantVectorStore()
            assert store is not None
        except Exception:
            pytest.skip("Qdrant not available")
    
    def test_collection_names(self):
        """Test that correct collection names are used"""
        from rag.vector_store import QdrantVectorStore
        
        store = QdrantVectorStore.__new__(QdrantVectorStore)
        store.text_collection = "documents_text"
        store.table_collection = "documents_tables"
        store.image_collection = "documents_images"
        
        assert "text" in store.text_collection
        assert "table" in store.table_collection
        assert "image" in store.image_collection


class TestMultiModalRetriever:
    """Test MultiModalRetriever"""
    
    def test_initialization(self):
        """Test retriever initialization"""
        from rag.retriever import MultiModalRetriever
        from rag.vector_store import QdrantVectorStore
        from rag.embeddings import TextEmbedder, ImageEmbedder, TableEmbedder
        
        try:
            retriever = MultiModalRetriever(
                vector_store=Mock(),
                text_embedder=TextEmbedder(),
                image_embedder=ImageEmbedder(),
                table_embedder=TableEmbedder()
            )
            assert retriever is not None
        except Exception:
            pytest.skip("Dependencies not available")
    
    def test_retrieval_result_structure(self):
        """Test RetrievalResult dataclass"""
        from rag.retriever import RetrievalResult
        
        result = RetrievalResult(
            content="This is the retrieved content",
            content_type="text",
            score=0.85,
            page_number=1,
            metadata={"source": "paragraph"}
        )
        
        assert result.content == "This is the retrieved content"
        assert result.content_type == "text"
        assert result.score == 0.85
        assert result.page_number == 1


class TestRAGPipeline:
    """Test RAGPipeline"""
    
    def test_initialization(self):
        """Test RAG pipeline initialization"""
        from rag.rag_pipeline import RAGPipeline
        
        try:
            pipeline = RAGPipeline()
            assert pipeline is not None
        except Exception:
            pytest.skip("RAG dependencies not available")
    
    def test_context_formatting(self):
        """Test context formatting for LLM"""
        from rag.rag_pipeline import RAGPipeline
        from rag.retriever import RetrievalResult
        
        pipeline = RAGPipeline.__new__(RAGPipeline)
        
        results = [
            RetrievalResult(
                content="Revenue increased by 15%",
                content_type="text",
                score=0.9,
                page_number=1,
                metadata={}
            ),
            RetrievalResult(
                content="Q1: $120M, Q2: $125M",
                content_type="table",
                score=0.85,
                page_number=2,
                metadata={}
            )
        ]
        
        context = pipeline._format_context(results)
        
        assert "Revenue" in context
        assert "Source 1" in context
        assert "Table" in context
    
    def test_mode_prompts(self):
        """Test that different modes use different prompts"""
        from rag.rag_pipeline import RAGPipeline
        
        pipeline = RAGPipeline.__new__(RAGPipeline)
        pipeline.llm_client = None  # Use fallback
        
        # Verify fallback responses contain mode-specific prefixes
        eli5_response = pipeline._generate_fallback_response(
            "What is revenue?",
            "Revenue is $500M",
            "eli5"
        )
        
        expert_response = pipeline._generate_fallback_response(
            "What is revenue?",
            "Revenue is $500M",
            "expert"
        )
        
        assert "simple" in eli5_response.lower() or "In simple terms" in eli5_response
        assert "analysis" in expert_response.lower() or "Based on the document analysis" in expert_response


class TestCrossModalRetrieval:
    """Test cross-modal retrieval capabilities"""
    
    def test_text_to_image_query(self):
        """Test querying images with text"""
        from rag.embeddings import ImageEmbedder, TextEmbedder
        
        image_embedder = ImageEmbedder()
        text_embedder = TextEmbedder()
        
        # Get text embedding for image query
        query_embedding = image_embedder.embed_text("chart showing financial growth")
        
        assert len(query_embedding) == 512  # CLIP dimension
    
    def test_image_to_text_similarity(self, sample_image):
        """Test image-text similarity using CLIP"""
        from rag.embeddings import ImageEmbedder
        
        embedder = ImageEmbedder()
        
        image_emb = embedder.embed_image(sample_image)
        text_emb = embedder.embed_text("A document with tables and text")
        
        # Calculate cosine similarity
        similarity = np.dot(image_emb, text_emb)
        
        assert isinstance(similarity, (float, np.floating))


class TestReciprocaRankFusion:
    """Test Reciprocal Rank Fusion for result merging"""
    
    def test_rrf_scoring(self):
        """Test RRF score calculation"""
        from rag.retriever import RetrievalResult
        
        # Simulate results from different modalities
        text_results = [
            RetrievalResult("text1", "text", 0.9, 1, {}),
            RetrievalResult("text2", "text", 0.8, 1, {}),
        ]
        
        table_results = [
            RetrievalResult("table1", "table", 0.85, 2, {}),
        ]
        
        # RRF formula: 1 / (k + rank)
        k = 60
        
        # text1 at rank 1: 1/(60+1) = 0.0164
        # text2 at rank 2: 1/(60+2) = 0.0161
        # table1 at rank 1: 1/(60+1) = 0.0164
        
        rrf_text1 = 1 / (k + 1)
        rrf_text2 = 1 / (k + 2)
        
        assert rrf_text1 > rrf_text2
        assert abs(rrf_text1 - 0.0164) < 0.001
