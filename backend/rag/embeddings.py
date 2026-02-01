"""
Embedding Generators

Provides embedding generation for different modalities:
- TextEmbedder: SentenceTransformers for text
- ImageEmbedder: CLIP for images  
- TableEmbedder: Row-aware table embeddings
"""

import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

# Try importing optional dependencies
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    from PIL import Image
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False


class TextEmbedder:
    """
    Text embedding using SentenceTransformers.
    
    Uses all-MiniLM-L6-v2 by default (384 dimensions).
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = None
    ):
        """
        Initialize text embedder.
        
        Args:
            model_name: SentenceTransformer model name
            device: Device to use ('cpu', 'cuda')
        """
        self.model_name = model_name
        self.device = device or ("cuda" if self._cuda_available() else "cpu")
        self.model = None
        self.dimension = 384
        
        self._initialize()
    
    def _cuda_available(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def _initialize(self):
        """Initialize the model"""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(self.model_name)
                self.model.to(self.device)
                self.dimension = self.model.get_sentence_embedding_dimension()
                logger.info(f"Loaded SentenceTransformer: {self.model_name} (dim={self.dimension})")
            except Exception as e:
                logger.error(f"Failed to load SentenceTransformer: {e}")
                self.model = None
        else:
            logger.warning("sentence-transformers not available, using mock embeddings")
    
    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        if self.model:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        else:
            # Mock embedding
            return self._mock_embedding(text)
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        if self.model:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        else:
            return [self._mock_embedding(t) for t in texts]
    
    def _mock_embedding(self, text: str) -> List[float]:
        """Generate mock embedding based on text hash"""
        np.random.seed(hash(text) % (2**32))
        embedding = np.random.randn(self.dimension)
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()


class ImageEmbedder:
    """
    Image embedding using CLIP.
    
    Uses openai/clip-vit-base-patch32 (512 dimensions).
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = None
    ):
        """
        Initialize image embedder.
        
        Args:
            model_name: CLIP model name
            device: Device to use
        """
        self.model_name = model_name
        self.device = device or ("cuda" if self._cuda_available() else "cpu")
        self.model = None
        self.processor = None
        self.dimension = 512
        
        self._initialize()
    
    def _cuda_available(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def _initialize(self):
        """Initialize CLIP model"""
        if CLIP_AVAILABLE:
            try:
                self.processor = CLIPProcessor.from_pretrained(self.model_name)
                self.model = CLIPModel.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model.eval()
                logger.info(f"Loaded CLIP model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to load CLIP: {e}")
                self.model = None
        else:
            logger.warning("CLIP not available, using mock embeddings")
    
    def embed_image(self, image: Union[np.ndarray, 'Image.Image']) -> List[float]:
        """
        Generate embedding for an image.
        
        Args:
            image: Input image (numpy array or PIL Image)
            
        Returns:
            Embedding vector
        """
        if self.model and self.processor:
            try:
                # Convert numpy to PIL if needed
                if isinstance(image, np.ndarray):
                    from PIL import Image as PILImage
                    if len(image.shape) == 3 and image.shape[2] == 3:
                        # BGR to RGB
                        import cv2
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = PILImage.fromarray(image)
                
                inputs = self.processor(images=image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    image_features = self.model.get_image_features(**inputs)
                
                embedding = image_features.cpu().numpy().flatten()
                # Normalize
                embedding = embedding / np.linalg.norm(embedding)
                return embedding.tolist()
                
            except Exception as e:
                logger.error(f"CLIP embedding failed: {e}")
                return self._mock_embedding()
        else:
            return self._mock_embedding()
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate CLIP text embedding (for cross-modal search).
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        if self.model and self.processor:
            try:
                inputs = self.processor(text=[text], return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    text_features = self.model.get_text_features(**inputs)
                
                embedding = text_features.cpu().numpy().flatten()
                embedding = embedding / np.linalg.norm(embedding)
                return embedding.tolist()
                
            except Exception as e:
                logger.error(f"CLIP text embedding failed: {e}")
                return self._mock_embedding()
        else:
            return self._mock_embedding()
    
    def _mock_embedding(self) -> List[float]:
        """Generate mock embedding"""
        embedding = np.random.randn(self.dimension)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()


class TableEmbedder:
    """
    Table-aware embedding generator.
    
    Creates embeddings that capture table structure:
    - Header semantics
    - Row-level content
    - Column relationships
    """
    
    def __init__(
        self,
        text_embedder: TextEmbedder = None
    ):
        """
        Initialize table embedder.
        
        Args:
            text_embedder: TextEmbedder to use for text embedding
        """
        self.text_embedder = text_embedder or TextEmbedder()
        self.dimension = self.text_embedder.dimension
    
    def embed_table(self, table: Dict[str, Any]) -> List[float]:
        """
        Generate embedding for a table.
        
        Creates a combined embedding from:
        - Headers
        - First few rows
        - Statistical summary
        
        Args:
            table: Table data with headers and rows
            
        Returns:
            Embedding vector
        """
        # Build table representation
        parts = []
        
        # Add headers
        headers = table.get("headers", [])
        if headers:
            parts.append("Headers: " + ", ".join(str(h) for h in headers))
        
        # Add first few rows
        rows = table.get("rows", [])
        for i, row in enumerate(rows[:3]):
            row_text = " | ".join(str(cell) for cell in row)
            parts.append(f"Row {i+1}: {row_text}")
        
        # Add summary
        summary = f"Table with {len(rows)} rows and {len(headers)} columns"
        parts.append(summary)
        
        # Combine and embed
        table_text = "\n".join(parts)
        return self.text_embedder.embed(table_text)
    
    def embed_row(
        self,
        row: List[Any],
        headers: List[str] = None
    ) -> List[float]:
        """
        Generate embedding for a single table row.
        
        Args:
            row: Row data
            headers: Optional column headers
            
        Returns:
            Embedding vector
        """
        if headers and len(headers) == len(row):
            # Create key-value pairs
            pairs = [f"{h}: {v}" for h, v in zip(headers, row)]
            row_text = ", ".join(pairs)
        else:
            row_text = " | ".join(str(cell) for cell in row)
        
        return self.text_embedder.embed(row_text)
    
    def embed_tables_batch(
        self,
        tables: List[Dict[str, Any]]
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple tables.
        
        Args:
            tables: List of table data
            
        Returns:
            List of embedding vectors
        """
        return [self.embed_table(table) for table in tables]
