"""Embedding generator supporting multiple providers."""
import os
from typing import List, Optional
from abc import ABC, abstractmethod
import numpy as np


class BaseEmbedding(ABC):
    """Base class for embedding generators."""
    
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass


class OpenAIEmbedding(BaseEmbedding):
    """OpenAI embedding generator."""
    
    def __init__(self, model: str = "text-embedding-ada-002", api_key: str = None):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
            self.model = model
            self._dimension = 1536  # ada-002 dimension
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API."""
        # Handle empty texts
        texts = [t if t else " " for t in texts]
        
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        
        return [e.embedding for e in response.data]
    
    @property
    def dimension(self) -> int:
        return self._dimension


class LocalEmbedding(BaseEmbedding):
    """Local embedding using sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self._dimension = self.model.get_sentence_embedding_dimension()
        except ImportError:
            raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using local model."""
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()
    
    @property
    def dimension(self) -> int:
        return self._dimension


class EmbeddingGenerator:
    """Factory for embedding generators."""
    
    def __init__(self, provider: str = "local", **kwargs):
        """
        Initialize embedding generator.
        
        Args:
            provider: 'openai' or 'local'
            **kwargs: Arguments for the provider
        """
        if provider == "openai":
            self.embedder = OpenAIEmbedding(**kwargs)
        elif provider == "local":
            self.embedder = LocalEmbedding(**kwargs)
        else:
            raise ValueError(f"Unknown provider: {provider}")
        
        self.provider = provider
    
    def embed(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for texts in batches.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts per batch
            
        Returns:
            List of embedding vectors
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.embedder.embed(batch)
            all_embeddings.extend(embeddings)
        
        return all_embeddings
    
    def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.embedder.embed([text])[0]
    
    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self.embedder.dimension
