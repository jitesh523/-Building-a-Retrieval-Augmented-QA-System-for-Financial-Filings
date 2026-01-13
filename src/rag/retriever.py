"""Retriever module for finding relevant chunks."""
from typing import List, Dict, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.embeddings.generator import EmbeddingGenerator
from src.database.vector_db import VectorDB
from src.config import settings


class Retriever:
    """Retrieves relevant chunks for a query."""
    
    def __init__(self, embedding_provider: str = "local", top_k: int = 10):
        """
        Initialize the retriever.
        
        Args:
            embedding_provider: 'openai' or 'local'
            top_k: Number of results to return
        """
        self.embedder = EmbeddingGenerator(provider=embedding_provider)
        self.vector_db = VectorDB(settings.database_url)
        self.top_k = top_k
        self.model_name = embedding_provider
    
    def retrieve(self, query: str, 
                 company: str = None,
                 filing_type: str = None,
                 top_k: int = None) -> List[Dict]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: User query string
            company: Optional filter by company name
            filing_type: Optional filter by filing type (10-K, 10-Q)
            top_k: Number of results (defaults to self.top_k)
            
        Returns:
            List of relevant chunks with metadata
        """
        # Generate query embedding
        query_embedding = self.embedder.embed_single(query)
        
        # Search for similar chunks
        results = self.vector_db.search_similar(
            query_embedding=query_embedding,
            top_k=top_k or self.top_k,
            model_name=self.model_name,
            company=company,
            filing_type=filing_type
        )
        
        return results
    
    def retrieve_with_threshold(self, query: str,
                                 threshold: float = 0.5,
                                 max_results: int = 20,
                                 company: str = None) -> List[Dict]:
        """
        Retrieve chunks above a similarity threshold.
        
        Args:
            query: User query
            threshold: Minimum similarity score
            max_results: Maximum results to return
            company: Optional company filter
            
        Returns:
            List of chunks with similarity >= threshold
        """
        results = self.retrieve(query, company=company, top_k=max_results)
        
        # Filter by threshold
        filtered = [r for r in results if r['similarity'] >= threshold]
        
        return filtered


class HybridRetriever:
    """Hybrid retriever combining vector and keyword search."""
    
    def __init__(self, embedding_provider: str = "local", top_k: int = 10):
        """Initialize hybrid retriever."""
        self.vector_retriever = Retriever(
            embedding_provider=embedding_provider, 
            top_k=top_k * 2  # Get more for reranking
        )
        self.top_k = top_k
    
    def retrieve(self, query: str, 
                 company: str = None,
                 filing_type: str = None) -> List[Dict]:
        """
        Retrieve using hybrid approach.
        
        Args:
            query: User query
            company: Optional company filter
            filing_type: Optional filing type filter
            
        Returns:
            List of relevant chunks
        """
        # Get vector search results
        vector_results = self.vector_retriever.retrieve(
            query, company=company, filing_type=filing_type
        )
        
        # Simple keyword boost: increase score for chunks containing query terms
        query_terms = set(query.lower().split())
        
        for result in vector_results:
            chunk_text_lower = result['chunk_text'].lower()
            term_matches = sum(1 for term in query_terms if term in chunk_text_lower)
            
            # Boost similarity score by keyword matches
            keyword_boost = term_matches * 0.05
            result['similarity'] = min(1.0, result['similarity'] + keyword_boost)
            result['keyword_matches'] = term_matches
        
        # Re-sort by adjusted similarity
        vector_results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return vector_results[:self.top_k]
