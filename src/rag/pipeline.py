"""Main RAG Pipeline orchestrating retrieval and generation."""
from typing import List, Dict, Optional
from dataclasses import dataclass
import time

from .retriever import Retriever, HybridRetriever
from .generator import AnswerGenerator, GeneratedAnswer


@dataclass
class RAGResponse:
    """Complete RAG response with all metadata."""
    query: str
    answer: str
    sources: List[Dict]
    confidence: float
    retrieved_chunks: List[Dict]
    response_time_ms: int


class RAGPipeline:
    """Main RAG pipeline for question answering."""
    
    
    def __init__(self, 
                 embedding_provider: str = "local",
                 llm_provider: str = "gemini",
                 top_k: int = 10,
                 use_hybrid: bool = False):
        """
        Initialize the RAG pipeline.
        
        Args:
            embedding_provider: 'openai' or 'local' for embeddings
            llm_provider: 'gemini', 'openai' or 'local' for answer generation
            top_k: Number of chunks to retrieve
            use_hybrid: Whether to use hybrid retrieval
        """
        if use_hybrid:
            self.retriever = HybridRetriever(
                embedding_provider=embedding_provider,
                top_k=top_k
            )
        else:
            self.retriever = Retriever(
                embedding_provider=embedding_provider,
                top_k=top_k
            )
        
        # Get API key from environment
        import os
        gemini_key = os.getenv("GEMINI_API_KEY")
        
        self.generator = AnswerGenerator(
            provider=llm_provider,
            api_key=gemini_key if llm_provider == "gemini" else None
        )
        self.top_k = top_k
    
    def query(self, question: str,
              company: str = None,
              filing_type: str = None) -> RAGResponse:
        """
        Process a user query through the RAG pipeline.
        
        Args:
            question: User's question
            company: Optional filter by company
            filing_type: Optional filter by filing type
            
        Returns:
            RAGResponse with answer and metadata
        """
        start_time = time.time()
        
        # Step 1: Retrieve relevant chunks
        if hasattr(self.retriever, 'retrieve'):
            retrieved_chunks = self.retriever.retrieve(
                question,
                company=company,
                filing_type=filing_type
            )
        else:
            retrieved_chunks = []
        
        # Step 2: Generate answer
        generated = self.generator.generate(question, retrieved_chunks)
        
        # Calculate response time
        response_time_ms = int((time.time() - start_time) * 1000)
        
        return RAGResponse(
            query=question,
            answer=generated.answer,
            sources=generated.sources,
            confidence=generated.confidence,
            retrieved_chunks=retrieved_chunks,
            response_time_ms=response_time_ms
        )
    
    def query_with_followup(self, question: str,
                           previous_context: str = None,
                           company: str = None) -> RAGResponse:
        """
        Handle follow-up questions with conversation context.
        
        Args:
            question: Current question
            previous_context: Context from previous answers
            company: Company filter
            
        Returns:
            RAGResponse
        """
        # Enhance query with previous context if available
        if previous_context:
            enhanced_query = f"{previous_context}\n\nFollow-up question: {question}"
        else:
            enhanced_query = question
        
        return self.query(enhanced_query, company=company)
    
    def get_companies(self) -> List[str]:
        """Get list of available companies."""
        stats = self.retriever.vector_db.get_stats()
        return list(stats.get('companies', {}).keys())


# Convenience function for quick usage
def ask(question: str, company: str = None) -> str:
    """
    Quick function to ask a question.
    
    Args:
        question: Your question about financial filings
        company: Optional company filter
        
    Returns:
        Answer string
    """
    pipeline = RAGPipeline(
        embedding_provider="local",
        llm_provider="local",
        top_k=5
    )
    
    response = pipeline.query(question, company=company)
    
    return response.answer
