"""Answer generator using LLM."""
import os
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class GeneratedAnswer:
    """Container for generated answer."""
    answer: str
    sources: List[Dict]
    confidence: float
    context_used: str


class AnswerGenerator:
    """Generates answers using an LLM based on retrieved context."""
    
    def __init__(self, provider: str = "gemini", model: str = None, api_key: str = None):
        """
        Initialize the answer generator.
        
        Args:
            provider: 'gemini', 'openai', or 'local'
            model: Specific model name
            api_key: API key (optional, falls back to env vars)
        """
        self.provider = provider
        
        if provider == "gemini":
            try:
                from google import genai
                key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                if not key:
                    raise ValueError("Gemini API key not found")
                self.client = genai.Client(api_key=key)
                self.model = model or "gemini-2.5-flash"
            except ImportError:
                raise ImportError("google-genai package not installed. Run: pip install google-genai")
        elif provider == "openai":
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
                self.model = model or "gpt-4-turbo-preview"
            except ImportError:
                raise ImportError("OpenAI package not installed")
        else:
            # Fallback to local generation (simple extractive approach)
            self.client = None
            self.model = "extractive"
    
    def generate(self, query: str, retrieved_chunks: List[Dict],
                 max_context_length: int = 8000) -> GeneratedAnswer:
        """
        Generate an answer based on query and retrieved context.
        
        Args:
            query: User question
            retrieved_chunks: List of relevant chunks from retriever
            max_context_length: Maximum context length
            
        Returns:
            GeneratedAnswer with answer, sources, and confidence
        """
        if not retrieved_chunks:
            return GeneratedAnswer(
                answer="I couldn't find relevant information to answer your question.",
                sources=[],
                confidence=0.0,
                context_used=""
            )
        
        # Build context from chunks
        context_parts = []
        sources = []
        total_length = 0
        
        for i, chunk in enumerate(retrieved_chunks):
            chunk_text = chunk['chunk_text']
            
            # Check if adding this chunk would exceed limit
            if total_length + len(chunk_text) > max_context_length:
                break
            
            context_parts.append(f"[Source {i+1} - {chunk.get('company', 'Unknown')}]: {chunk_text}")
            sources.append({
                'index': i + 1,
                'company': chunk.get('company'),
                'filing_type': chunk.get('filing_type'),
                'filing_date': chunk.get('filing_date'),
                'section': chunk.get('section'),
                'similarity': chunk.get('similarity', 0)
            })
            total_length += len(chunk_text)
        
        context = "\n\n".join(context_parts)
        
        if self.provider == "gemini":
            answer = self._generate_gemini(query, context)
        elif self.provider == "openai" and self.client:
            answer = self._generate_openai(query, context)
        else:
            answer = self._generate_extractive(query, context, retrieved_chunks)
        
        # Calculate confidence based on top chunk similarity
        top_similarity = retrieved_chunks[0].get('similarity', 0.5) if retrieved_chunks else 0
        confidence = min(0.95, top_similarity + 0.1)
        
        return GeneratedAnswer(
            answer=answer,
            sources=sources,
            confidence=confidence,
            context_used=context
        )
    
    def _generate_gemini(self, query: str, context: str) -> str:
        """Generate answer using Google Gemini API."""
        prompt = f"""You are a financial analyst assistant specialized in analyzing SEC filings (10-K, 10-Q reports).
Your task is to answer questions based ONLY on the provided context from financial filings.

Guidelines:
- Answer the question directly and concisely
- Use specific numbers, dates, and facts from the context
- If the context doesn't contain enough information to fully answer, say so
- Cite sources using [Source N] notation when referencing specific information
- If the question asks about multiple companies, synthesize information from all relevant sources
- Do not make up information not present in the context

Context from SEC filings:
{context}

Question: {query}

Please provide a clear, accurate answer based on the context above."""

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            return response.text
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def _generate_openai(self, query: str, context: str) -> str:
        """Generate answer using OpenAI API."""
        system_prompt = """You are a financial analyst assistant specialized in analyzing SEC filings (10-K, 10-Q reports).
Your task is to answer questions based ONLY on the provided context from financial filings.

Guidelines:
- Answer the question directly and concisely
- Use specific numbers, dates, and facts from the context
- If the context doesn't contain enough information, say so
- Cite sources using [Source N] notation when referencing specific information
- Do not make up information not present in the context"""

        user_prompt = f"""Context from SEC filings:
{context}

Question: {query}

Please provide a clear, accurate answer based on the context above."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    
    def _generate_extractive(self, query: str, context: str, 
                              chunks: List[Dict]) -> str:
        """Simple extractive answer generation without LLM API."""
        # Return the most relevant chunk with some formatting
        if not chunks:
            return "No relevant information found."
        
        # Get top chunk
        top_chunk = chunks[0]
        company = top_chunk.get('company', 'Unknown')
        
        answer = f"Based on {company}'s SEC filing:\n\n"
        answer += f'"{top_chunk["chunk_text"][:500]}..."'
        answer += f"\n\n[Source: {company} {top_chunk.get('filing_type', '')} {top_chunk.get('filing_date', '')}]"
        
        return answer

