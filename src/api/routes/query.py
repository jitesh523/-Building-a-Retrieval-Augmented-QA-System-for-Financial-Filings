"""Query endpoints for RAG pipeline."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import logging

from src.rag.pipeline import RAGPipeline

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize RAG pipeline
_pipeline = None

def get_pipeline():
    """Get or create RAG pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(
            embedding_provider="local",
            llm_provider="local",
            top_k=10,
            use_hybrid=True
        )
    return _pipeline


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    question: str = Field(..., min_length=3, description="Question to ask")
    company: Optional[str] = Field(None, description="Filter by company name")
    filing_type: Optional[str] = Field(None, description="Filter by filing type (10-K, 10-Q)")
    top_k: Optional[int] = Field(10, ge=1, le=50, description="Number of chunks to retrieve")


class SourceInfo(BaseModel):
    """Source information for a response."""
    index: int
    company: Optional[str]
    filing_type: Optional[str]
    filing_date: Optional[str]
    section: Optional[str]
    similarity: float


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    question: str
    answer: str
    sources: List[SourceInfo]
    confidence: float
    response_time_ms: int


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query financial filings using RAG.
    
    - **question**: Your question about the financial filings
    - **company**: Optional filter by company (e.g., "Apple Inc.", "Tesla Inc.")
    - **filing_type**: Optional filter by filing type (10-K, 10-Q)
    - **top_k**: Number of context chunks to retrieve (default: 10)
    """
    try:
        logger.info(f"Processing query: {request.question[:100]}...")
        
        pipeline = get_pipeline()
        response = pipeline.query(
            question=request.question,
            company=request.company,
            filing_type=request.filing_type
        )
        
        return QueryResponse(
            question=response.query,
            answer=response.answer,
            sources=[SourceInfo(**s) for s in response.sources],
            confidence=response.confidence,
            response_time_ms=response.response_time_ms
        )
        
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/companies")
async def list_companies():
    """Get list of available companies in the database."""
    try:
        pipeline = get_pipeline()
        companies = pipeline.get_companies()
        return {"companies": companies}
    except Exception as e:
        logger.error(f"Error fetching companies: {e}")
        raise HTTPException(status_code=500, detail=str(e))
