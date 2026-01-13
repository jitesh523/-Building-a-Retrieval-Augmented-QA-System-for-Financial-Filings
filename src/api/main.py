"""FastAPI application for Financial RAG QA System."""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .routes import query, documents, health

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("🚀 Starting Financial RAG QA System...")
    yield
    logger.info("👋 Shutting down Financial RAG QA System...")


# Create FastAPI app
app = FastAPI(
    title="Financial RAG QA System",
    description="Retrieval-Augmented Generation system for querying SEC financial filings",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(query.router, prefix="/api", tags=["Query"])
app.include_router(documents.router, prefix="/api", tags=["Documents"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Financial RAG QA System",
        "docs": "/docs",
        "health": "/health"
    }


# For running with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
