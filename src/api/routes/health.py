"""Health check endpoints."""
from fastapi import APIRouter
from src.database.vector_db import VectorDB
from src.config import settings

router = APIRouter()


@router.get("/health")
async def health_check():
    """Check API health status."""
    return {"status": "healthy"}


@router.get("/health/db")
async def database_health():
    """Check database connection and stats."""
    try:
        vector_db = VectorDB(settings.database_url)
        stats = vector_db.get_stats()
        
        return {
            "status": "healthy",
            "database": "connected",
            "stats": stats
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e)
        }
