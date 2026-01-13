"""Document management endpoints."""
from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import tempfile
import os
import logging

from src.database.vector_db import VectorDB
from src.data.parsers import DocumentParser
from src.data.chunker import DocumentChunker
from src.embeddings.generator import EmbeddingGenerator
from src.config import settings
import psycopg2

router = APIRouter()
logger = logging.getLogger(__name__)


class DocumentInfo(BaseModel):
    """Document information model."""
    id: int
    company: str
    filing_type: str
    filing_date: str
    file_path: str
    file_type: str


class DocumentStats(BaseModel):
    """Database statistics model."""
    documents: int
    chunks: int
    embeddings: int
    companies: dict


@router.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    """Get list of all documents in the database."""
    try:
        conn = psycopg2.connect(settings.database_url)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, company, filing_type, filing_date, file_path, file_type
            FROM documents
            ORDER BY company, filing_date DESC
        """)
        
        documents = []
        for row in cursor.fetchall():
            documents.append(DocumentInfo(
                id=row[0],
                company=row[1],
                filing_type=row[2],
                filing_date=str(row[3]),
                file_path=row[4],
                file_type=row[5]
            ))
        
        cursor.close()
        conn.close()
        
        return documents
        
    except Exception as e:
        logger.error(f"Error fetching documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{doc_id}")
async def get_document(doc_id: int):
    """Get details of a specific document."""
    try:
        conn = psycopg2.connect(settings.database_url)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT d.id, d.company, d.filing_type, d.filing_date, d.file_path, d.file_type,
                   COUNT(c.id) as chunk_count
            FROM documents d
            LEFT JOIN chunks c ON d.id = c.document_id
            WHERE d.id = %s
            GROUP BY d.id
        """, (doc_id,))
        
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Document not found")
        
        cursor.close()
        conn.close()
        
        return {
            "id": row[0],
            "company": row[1],
            "filing_type": row[2],
            "filing_date": str(row[3]),
            "file_path": row[4],
            "file_type": row[5],
            "chunk_count": row[6]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=DocumentStats)
async def get_stats():
    """Get database statistics."""
    try:
        vector_db = VectorDB(settings.database_url)
        stats = vector_db.get_stats()
        
        return DocumentStats(
            documents=stats['documents'],
            chunks=stats['chunks'],
            embeddings=stats['embeddings'],
            companies=stats['companies']
        )
        
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_document_async(file_path: str, company: str, 
                                  filing_type: str, filing_date: str):
    """Process a document in the background."""
    try:
        logger.info(f"Processing document: {file_path}")
        
        # Parse document
        parser = DocumentParser()
        parsed = parser.parse(file_path)
        
        # Chunk document
        chunker = DocumentChunker(strategy='fixed', chunk_size=500, overlap=100)
        chunks = chunker.chunk_document(parsed)
        
        # Store in database
        conn = psycopg2.connect(settings.database_url)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO documents (company, filing_type, filing_date, file_path, file_type)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        """, (company, filing_type, filing_date, file_path, parsed['metadata']['file_type']))
        
        doc_id = cursor.fetchone()[0]
        
        # Insert chunks
        for chunk in chunks:
            cursor.execute("""
                INSERT INTO chunks (document_id, chunk_text, chunk_index, section, page_number, token_count)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (doc_id, chunk.text, chunk.index, chunk.section, chunk.page_number, chunk.token_count))
        
        conn.commit()
        
        # Generate embeddings
        embedder = EmbeddingGenerator(provider="local")
        vector_db = VectorDB(settings.database_url)
        
        chunk_ids = []
        chunk_texts = []
        
        cursor.execute("SELECT id, chunk_text FROM chunks WHERE document_id = %s", (doc_id,))
        for row in cursor.fetchall():
            chunk_ids.append(row[0])
            chunk_texts.append(row[1])
        
        embeddings = embedder.embed(chunk_texts)
        vector_db.store_embeddings(chunk_ids, embeddings, model_name="local")
        
        cursor.close()
        conn.close()
        
        logger.info(f"Successfully processed document: {company} - {len(chunks)} chunks")
        
    except Exception as e:
        logger.error(f"Error processing document: {e}")


@router.post("/ingest")
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    company: str = "Unknown",
    filing_type: str = "10-K",
    filing_date: str = "2024-01-01"
):
    """
    Ingest a new financial filing.
    
    Upload a PDF or HTML file to be processed and added to the database.
    Processing happens in the background.
    """
    try:
        # Save uploaded file
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Process in background
        background_tasks.add_task(
            process_document_async,
            tmp_path, company, filing_type, filing_date
        )
        
        return {
            "message": "Document uploaded and processing started",
            "filename": file.filename,
            "company": company,
            "filing_type": filing_type
        }
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
