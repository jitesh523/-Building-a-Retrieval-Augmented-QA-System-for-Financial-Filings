"""Vector database operations for embedding storage and retrieval."""
import psycopg2
from psycopg2.extras import execute_values
import numpy as np
from typing import List, Dict, Optional, Tuple
import json
import struct


class VectorDB:
    """PostgreSQL vector database for embedding storage."""
    
    def __init__(self, connection_string: str):
        """Initialize with database connection string."""
        self.connection_string = connection_string
    
    def _get_connection(self):
        """Get database connection."""
        return psycopg2.connect(self.connection_string)
    
    def _embedding_to_bytes(self, embedding: List[float]) -> bytes:
        """Convert embedding to bytes for storage."""
        return struct.pack(f'{len(embedding)}f', *embedding)
    
    def _bytes_to_embedding(self, data: bytes) -> List[float]:
        """Convert bytes back to embedding."""
        n = len(data) // 4  # 4 bytes per float
        return list(struct.unpack(f'{n}f', data))
    
    def store_embeddings(self, chunk_ids: List[int], embeddings: List[List[float]], 
                         model_name: str = "local") -> int:
        """
        Store embeddings for chunks.
        
        Args:
            chunk_ids: List of chunk IDs
            embeddings: List of embedding vectors
            model_name: Name of the embedding model
            
        Returns:
            Number of embeddings stored
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            # Convert embeddings to bytes
            data = [
                (chunk_id, self._embedding_to_bytes(emb), model_name)
                for chunk_id, emb in zip(chunk_ids, embeddings)
            ]
            
            # Insert embeddings
            execute_values(
                cursor,
                """
                INSERT INTO embeddings (chunk_id, embedding_data, model_name)
                VALUES %s
                ON CONFLICT (chunk_id, model_name) DO UPDATE
                SET embedding_data = EXCLUDED.embedding_data
                """,
                data
            )
            
            conn.commit()
            return len(data)
            
        finally:
            cursor.close()
            conn.close()
    
    def get_chunks_without_embeddings(self, model_name: str = "local", 
                                       limit: int = None) -> List[Tuple[int, str]]:
        """
        Get chunks that don't have embeddings yet.
        
        Args:
            model_name: Embedding model name
            limit: Maximum number of chunks to return
            
        Returns:
            List of (chunk_id, chunk_text) tuples
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            query = """
                SELECT c.id, c.chunk_text
                FROM chunks c
                LEFT JOIN embeddings e ON c.id = e.chunk_id AND e.model_name = %s
                WHERE e.id IS NULL
            """
            if limit:
                query += f" LIMIT {limit}"
            
            cursor.execute(query, (model_name,))
            return cursor.fetchall()
            
        finally:
            cursor.close()
            conn.close()
    
    def search_similar(self, query_embedding: List[float], 
                       top_k: int = 10,
                       model_name: str = "local",
                       company: str = None,
                       filing_type: str = None) -> List[Dict]:
        """
        Search for similar chunks using cosine similarity.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            model_name: Embedding model name
            company: Optional filter by company
            filing_type: Optional filter by filing type
            
        Returns:
            List of matching chunks with metadata
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            # Get all embeddings (we'll compute similarity in Python for now)
            query = """
                SELECT e.chunk_id, e.embedding_data, c.chunk_text, 
                       c.section, c.page_number, d.company, d.filing_type, d.filing_date
                FROM embeddings e
                JOIN chunks c ON e.chunk_id = c.id
                JOIN documents d ON c.document_id = d.id
                WHERE e.model_name = %s
            """
            params = [model_name]
            
            if company:
                query += " AND d.company = %s"
                params.append(company)
            
            if filing_type:
                query += " AND d.filing_type = %s"
                params.append(filing_type)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            if not rows:
                return []
            
            # Compute cosine similarity
            query_np = np.array(query_embedding)
            query_norm = np.linalg.norm(query_np)
            
            results = []
            for row in rows:
                chunk_id, emb_data, chunk_text, section, page_num, comp, ftype, fdate = row
                
                emb = np.array(self._bytes_to_embedding(bytes(emb_data)))
                emb_norm = np.linalg.norm(emb)
                
                if emb_norm > 0 and query_norm > 0:
                    similarity = float(np.dot(query_np, emb) / (query_norm * emb_norm))
                else:
                    similarity = 0.0
                
                results.append({
                    'chunk_id': chunk_id,
                    'chunk_text': chunk_text,
                    'section': section,
                    'page_number': page_num,
                    'company': comp,
                    'filing_type': ftype,
                    'filing_date': str(fdate),
                    'similarity': similarity
                })
            
            # Sort by similarity and return top_k
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:top_k]
            
        finally:
            cursor.close()
            conn.close()
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT COUNT(*) FROM documents")
            doc_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM chunks")
            chunk_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM embeddings")
            emb_count = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT company, COUNT(*) as cnt
                FROM documents
                GROUP BY company
            """)
            companies = {row[0]: row[1] for row in cursor.fetchall()}
            
            return {
                'documents': doc_count,
                'chunks': chunk_count,
                'embeddings': emb_count,
                'companies': companies
            }
            
        finally:
            cursor.close()
            conn.close()
