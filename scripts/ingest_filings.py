"""Script to ingest financial filings into the database."""
import sys
from pathlib import Path
import psycopg2
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.data.parsers import DocumentParser
from src.data.chunker import DocumentChunker


def extract_metadata_from_filename(filename: str) -> dict:
    """Extract company and metadata from filename."""
    name = Path(filename).stem.lower()
    
    # Map filenames to companies and types
    company_map = {
        'aapl': ('Apple Inc.', '10-K', '2025-09-27'),
        'meta': ('Meta Platforms Inc.', '10-K', '2024-12-31'),
        'tesla': ('Tesla Inc.', '10-K', '2024-12-31'),
        'google': ('Alphabet Inc.', '10-K', '2024-12-31'),
        'netflix': ('Netflix Inc.', '10-K', '2024-12-31'),
        'warnerbros': ('Warner Bros. Discovery Inc.', '10-K', '2024-12-31'),
    }
    
    for key, (company, filing_type, filing_date) in company_map.items():
        if key in name:
            return {
                'company': company,
                'filing_type': filing_type,
                'filing_date': filing_date
            }
    
    # Default
    return {
        'company': 'Unknown',
        'filing_type': '10-K',
        'filing_date': datetime.now().strftime('%Y-%m-%d')
    }


def ingest_document(file_path: str):
    """Ingest a single document into the database."""
    print(f"\n📄 Processing: {file_path}")
    
    # Parse document
    parser = DocumentParser()
    parsed = parser.parse(file_path)
    
    # Extract metadata
    metadata = extract_metadata_from_filename(file_path)
    metadata.update(parsed['metadata'])
    
    print(f"✓ Company: {metadata['company']}")
    print(f"✓ Filing Type: {metadata['filing_type']}")
    
    # Chunk document
    chunker = DocumentChunker(strategy='fixed', chunk_size=500, overlap=100)
    chunks = chunker.chunk_document(parsed)
    
    print(f"✓ Generated {len(chunks)} chunks")
    
    # Connect to database
    conn = psycopg2.connect(settings.database_url)
    cursor = conn.cursor()
    
    try:
        # Insert document
        cursor.execute("""
            INSERT INTO documents (company, filing_type, filing_date, file_path, file_type, metadata)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (company, filing_type, filing_date) DO UPDATE
            SET file_path = EXCLUDED.file_path,
                processed_at = CURRENT_TIMESTAMP
            RETURNING id;
        """, (
            metadata['company'],
            metadata['filing_type'],
            metadata['filing_date'],
            file_path,
            metadata['file_type'],
            '{}',
        ))
        
        doc_id = cursor.fetchone()[0]
        print(f"✓ Document ID: {doc_id}")
        
        # Insert chunks
        for chunk in chunks:
            cursor.execute("""
                INSERT INTO chunks (document_id, chunk_text, chunk_index, section, page_number, token_count, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (document_id, chunk_index) DO UPDATE
                SET chunk_text = EXCLUDED.chunk_text,
                    token_count = EXCLUDED.token_count;
            """, (
                doc_id,
                chunk.text,
                chunk.index,
                chunk.section,
                chunk.page_number,
                chunk.token_count,
                '{}',
            ))
        
        conn.commit()
        print(f"✅ Successfully ingested {len(chunks)} chunks")
        
    except Exception as e:
        conn.rollback()
        print(f"❌ Error: {e}")
        raise
    finally:
        cursor.close()
        conn.close()


def main():
    """Ingest all financial filings."""
    project_root = Path(__file__).parent.parent
    
    # List of files to ingest
    files = [
        project_root / "aapl-20250927.pdf",
        project_root / "meta-20241231.pdf",
        project_root / "tesla.pdf",
        project_root / "google.html",
        project_root / "netflix.html",
        project_root / "warnerbros.html",
    ]
    
    print("🚀 Starting document ingestion...")
    
    for file_path in files:
        if file_path.exists():
            try:
                ingest_document(str(file_path))
            except Exception as e:
                print(f"❌ Failed to ingest {file_path.name}: {e}")
                continue
        else:
            print(f"⚠️  File not found: {file_path}")
    
    print("\n✅ Ingestion complete!")


if __name__ == "__main__":
    main()
