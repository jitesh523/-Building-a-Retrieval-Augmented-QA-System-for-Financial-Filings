-- Simplified schema without pgvector extension initially
-- We'll store embeddings as bytea or text and migrate to vector later

-- Documents table: stores raw financial filings
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    company VARCHAR(100) NOT NULL,
    filing_type VARCHAR(50) NOT NULL,  -- 10-K, 10-Q, etc.
    filing_date DATE NOT NULL,
    file_path TEXT NOT NULL,
    file_type VARCHAR(20) NOT NULL,  -- PDF, HTML
    raw_text TEXT,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB,
    UNIQUE(company, filing_type, filing_date)
);

-- Chunks table: stores text chunks from documents
CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    section VARCHAR(200),  -- e.g., "Risk Factors", "Financial Statements"
    page_number INTEGER,
    token_count INTEGER,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(document_id, chunk_index)
);

-- Embeddings table: stores embeddings (as byte array for now)
CREATE TABLE IF NOT EXISTS embeddings (
    id SERIAL PRIMARY KEY,
    chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    embedding_data BYTEA,  -- Will store as bytes for now
    model_name VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(chunk_id, model_name)
);

-- Create indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_documents_company ON documents(company);
CREATE INDEX IF NOT EXISTS idx_documents_filing_type ON documents(filing_type);
CREATE INDEX IF NOT EXISTS idx_documents_filing_date ON documents(filing_date);
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_section ON chunks(section);
CREATE INDEX IF NOT EXISTS idx_embeddings_chunk_id ON embeddings(chunk_id);

-- Query history table for logging and evaluation
CREATE TABLE IF NOT EXISTS query_history (
    id SERIAL PRIMARY KEY,
    query_text TEXT NOT NULL,
    retrieved_chunk_ids INTEGER[],
    answer_text TEXT,
    sources JSONB,
    confidence_score FLOAT,
    response_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_query_history_created_at ON query_history(created_at);

-- User feedback table for continuous improvement
CREATE TABLE IF NOT EXISTS user_feedback (
    id SERIAL PRIMARY KEY,
    query_id INTEGER REFERENCES query_history(id) ON DELETE CASCADE,
    helpful BOOLEAN,
    feedback_text TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- View for easy retrieval of chunks with metadata
CREATE OR REPLACE VIEW chunk_details AS
SELECT 
    c.id as chunk_id,
    c.chunk_text,
    c.chunk_index,
    c.section,
    c.page_number,
    d.company,
    d.filing_type,
    d.filing_date,
    d.file_path
FROM chunks c
JOIN documents d ON c.document_id = d.id;
