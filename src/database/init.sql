-- This script runs automatically when the PostgreSQL container starts
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Log confirmation
DO $$
BEGIN
    RAISE NOTICE 'pgvector extension enabled successfully';
END $$;
