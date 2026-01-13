"""Script to generate embeddings for all chunks."""
import sys
from pathlib import Path
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.embeddings.generator import EmbeddingGenerator
from src.database.vector_db import VectorDB


def generate_embeddings(provider: str = "local", batch_size: int = 32):
    """Generate embeddings for all chunks without embeddings."""
    print(f"🔧 Generating embeddings using provider: {provider}")
    
    # Initialize components
    embedder = EmbeddingGenerator(provider=provider)
    vector_db = VectorDB(settings.database_url)
    
    print(f"   Embedding dimension: {embedder.dimension}")
    
    # Get chunks without embeddings
    model_name = provider
    chunks = vector_db.get_chunks_without_embeddings(model_name=model_name)
    
    if not chunks:
        print("✅ All chunks already have embeddings!")
        return
    
    print(f"📝 Found {len(chunks)} chunks to embed")
    
    # Process in batches
    total_embedded = 0
    
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding batches"):
        batch = chunks[i:i + batch_size]
        chunk_ids = [c[0] for c in batch]
        chunk_texts = [c[1] for c in batch]
        
        # Generate embeddings
        embeddings = embedder.embed(chunk_texts)
        
        # Store in database
        stored = vector_db.store_embeddings(chunk_ids, embeddings, model_name=model_name)
        total_embedded += stored
    
    print(f"\n✅ Successfully generated {total_embedded} embeddings")
    
    # Print stats
    stats = vector_db.get_stats()
    print(f"\n📊 Database Stats:")
    print(f"   Documents: {stats['documents']}")
    print(f"   Chunks: {stats['chunks']}")
    print(f"   Embeddings: {stats['embeddings']}")


if __name__ == "__main__":
    # Use local embeddings by default (doesn't require API key)
    generate_embeddings(provider="local", batch_size=32)
