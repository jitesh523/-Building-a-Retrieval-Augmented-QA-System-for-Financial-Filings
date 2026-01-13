"""Database setup and initialization."""
import psycopg2
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import settings


def setup_database():
    """Initialize the database schema."""
    print("🔧 Setting up database...")
    
    # Connect to database
    conn = psycopg2.connect(settings.database_url)
    conn.autocommit = True
    cursor = conn.cursor()
    
    try:
        # Read and execute schema
        schema_path = Path(__file__).parent / "schema.sql"
        with open(schema_path, 'r') as f:
            schema_sql = f.read()
        
        cursor.execute(schema_sql)
        print("✅ Schema created successfully")
        
        # Verify tables
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_type = 'BASE TABLE';
        """)
        tables = cursor.fetchall()
        print(f"✅ Created {len(tables)} tables: {', '.join([t[0] for t in tables])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error setting up database: {e}")
        return False
    
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    success = setup_database()
    sys.exit(0 if success else 1)
