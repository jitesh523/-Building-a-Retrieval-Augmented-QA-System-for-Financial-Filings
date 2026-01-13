# Financial RAG QA System

A Retrieval-Augmented Generation (RAG) system for querying SEC financial filings (10-K, 10-Q). Ask natural language questions and get accurate, source-cited answers from financial documents.

## 🚀 Features

- **Document Processing**: Parse PDF and HTML SEC filings
- **Smart Chunking**: Fixed-size and semantic chunking strategies
- **Vector Search**: Embeddings stored in PostgreSQL with similarity search
- **RAG Pipeline**: Hybrid retrieval + LLM-based answer generation
- **REST API**: FastAPI endpoints for queries and document management
- **Streamlit UI**: Interactive web interface for asking questions

## 📊 Ingested Documents

| Company | Filing Type | Chunks |
|---------|------------|--------|
| Apple Inc. | 10-K | 111 |
| Meta Platforms Inc. | 10-K | 75 |
| Tesla Inc. | 10-K | 81 |
| Alphabet Inc. | 10-K | 139 |
| Netflix Inc. | 10-K | 100 |
| Warner Bros. Discovery Inc. | 10-K | 200 |

**Total: 706 embedded chunks**

## 🛠️ Tech Stack

- **Backend**: FastAPI, PostgreSQL
- **Embeddings**: sentence-transformers (MiniLM-L6-v2)
- **Vector DB**: PostgreSQL with custom similarity search
- **Frontend**: Streamlit
- **LLM**: OpenAI GPT-4 (optional) or extractive fallback

## 📁 Project Structure

```
├── src/
│   ├── api/              # FastAPI application
│   │   ├── main.py
│   │   └── routes/
│   ├── data/             # Document parsers and chunkers
│   ├── database/         # Database schema and vector DB
│   ├── embeddings/       # Embedding generators
│   ├── rag/              # RAG pipeline (retriever, generator)
│   └── config.py
├── frontend/
│   └── app.py            # Streamlit interface
├── scripts/
│   ├── ingest_filings.py
│   └── generate_embeddings.py
├── tests/
├── requirements.txt
└── docker-compose.yml
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Start PostgreSQL

```bash
brew install postgresql@14
brew services start postgresql@14
createdb financial_rag
```

### 3. Initialize Database

```bash
python src/database/setup.py
```

### 4. Ingest Documents

```bash
python scripts/ingest_filings.py
```

### 5. Generate Embeddings

```bash
python scripts/generate_embeddings.py
```

### 6. Start the API

```bash
uvicorn src.api.main:app --reload
```

### 7. Start the Frontend

```bash
streamlit run frontend/app.py
```

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/health/db` | GET | Database status |
| `/api/query` | POST | Query financial filings |
| `/api/companies` | GET | List available companies |
| `/api/documents` | GET | List all documents |
| `/api/documents/{id}` | GET | Get document details |
| `/api/ingest` | POST | Upload new document |
| `/api/stats` | GET | Database statistics |

### Example Query

```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main risk factors for Tesla?"}'
```

## 🔧 Configuration

Create a `.env` file:

```
DATABASE_URL=postgresql://localhost:5432/financial_rag
OPENAI_API_KEY=your_key_here  # Optional, for GPT-4 answers
```

## 📈 Example Queries

- "What was Apple's total revenue in fiscal 2025?"
- "What are the main risk factors for Tesla?"
- "Describe Meta's business model"
- "What are Netflix's content costs?"
- "Summarize Warner Bros' strategic priorities"

## 📝 License

MIT License
