"""Streamlit frontend for Financial RAG QA System."""
import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.pipeline import RAGPipeline
from src.database.vector_db import VectorDB
from src.config import settings

# Page configuration
st.set_page_config(
    page_title="Financial RAG QA System",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .source-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .confidence-badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: bold;
    }
    .stTextArea textarea {
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline():
    """Initialize and cache the RAG pipeline."""
    return RAGPipeline(
        embedding_provider="local",
        llm_provider="local",
        top_k=10,
        use_hybrid=True
    )


@st.cache_resource
def get_vector_db():
    """Initialize and cache the vector database."""
    return VectorDB(settings.database_url)


def get_db_stats():
    """Get database statistics."""
    try:
        db = get_vector_db()
        return db.get_stats()
    except Exception as e:
        return {"error": str(e)}


def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Financial RAG QA System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask questions about SEC financial filings (10-K, 10-Q)</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Database stats
        stats = get_db_stats()
        if "error" not in stats:
            st.metric("📄 Documents", stats.get('documents', 0))
            st.metric("🔖 Chunks", stats.get('chunks', 0))
            st.metric("🔢 Embeddings", stats.get('embeddings', 0))
            
            # Company filter
            companies = list(stats.get('companies', {}).keys())
            if companies:
                company_filter = st.selectbox(
                    "Filter by Company",
                    ["All Companies"] + companies
                )
            else:
                company_filter = "All Companies"
        else:
            st.error(f"Database error: {stats['error']}")
            company_filter = "All Companies"
        
        st.divider()
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            top_k = st.slider("Number of sources", 3, 20, 10)
            show_sources = st.checkbox("Show source details", value=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Query input
        query = st.text_area(
            "Ask a question about financial filings:",
            placeholder="Example: What was Apple's revenue in fiscal 2025? What are the main risk factors for Tesla?",
            height=100
        )
        
        # Submit button
        submit = st.button("🔍 Search", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("### Quick Questions")
        quick_questions = [
            "What was Apple's total revenue?",
            "What are Tesla's main risk factors?",
            "Describe Meta's business model",
            "What are Netflix's content costs?",
        ]
        for q in quick_questions:
            if st.button(q, key=f"quick_{q[:20]}"):
                query = q
                submit = True
    
    # Process query
    if submit and query:
        with st.spinner("🔄 Searching financial filings..."):
            try:
                pipeline = get_pipeline()
                
                # Apply company filter
                company = None if company_filter == "All Companies" else company_filter
                
                response = pipeline.query(
                    question=query,
                    company=company
                )
                
                # Display answer
                st.markdown("---")
                st.markdown("### 💡 Answer")
                st.markdown(response.answer)
                
                # Confidence and timing
                col_conf, col_time = st.columns(2)
                with col_conf:
                    confidence_pct = int(response.confidence * 100)
                    color = "green" if confidence_pct > 70 else "orange" if confidence_pct > 50 else "red"
                    st.markdown(f"**Confidence:** <span style='color:{color}'>{confidence_pct}%</span>", unsafe_allow_html=True)
                with col_time:
                    st.markdown(f"**Response time:** {response.response_time_ms}ms")
                
                # Sources
                if show_sources and response.sources:
                    st.markdown("---")
                    st.markdown("### 📚 Sources")
                    
                    for source in response.sources:
                        with st.expander(f"[{source['index']}] {source.get('company', 'Unknown')} - {source.get('filing_type', '')} ({source.get('filing_date', '')})"):
                            st.markdown(f"**Section:** {source.get('section', 'N/A')}")
                            st.markdown(f"**Similarity:** {source.get('similarity', 0):.2%}")
                            
                            # Show chunk text if available
                            matching_chunk = next(
                                (c for c in response.retrieved_chunks if c.get('chunk_id') == source.get('chunk_id')),
                                None
                            )
                            if matching_chunk:
                                st.text(matching_chunk.get('chunk_text', '')[:500] + "...")
                
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """<div style='text-align: center; color: #888;'>
        Financial RAG QA System | Built with Streamlit, FastAPI, and PostgreSQL
        </div>""",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
