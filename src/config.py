"""Configuration management for the RAG system."""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    # Database
    database_url: str = "postgresql://localhost:5432/financial_rag"
    
    # OpenAI
    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-ada-002"
    openai_llm_model: str = "gpt-4-turbo-preview"
    
    # Cohere
    cohere_api_key: str = ""
    
    # Gemini
    gemini_api_key: str = ""
    
    # Embedding Configuration
    embedding_provider: Literal["openai", "cohere", "local"] = "openai"
    embedding_dimension: int = 1536
    
    # RAG Configuration
    chunk_size: int = 500
    chunk_overlap: int = 100
    top_k_retrieval: int = 10
    rerank_top_k: int = 5
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Logging
    log_level: str = "INFO"


# Global settings instance
settings = Settings()
