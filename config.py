"""
Configuration management for the RAG Assistant project.
"""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings and configuration."""
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    model_name: str = Field(default="gpt-3.5-turbo", env="MODEL_NAME")
    temperature: float = Field(default=0.7, env="TEMPERATURE")
    max_tokens: int = Field(default=1000, env="MAX_TOKENS")
    
    # Vector Store Configuration
    vector_store_type: str = Field(default="faiss", env="VECTOR_STORE_TYPE")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    
    # Retrieval Configuration
    top_k: int = Field(default=5, env="TOP_K")
    similarity_threshold: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")
    
    # Memory Configuration
    enable_memory: bool = Field(default=True, env="ENABLE_MEMORY")
    memory_window: int = Field(default=10, env="MEMORY_WINDOW")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    enable_logging: bool = Field(default=True, env="ENABLE_LOGGING")
    
    # File Paths
    data_dir: str = Field(default="data", env="DATA_DIR")
    vector_store_path: str = Field(default="vector_store", env="VECTOR_STORE_PATH")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


def validate_config() -> bool:
    """Validate the configuration and return True if valid."""
    try:
        # Check if OpenAI API key is set
        if not settings.openai_api_key:
            print("❌ OPENAI_API_KEY is required. Please set it in your .env file.")
            return False
        
        # Validate vector store type
        if settings.vector_store_type not in ["faiss", "chroma"]:
            print("❌ VECTOR_STORE_TYPE must be either 'faiss' or 'chroma'")
            return False
        
        # Validate similarity threshold
        if not 0 <= settings.similarity_threshold <= 1:
            print("❌ SIMILARITY_THRESHOLD must be between 0 and 1")
            return False
        
        print("✅ Configuration validated successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False


if __name__ == "__main__":
    # Test configuration
    print("🔧 RAG Assistant Configuration")
    print("=" * 40)
    print(f"Model: {settings.model_name}")
    print(f"Vector Store: {settings.vector_store_type}")
    print(f"Top K: {settings.top_k}")
    print(f"Temperature: {settings.temperature}")
    print(f"Memory Enabled: {settings.enable_memory}")
    print("=" * 40)
    
    validate_config()
