from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # API Keys
    cohere_api_key: str
    openrouter_api_key: str
    qdrant_api_key: str
    qdrant_url: str
    neon_database_url: str = ""

    # OpenRouter Configuration
    openrouter_model: str = "mistralai/devstral-2512:free"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Configuration
    qdrant_collection_name: str = "documents"
    embedding_model: str = "embed-english-v3.0"  # Cohere embedding model
    max_doc_size: int = 10485760  # 10MB in bytes

    class Config:
        env_file = ".env"
        extra = "ignore"  # Ignore extra environment variables


settings = Settings()