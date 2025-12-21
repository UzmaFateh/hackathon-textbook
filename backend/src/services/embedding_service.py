from src.config.llm_config import cohere_client
from src.config.settings import settings
import logging
from typing import List

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self):
        self.client = cohere_client
        self.model = settings.embedding_model

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using Cohere
        """
        try:
            response = self.client.embed(
                texts=texts,
                model=self.model,
                input_type="search_document"  # Using search_document for knowledge base content
            )
            return [embedding for embedding in response.embeddings]
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def generate_single_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        """
        return self.generate_embeddings([text])[0]