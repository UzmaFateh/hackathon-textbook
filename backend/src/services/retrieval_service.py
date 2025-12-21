from qdrant_client import models
from src.config.qdrant_config import qdrant_client, settings
from src.services.embedding_service import EmbeddingService
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class RetrievalService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.collection_name = settings.qdrant_collection_name

    def search_similar(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents to the query in the vector database
        """
        try:
            # Generate embedding for the query
            query_embedding = self.embedding_service.generate_single_embedding(query)

            # Search in Qdrant
            search_results = qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                with_payload=True
            )

            # Format results
            results = []
            for hit in search_results:
                results.append({
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload
                })

            return results
        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            raise

    def get_document_content(self, document_id: str) -> str:
        """
        Retrieve the content of a specific document by ID
        """
        try:
            # Search for points with the given document_id in payload
            results = qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id)
                        )
                    ]
                ),
                limit=1
            )

            if results[0]:
                return results[0].payload.get("content", "")
            return ""
        except Exception as e:
            logger.error(f"Error retrieving document content: {str(e)}")
            return ""