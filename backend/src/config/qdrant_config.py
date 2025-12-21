from qdrant_client import QdrantClient
from qdrant_client.http import models
from src.config.settings import settings
import logging

logger = logging.getLogger(__name__)

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key,
    https=True
)

# Define vector dimensions based on Google's embedding-001 model
# embedding-001 produces 768-dimensional vectors
VECTOR_SIZE = 768
DISTANCE = models.Distance.COSINE


def initialize_qdrant_collection():
    """
    Initialize the Qdrant collection for document embeddings if it doesn't exist
    """
    try:
        # Check if collection exists
        collections = qdrant_client.get_collections()
        collection_names = [collection.name for collection in collections.collections]

        if settings.qdrant_collection_name not in collection_names:
            # Create the collection
            qdrant_client.create_collection(
                collection_name=settings.qdrant_collection_name,
                vectors_config=models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=DISTANCE
                )
            )
            logger.info(f"Created Qdrant collection: {settings.qdrant_collection_name}")
        else:
            logger.info(f"Qdrant collection {settings.qdrant_collection_name} already exists")
    except Exception as e:
        logger.error(f"Error initializing Qdrant collection: {str(e)}")
        raise


# Initialize the collection when this module is imported
initialize_qdrant_collection()