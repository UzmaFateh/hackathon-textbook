from sqlalchemy import Column, String, Integer, Text, DateTime, ForeignKey
from sqlalchemy.sql import func
from src.config.database import Base


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    chunk_id = Column(String, unique=True, index=True, nullable=False)
    document_id = Column(String, ForeignKey("document_metadata.document_id"), nullable=False)
    content = Column(Text, nullable=False)  # Chunked content from the document
    chunk_index = Column(Integer, nullable=False)  # Position of this chunk in the original document
    embedding_vector_id = Column(String, nullable=False)  # ID in Qdrant vector storage
    created_date = Column(DateTime(timezone=True), server_default=func.now())