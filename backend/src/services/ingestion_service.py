from src.services.embedding_service import EmbeddingService
from src.config.qdrant_config import qdrant_client, settings, VECTOR_SIZE
from src.config.database import SessionLocal
from src.models.document_metadata import DocumentMetadata
from src.utils.markdown_parser import MarkdownParser
from src.utils.chunking_utils import chunk_text
import hashlib
import logging
import os
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class IngestionService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.markdown_parser = MarkdownParser()
        self.collection_name = settings.qdrant_collection_name

    def process_document_directory(self, directory_path: str) -> Dict[str, Any]:
        """
        Process all markdown files in a directory and ingest them into the system
        """
        processed_files = 0
        errors = []

        # Get all markdown files in the directory
        markdown_files = []
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith(('.md', '.mdx')):
                    markdown_files.append(os.path.join(root, file))

        for file_path in markdown_files:
            try:
                self.ingest_single_document(file_path)
                processed_files += 1
                logger.info(f"Successfully processed: {file_path}")
            except Exception as e:
                error_msg = f"Error processing {file_path}: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)

        return {
            "status": "completed" if not errors else "partial",
            "processed_files": processed_files,
            "errors": errors
        }

    def ingest_single_document(self, file_path: str):
        """
        Ingest a single markdown document
        """
        # Read the file content
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Calculate content hash
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        # Extract title and relative path
        title = self._extract_title(content) or os.path.basename(file_path)
        # Calculate relative path from the current working directory
        relative_path = os.path.relpath(file_path)

        # Check if document already exists and hasn't changed
        db = SessionLocal()
        try:
            existing_doc = db.query(DocumentMetadata).filter(
                DocumentMetadata.path == relative_path
            ).first()

            if existing_doc and existing_doc.content_hash == content_hash:
                logger.info(f"Document {relative_path} unchanged, skipping...")
                return  # Document hasn't changed, no need to reprocess

            # Parse the markdown content
            parsed_content = self.markdown_parser.parse(content)

            # Chunk the content
            chunks = chunk_text(parsed_content, max_chunk_size=1000, overlap=200)

            # Generate embeddings for each chunk
            chunk_texts = [chunk['content'] for chunk in chunks]
            embeddings = self.embedding_service.generate_embeddings(chunk_texts)

            # Create document metadata entry
            if existing_doc:
                # Update existing document
                existing_doc.title = title
                existing_doc.content_hash = content_hash
                existing_doc.updated_date = None  # This will be updated by the database
                doc_metadata = existing_doc
            else:
                # Create new document
                doc_metadata = DocumentMetadata(
                    document_id=f"doc_{hash(content_hash)}",
                    title=title,
                    path=relative_path,
                    content_hash=content_hash
                )
                db.add(doc_metadata)

            db.commit()

            # Store each chunk in Qdrant with its embedding
            import uuid
            for i, (chunk_data, embedding) in enumerate(zip(chunks, embeddings)):
                # Use UUID for Qdrant point ID as it expects either unsigned integer or UUID
                chunk_id = str(uuid.uuid4())

                # Store in Qdrant
                qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=[
                        {
                            "id": chunk_id,
                            "vector": embedding,
                            "payload": {
                                "content": chunk_data['content'],
                                "document_id": doc_metadata.document_id,
                                "title": title,
                                "path": relative_path,
                                "chunk_index": i
                            }
                        }
                    ]
                )

        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()

    def _extract_title(self, content: str) -> str:
        """
        Extract title from markdown content (look for # Title pattern)
        """
        lines = content.split('\n')
        for line in lines[:10]:  # Check first 10 lines
            if line.strip().startswith('# '):
                return line.strip()[2:]  # Remove '# ' prefix
        return ""

    def delete_document(self, document_path: str):
        """
        Delete a document and its chunks from the system
        """
        db = SessionLocal()
        try:
            # Find the document in database
            doc_metadata = db.query(DocumentMetadata).filter(
                DocumentMetadata.path == document_path
            ).first()

            if not doc_metadata:
                raise ValueError(f"Document {document_path} not found")

            # Find and delete all chunks from Qdrant
            search_results = qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter={
                    "must": [
                        {
                            "key": "document_id",
                            "match": {
                                "value": doc_metadata.document_id
                            }
                        }
                    ]
                }
            )

            chunk_ids = [point.id for point in search_results[0]]
            if chunk_ids:
                qdrant_client.delete(
                    collection_name=self.collection_name,
                    points_selector=chunk_ids
                )

            # Delete from database
            db.delete(doc_metadata)
            db.commit()

        finally:
            db.close()