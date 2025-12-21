from fastapi import APIRouter
from src.models.request_models import IngestionRequest
from src.models.response_models import IngestionResponse
from src.services.ingestion_service import IngestionService

router = APIRouter()
ingestion_service = IngestionService()


@router.post("/ingest", response_model=IngestionResponse)
async def ingest_documents(request: IngestionRequest):
    """
    Ingest Docusaurus markdown files for RAG
    """
    result = ingestion_service.process_document_directory(request.directory_path)
    return IngestionResponse(
        status=result["status"],
        processed_files=result["processed_files"],
        errors=result["errors"]
    )