from fastapi import APIRouter
from src.models.request_models import IngestionRequest
from src.models.response_models import IngestionResponse
from src.services.ingestion_service import IngestionService

router = APIRouter()
ingestion_service = IngestionService()


@router.post("/ingest", response_model=IngestionResponse)
async def ingest_documents(request: IngestionRequest):
    """
    Ingest Docusaurus markdown files for RAG - supports both local directory and live site
    """
    if request.site_url:
        # Process from live site URL
        result = ingestion_service.process_from_site_url(request.site_url)
    elif request.directory_path:
        # Process from local directory
        result = ingestion_service.process_document_directory(request.directory_path)
    else:
        # Neither provided
        result = {
            "status": "error",
            "processed_files": 0,
            "errors": ["Either site_url or directory_path must be provided"]
        }

    return IngestionResponse(
        status=result["status"],
        processed_files=result["processed_files"],
        errors=result["errors"]
    )