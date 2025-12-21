from fastapi import APIRouter
from datetime import datetime
from src.models.response_models import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify the API is running
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now()
    )