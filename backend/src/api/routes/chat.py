from fastapi import APIRouter, Depends
from typing import Optional
import uuid
from src.models.request_models import ChatRequest
from src.models.response_models import ChatResponse
from src.services.rag_service import RAGService
from src.config.database import get_db
from sqlalchemy.orm import Session

router = APIRouter()
rag_service = RAGService()


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    """
    Process chat queries with RAG to provide context-aware responses
    """
    # Generate a session ID if not provided
    session_id = request.session_id or str(uuid.uuid4())

    # Generate response using RAG service
    result = rag_service.generate_response(
        query=request.query,
        selected_text=request.selected_text
    )

    return ChatResponse(
        response=result["response"],
        sources=result["sources"],
        session_id=session_id
    )