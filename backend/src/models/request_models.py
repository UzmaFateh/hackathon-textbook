from pydantic import BaseModel
from typing import Optional


class ChatRequest(BaseModel):
    query: str
    selected_text: Optional[str] = None
    session_id: Optional[str] = None


class IngestionRequest(BaseModel):
    directory_path: str


class DocumentListResponse(BaseModel):
    documents: list