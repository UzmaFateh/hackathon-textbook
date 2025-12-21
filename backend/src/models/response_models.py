from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime


class IngestionResponse(BaseModel):
    status: str
    processed_files: int
    errors: List[str]


class ChatResponse(BaseModel):
    response: str
    sources: List[str]
    session_id: str


class DocumentMetadataResponse(BaseModel):
    document_id: str
    title: str
    path: str
    content_hash: str
    created_date: datetime
    updated_date: Optional[datetime] = None