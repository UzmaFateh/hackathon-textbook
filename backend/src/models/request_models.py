# from pydantic import BaseModel
# from typing import Optional


# class ChatRequest(BaseModel):
#     query: str
#     selected_text: Optional[str] = None
#     session_id: Optional[str] = None


# class IngestionRequest(BaseModel):
#     directory_path: str


# class DocumentListResponse(BaseModel):
#     documents: list

from pydantic import BaseModel
from typing import Optional


class ChatRequest(BaseModel):
    query: str
    selected_text: Optional[str] = None
    session_id: Optional[str] = None


class IngestionRequest(BaseModel):
    site_url: Optional[str] = None        # NEW: For crawling live Docusaurus site
    directory_path: Optional[str] = None  # OLD: Keep for local directory ingestion (backward compatibility)


class DocumentListResponse(BaseModel):
    documents: list