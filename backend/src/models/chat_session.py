from sqlalchemy import Column, String, DateTime, Text
from sqlalchemy.sql import func
from src.config.database import Base


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    session_id = Column(String, primary_key=True, index=True)
    selected_text = Column(Text, nullable=True)  # Optional, text selected by user for context
    created_date = Column(DateTime(timezone=True), server_default=func.now())
    updated_date = Column(DateTime(timezone=True), onupdate=func.now())