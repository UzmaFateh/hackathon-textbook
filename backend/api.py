from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import chat, ingestion, health
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Docusaurus RAG Chatbot API",
    description="API for the Docusaurus RAG chatbot system",
    version="1.0.0"
)

# Configure CORS middleware for Docusaurus frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(ingestion.router, prefix="/api", tags=["ingestion"])
app.include_router(health.router, prefix="/api", tags=["health"])

# Health check endpoint
@app.get("/")
def read_root():
    return {"status": "healthy", "message": "Docusaurus RAG Chatbot API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)