# Quickstart Guide: Docusaurus RAG Chatbot

**Feature**: Docusaurus RAG Chatbot
**Date**: 2025-12-18
**Branch**: 001-docusaurus-rag-chatbot

## Overview

This guide provides step-by-step instructions to quickly set up and run the Docusaurus RAG chatbot. Follow these steps to get the system running in your local environment.

## Prerequisites

- Python 3.11+
- Node.js 18+ (for Docusaurus frontend)
- Access to Cohere API (for embeddings) and OpenRouter API (for generation)
- Qdrant Cloud account (or local Qdrant instance)
- Neon Postgres account (or local Postgres instance)

## Environment Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Create a `.env` file** in the root directory with the following variables:
   ```env
   # API Keys
   COHERE_API_KEY=your_cohere_api_key_here
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   OPENROUTER_MODEL=mistralai/devstral-2512:free
   OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
   QDRANT_API_KEY=your_qdrant_api_key_here
   QDRANT_HOST=your_qdrant_cloud_url
   NEON_DATABASE_URL=your_neon_postgres_connection_string

   # Optional settings
   QDRANT_COLLECTION_NAME=documents
   EMBEDDING_MODEL=embed-english-v3.0
   MAX_DOC_SIZE=10485760  # 10MB in bytes
   ```

3. **Install backend dependencies**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

4. **Install frontend dependencies**
   ```bash
   cd ../frontend
   npm install
   ```

## Backend Setup

1. **Navigate to the backend directory**
   ```bash
   cd backend
   ```

2. **Run the backend server**
   ```bash
   uvicorn src.api.main:app --reload
   ```

3. **Verify the backend is running**
   - Open http://localhost:8000/docs to see the FastAPI documentation
   - The health check endpoint should be available at http://localhost:8000/api/health

## Frontend Setup

1. **Navigate to the frontend directory**
   ```bash
   cd frontend
   ```

2. **Run the development server**
   ```bash
   npm start
   ```

3. **For Docusaurus integration**, add the chatbot component to your Docusaurus config:
   - Add the theme component to your `docusaurus.config.js`
   - The floating chat button will appear on all documentation pages

## Data Ingestion

1. **Prepare your Docusaurus documentation**
   - Ensure your docs are in the standard Docusaurus `docs/` directory structure
   - Verify that all markdown files are properly formatted

2. **Run the ingestion process**
   ```bash
   # Using the API endpoint
   curl -X POST http://localhost:8000/api/ingest \
     -H "Content-Type: application/json" \
     -d '{"directory_path": "/path/to/your/docusaurus/docs"}'
   ```

3. **Monitor the ingestion process**
   - Check the response for the number of files processed
   - Verify that documents appear in your Qdrant collection
   - Confirm metadata is stored in Neon Postgres

## Testing the Chatbot

1. **Open your Docusaurus site** in a browser
2. **Click the floating chat button** to open the chat interface
3. **Ask a question** about your documentation content
4. **Verify that responses include proper citations** to source documents

## API Endpoints

### Ingestion API
- `POST /api/ingest` - Ingest Docusaurus markdown files
  - Request: `{"directory_path": "path/to/docs"}`
  - Response: `{"status": "success", "processed_files": 5, "errors": []}`

### Chat API
- `POST /api/chat` - Process chat queries with RAG
  - Request: `{"query": "your question", "selected_text": "optional context", "session_id": "optional"}`
  - Response: `{"response": "AI response", "sources": ["doc1", "doc2"], "session_id": "session-id"}`

### Health Check
- `GET /api/health` - Check system health
  - Response: `{"status": "healthy", "timestamp": "2025-12-18T10:00:00Z"}`

## Configuration Options

### Backend Configuration
- `BACKEND_PORT`: Port for the FastAPI server (default: 8000)
- `WORKERS`: Number of uvicorn workers (default: 1 for development)
- `LOG_LEVEL`: Logging level (default: info)

### Frontend Configuration
- `REACT_APP_API_BASE_URL`: Backend API URL (default: http://localhost:8000)
- `CHATBOT_POSITION`: Position of the floating button (default: bottom-right)

## Troubleshooting

### Common Issues

1. **API Keys Not Working**
   - Verify all API keys are correctly set in the `.env` file
   - Check that API keys have the necessary permissions

2. **Document Ingestion Failing**
   - Ensure the directory path is correct and accessible
   - Verify that markdown files are properly formatted
   - Check that file sizes are within the 10MB limit

3. **Chat Responses Are Generic**
   - Verify that documents were successfully ingested
   - Check that the vector database contains your document embeddings
   - Confirm that the RAG pipeline is properly retrieving relevant content

4. **Frontend Not Connecting to Backend**
   - Ensure the backend server is running
   - Verify the API URL configuration in the frontend
   - Check CORS settings if running on different ports

## Next Steps

1. **Customize the chatbot UI** to match your site's design
2. **Fine-tune the RAG parameters** for optimal response quality
3. **Add analytics** to track chatbot usage and effectiveness
4. **Implement rate limiting** for production deployment
5. **Set up monitoring and alerting** for system health