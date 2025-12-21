# Docusaurus RAG Chatbot

A RAG (Retrieval-Augmented Generation) chatbot for Docusaurus documentation sites. This system allows users to ask questions about documentation content and receive accurate answers based on the documentation, with proper citations to source documents.

## Features

- **Documentation Q&A**: Ask questions about your Docusaurus documentation and get accurate answers
- **Selected Text Context**: Ask questions about specific text you've selected on the page
- **Source Citations**: Responses include citations to the source documentation
- **Floating Chat Interface**: Unobtrusive chat interface that appears as a floating button
- **Easy Docusaurus Integration**: Simple integration with existing Docusaurus sites

## Tech Stack

- **Backend**: FastAPI, Python
- **Frontend**: React, TypeScript
- **Vector Storage**: Qdrant Cloud
- **Metadata Storage**: Neon Serverless Postgres
- **Embeddings**: Cohere API
- **Generation**: OpenRouter API
- **Documentation Platform**: Docusaurus

## Architecture

The system consists of:

1. **Backend API**: FastAPI server handling document ingestion, RAG logic, and chat endpoints
2. **Vector Database**: Qdrant for storing document embeddings and enabling semantic search
3. **Metadata Database**: Neon Postgres for storing document metadata
4. **Frontend Component**: React chatbot component that integrates with Docusaurus
5. **RAG Pipeline**: Retrieval-Augmented Generation system using Cohere embeddings and OpenRouter LLM

## Setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- Access to Cohere API and OpenRouter API
- Qdrant Cloud account
- Neon Postgres account

### Backend Setup

1. Install Python dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. Set up environment variables in `.env`:
   ```env
   COHERE_API_KEY=your_cohere_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=mistralai/devstral-2512:free
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
   QDRANT_API_KEY=your_qdrant_api_key
   QDRANT_URL=your_qdrant_cloud_url
   NEON_DATABASE_URL=your_neon_postgres_connection_string
   ```

3. Start the backend server:
   ```bash
   uvicorn src.api.main:app --reload
   ```

### Frontend Setup

1. Install frontend dependencies:
   ```bash
   cd frontend
   npm install
   ```

2. Start the development server:
   ```bash
   npm start
   ```

## Usage

### Document Ingestion

To ingest documentation into the system:

```bash
curl -X POST http://localhost:8000/api/ingest \
  -H "Content-Type: application/json" \
  -d '{"directory_path": "/path/to/your/docusaurus/docs"}'
```

### Chat API

To ask questions about the documentation:

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Your question here", "selected_text": "Optional selected text context"}'
```

### Frontend Integration

To integrate the chatbot into your Docusaurus site, add the chatbot component to your theme:

```jsx
import Chatbot from './src/theme/Chatbot';

// Use the Chatbot component in your layout
<Chatbot />
```

## API Endpoints

- `GET /api/health` - Health check
- `POST /api/inggest` - Ingest documentation files
- `POST /api/chat` - Chat with the documentation assistant

## Configuration

- `QDRANT_COLLECTION_NAME`: Name of the Qdrant collection (default: "documents")
- `EMBEDDING_MODEL`: Cohere embedding model (default: "embed-english-v3.0")
- `MAX_DOC_SIZE`: Maximum document size in bytes (default: 10485760 = 10MB)
- `BACKEND_PORT`: Backend server port (default: 8000)

## Development

The project follows a modular architecture:

- `backend/src/models/` - Database models
- `backend/src/services/` - Business logic services
- `backend/src/api/` - API routes and controllers
- `backend/src/config/` - Configuration and settings
- `backend/src/utils/` - Utility functions
- `frontend/src/components/` - React components
- `frontend/src/services/` - API service layer
- `frontend/src/hooks/` - React hooks

## Original Docusaurus Project Info

This website is built using [Docusaurus](https://docusaurus.io/), a modern static website generator.

### Installation

```bash
yarn
```

### Local Development

```bash
yarn start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

### Build

```bash
yarn build
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.

### Deployment

Using SSH:

```bash
USE_SSH=true yarn deploy
```

Not using SSH:

```bash
GIT_USER=<Your GitHub username> yarn deploy
```

If you are using GitHub pages for hosting, this command is a convenient way to build the website and push to the `gh-pages` branch.
