# Implementation Plan: Docusaurus RAG Chatbot

**Branch**: `001-docusaurus-rag-chatbot` | **Date**: 2025-12-18 | **Spec**: [specs/001-docusaurus-rag-chatbot/spec.md](specs/001-docusaurus-rag-chatbot/spec.md)

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Implementation of a RAG-based chatbot for Docusaurus sites with FastAPI backend, Qdrant vector storage, and Neon Postgres for metadata. The system will ingest Docusaurus markdown files, provide a floating chat UI, and enable Q&A based on documentation content with support for selected text context. The plan includes: 1) FastAPI environment setup with API key management, 2) Data ingestion pipeline from Markdown to Cohere embeddings to Qdrant storage, 3) Backend API endpoints for RAG logic, and 4) React/Docusaurus component for chatbot UI.

## Technical Context

**Language/Version**: Python 3.11, TypeScript/JavaScript for frontend components
**Primary Dependencies**: FastAPI, Qdrant, Cohere API, Neon Postgres, React, Docusaurus
**Storage**: Qdrant Cloud (vector storage), Neon Serverless Postgres (metadata)
**Testing**: pytest for backend, Jest for frontend
**Target Platform**: Web application (Docusaurus integration)
**Project Type**: Web (backend API + frontend component)
**Performance Goals**: Response time under 5 seconds, 95% accuracy in content retrieval
**Constraints**: <200ms p95 for API calls, <10MB document size limit, 99% availability
**Scale/Scope**: Support for 10k+ documentation pages, 1000+ concurrent users

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

1. **Technical Accuracy**: All technical concepts must be verified against official documentation - ensure all implementation details align with official sources and documentation.
2. **Clarity**: Content must be accessible to developers and CS students - all implementation decisions should consider the target audience and maintain clear, understandable code and documentation.
3. **Practicality**: Focus on reproducible code examples and real-world workflows - ensure all implementations are practical and can be reproduced by readers.
4. **Documentation Standards**: Use Docusaurus Markdown features (admonitions, code blocks) and maintain APA style citations - all documentation must follow these standards.
5. **Originality and Quality**: 100% human-verified content with 0% plagiarism tolerance - ensure all code and content are original and properly attributed.

## Project Structure

### Documentation (this feature)

```text
specs/001-docusaurus-rag-chatbot/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
backend/
├── src/
│   ├── models/
│   │   ├── document.py
│   │   ├── document_metadata.py
│   │   └── chat_session.py
│   ├── services/
│   │   ├── ingestion_service.py
│   │   ├── embedding_service.py
│   │   ├── retrieval_service.py
│   │   └── rag_service.py
│   ├── api/
│   │   ├── main.py
│   │   ├── routes/
│   │   │   ├── ingestion.py
│   │   │   └── chat.py
│   │   └── dependencies.py
│   ├── config/
│   │   └── settings.py
│   └── utils/
│       └── markdown_parser.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── contract/
└── requirements.txt

frontend/
├── src/
│   ├── components/
│   │   └── Chatbot/
│   │       ├── Chatbot.tsx
│   │       ├── ChatWindow.tsx
│   │       └── FloatingButton.tsx
│   ├── hooks/
│   │   └── useChatbot.ts
│   ├── services/
│   │   └── api.ts
│   └── styles/
│       └── chatbot.css
└── package.json

.env
README.md
```

**Structure Decision**: Web application structure with separate backend and frontend directories. Backend uses FastAPI with models, services, and API routes. Frontend provides React components for the chatbot UI that integrates with Docusaurus.

## Phase 0: Research & Unknowns Resolution

### Research Findings

**Decision**: Use FastAPI with async support for high-performance API endpoints
**Rationale**: FastAPI provides excellent performance with async/await, automatic OpenAPI documentation, and strong type hints
**Alternatives considered**: Flask, Django REST Framework - but FastAPI offers better performance and modern async support

**Decision**: Use Google's embedding-001 model for document embeddings
**Rationale**: Integrates seamlessly with Gemini LLM and provides high-quality embeddings for RAG applications
**Alternatives considered**: OpenAI embeddings, Sentence Transformers - Google embeddings offer better integration with Gemini and cost-effectiveness

**Decision**: Implement React component as Docusaurus theme component for easy integration
**Rationale**: Docusaurus theme components can be easily added to any Docusaurus site without modifying core files
**Alternatives considered**: Plugin approach, direct integration - theme component provides clean separation

## Phase 1: Data Model & API Contracts

### Data Model (data-model.md)

**Document**:
- document_id: str (unique identifier)
- title: str (document title)
- path: str (file path in Docusaurus structure)
- content_hash: str (hash of content for change detection)
- created_date: datetime
- updated_date: datetime

**DocumentMetadata**:
- document_id: str (foreign key to Document)
- title: str
- path: str
- content_hash: str
- created_date: datetime

**ChatSession**:
- session_id: str (unique identifier)
- created_date: datetime
- updated_date: datetime
- selected_text: str (optional, for context)

**QueryResult**:
- query: str (user query)
- response: str (AI response)
- sources: List[str] (document references)
- timestamp: datetime

### API Contracts (contracts/)

**POST /api/ingest** - Ingest Docusaurus markdown files
- Request: { directory_path: str }
- Response: { status: str, processed_files: int, errors: List[str] }

**POST /api/chat** - Process chat queries with RAG
- Request: { query: str, selected_text: str, session_id: str }
- Response: { response: str, sources: List[str], session_id: str }

**GET /api/health** - Health check endpoint
- Response: { status: str, timestamp: datetime }

## Quickstart Guide (quickstart.md)

1. Clone the repository
2. Set up environment variables in `.env` file
3. Install backend dependencies: `pip install -r requirements.txt`
4. Install frontend dependencies: `npm install`
5. Run backend: `uvicorn backend.src.api.main:app --reload`
6. Run frontend: `npm start`
7. Access the Docusaurus site with integrated chatbot

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Multi-project structure | Need to separate concerns between backend and frontend | Single project would create complexity with different tech stacks |