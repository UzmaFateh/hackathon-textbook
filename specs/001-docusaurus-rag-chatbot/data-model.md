# Data Model: Docusaurus RAG Chatbot

**Feature**: Docusaurus RAG Chatbot
**Date**: 2025-12-18
**Branch**: 001-docusaurus-rag-chatbot

## Entity Definitions

### Document
Represents a documentation file from the Docusaurus site with content and source information.

**Fields:**
- `document_id`: str (primary key, unique identifier)
- `title`: str (document title)
- `path`: str (file path in Docusaurus structure, e.g., "docs/intro.md")
- `content`: str (raw content of the document)
- `content_hash`: str (SHA-256 hash of content for change detection)
- `created_date`: datetime (timestamp when document was first processed)
- `updated_date`: datetime (timestamp when document was last updated)

**Relationships:**
- One-to-many with DocumentChunk (a document can have multiple chunks)

**Validation Rules:**
- `document_id` must be unique
- `path` must follow Docusaurus standard structure
- `content_hash` is required for change detection

### DocumentMetadata
Represents metadata stored in Neon Postgres with fields: document_id, title, path, content_hash, created_date

**Fields:**
- `id`: int (primary key, auto-increment)
- `document_id`: str (foreign key to Document)
- `title`: str (document title)
- `path`: str (file path in Docusaurus structure)
- `content_hash`: str (hash of content for change detection)
- `created_date`: datetime (timestamp when metadata was created)
- `updated_date`: datetime (timestamp when metadata was last updated)

**Validation Rules:**
- `document_id` must reference an existing Document
- `path` must be unique within the system

### DocumentChunk
Represents a chunk of a document that has been processed for vector storage.

**Fields:**
- `chunk_id`: str (primary key, unique identifier)
- `document_id`: str (foreign key to Document)
- `content`: str (chunked content from the document)
- `chunk_index`: int (position of this chunk in the original document)
- `embedding_vector_id`: str (ID in Qdrant vector storage)
- `created_date`: datetime (timestamp when chunk was created)

**Relationships:**
- Many-to-one with Document (multiple chunks belong to one document)

**Validation Rules:**
- `document_id` must reference an existing Document
- `chunk_index` must be non-negative

### ChatSession
Represents a conversation context with history and selected text context.

**Fields:**
- `session_id`: str (primary key, unique identifier)
- `selected_text`: str (optional, text selected by user for context)
- `created_date`: datetime (timestamp when session was created)
- `updated_date`: datetime (timestamp when session was last updated)

**Validation Rules:**
- `session_id` must be unique
- `selected_text` is optional

### ChatMessage
Represents a single message in a chat session.

**Fields:**
- `message_id`: str (primary key, unique identifier)
- `session_id`: str (foreign key to ChatSession)
- `role`: str (either "user" or "assistant")
- `content`: str (message content)
- `timestamp`: datetime (when the message was created)
- `sources`: List[str] (optional, list of document sources referenced)

**Relationships:**
- Many-to-one with ChatSession (multiple messages in one session)

**Validation Rules:**
- `session_id` must reference an existing ChatSession
- `role` must be either "user" or "assistant"

### QueryResult
Contains retrieved relevant content with source citations.

**Fields:**
- `query`: str (original user query)
- `response`: str (AI-generated response)
- `sources`: List[str] (document references used to generate response)
- `timestamp`: datetime (when the query was processed)
- `session_id`: str (optional, session identifier)

**Validation Rules:**
- `sources` must be a list of valid document references

## Database Schema

### Neon Postgres Schema

```sql
-- Document metadata table
CREATE TABLE document_metadata (
    id SERIAL PRIMARY KEY,
    document_id VARCHAR(255) NOT NULL UNIQUE,
    title TEXT NOT NULL,
    path TEXT NOT NULL UNIQUE,
    content_hash VARCHAR(64) NOT NULL,
    created_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Chat sessions table
CREATE TABLE chat_sessions (
    session_id VARCHAR(255) PRIMARY KEY,
    selected_text TEXT,
    created_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Chat messages table
CREATE TABLE chat_messages (
    message_id VARCHAR(255) PRIMARY KEY,
    session_id VARCHAR(255) REFERENCES chat_sessions(session_id),
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    sources TEXT[]
);

-- Indexes for performance
CREATE INDEX idx_document_metadata_path ON document_metadata(path);
CREATE INDEX idx_document_metadata_hash ON document_metadata(content_hash);
CREATE INDEX idx_chat_sessions_created ON chat_sessions(created_date);
CREATE INDEX idx_chat_messages_session ON chat_messages(session_id);
```

## State Transitions

### Document States
1. **NEW**: Document discovered but not yet processed
2. **PROCESSING**: Document being chunked and embedded
3. **PROCESSED**: Document successfully processed and stored in vector database
4. **UPDATED**: Document content changed and needs reprocessing
5. **DELETED**: Document removed from source, needs removal from vector database

### Chat Session States
1. **ACTIVE**: Session is open and accepting messages
2. **INACTIVE**: Session has been idle for a period (may be archived)
3. **EXPIRED**: Session has exceeded retention period (may be deleted)

## Relationships

- Document → DocumentChunk (one-to-many)
- ChatSession → ChatMessage (one-to-many)
- DocumentMetadata → Document (one-to-one or one-to-many depending on implementation)

## Constraints

- Document content size limited to 10MB as per feature specification
- Chat session retention limited to 30 days as per feature specification
- Unique constraint on document path to prevent duplicates
- Foreign key constraints to maintain referential integrity