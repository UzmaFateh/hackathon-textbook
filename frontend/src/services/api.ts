// const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';
const API_BASE_URL = 'https://uzifateh-ai-book-chatbot-backend.hf.space';

interface ChatRequest {
  query: string;
  selected_text?: string;
  session_id?: string;
}

interface ChatResponse {
  response: string;
  sources: string[];
  session_id: string;
}

interface IngestionRequest {
  directory_path: string;
}

interface IngestionResponse {
  status: string;
  processed_files: number;
  errors: string[];
}

export const chatApi = {
  async sendMessage(request: ChatRequest): Promise<ChatResponse> {
    const response = await fetch(`${API_BASE_URL}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return response.json();
  },

  async ingestDocuments(request: IngestionRequest): Promise<IngestionResponse> {
    const response = await fetch(`${API_BASE_URL}/api/ingest`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return response.json();
  },

  async healthCheck(): Promise<{ status: string; timestamp: string }> {
    const response = await fetch(`${API_BASE_URL}/api/health`);

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return response.json();
  },
};