from src.services.retrieval_service import RetrievalService
from src.config.llm_config import openrouter_client
from src.config.settings import settings
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self):
        self.retrieval_service = RetrievalService()

    def generate_response(self, query: str, selected_text: str = None) -> Dict[str, Any]:
        """
        Generate a response using RAG (Retrieval Augmented Generation)
        """
        try:
            # Prepare the context for the LLM
            context_parts = []

            # Add selected text context if provided
            if selected_text:
                context_parts.append(f"Selected text context: {selected_text}")

            # Retrieve relevant documents based on the query
            search_results = self.retrieval_service.search_similar(query, top_k=5)

            if search_results:
                context_parts.append("Relevant information from documentation:")
                for result in search_results:
                    content = result["payload"].get("content", "")
                    doc_title = result["payload"].get("title", "Unknown")
                    doc_path = result["payload"].get("path", "Unknown")

                    context_parts.append(f"Document: {doc_title} ({doc_path})")
                    context_parts.append(f"Content: {content[:500]}...")  # Limit content length
            else:
                context_parts.append("No relevant documentation found for this query.")

            # Combine all context parts
            context = "\n\n".join(context_parts)

            # Create the full prompt for the LLM
            full_prompt = f"""
            You are a helpful assistant for documentation. Answer the user's question based on the provided context.
            If the context doesn't contain relevant information, say so clearly.
            Always cite the source document when providing information from the documentation.

            Context:
            {context}

            Question: {query}

            Answer:
            """

            # Generate response using OpenRouter
            response = openrouter_client.chat.completions.create(
                model=settings.openrouter_model,  # Using the configured OpenRouter model
                messages=[
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ],
                max_tokens=1000,
                temperature=0.7
            )

            # Extract the answer
            answer = response.choices[0].message.content

            # Extract sources from search results
            sources = []
            for result in search_results:
                doc_path = result["payload"].get("path", "Unknown")
                if doc_path not in sources:
                    sources.append(doc_path)

            return {
                "response": answer,
                "sources": sources
            }
        except Exception as e:
            logger.error(f"Error generating RAG response: {str(e)}")
            return {
                "response": "Sorry, I encountered an error while processing your request.",
                "sources": []
            }