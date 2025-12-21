#!/usr/bin/env python3
"""
Test script to verify that the full RAG pipeline works with Cohere and OpenRouter
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables from backend/.env
backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
env_path = os.path.join(backend_dir, '.env')
load_dotenv(env_path)

# Add backend to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.src.services.rag_service import RAGService
from backend.src.services.ingestion_service import IngestionService
from backend.src.config.settings import settings

def test_rag_service():
    """Test the RAG service to ensure it works with Cohere and OpenRouter"""
    try:
        print("Testing RAG service initialization...")
        rag_service = RAGService()
        print("[SUCCESS] RAG service initialized successfully")

        print("\nTesting RAG response generation...")
        # Test with a simple query that doesn't require documents
        # This will test the LLM integration
        result = rag_service.generate_response("What is machine learning?", "Machine learning is a subset of artificial intelligence.")

        print(f"[SUCCESS] RAG response generated")
        print(f"Response length: {len(result['response'])} characters")
        print(f"Sources found: {len(result['sources'])}")

        if result['response'] and len(result['response']) > 0:
            print("[SUCCESS] Valid response received from OpenRouter")
            return True
        else:
            print("[ERROR] Empty response received")
            return False

    except Exception as e:
        print(f"[ERROR] Error during RAG test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_ingestion_service():
    """Test the ingestion service to ensure it works with Cohere embeddings"""
    try:
        print("\nTesting ingestion service initialization...")
        ingestion_service = IngestionService()
        print("[SUCCESS] Ingestion service initialized successfully")

        # Test creating embeddings with sample text
        sample_text = "Artificial intelligence and machine learning are related fields in computer science."
        embedding = ingestion_service.embedding_service.generate_single_embedding(sample_text)

        print(f"[SUCCESS] Embedding generated for sample text")
        print(f"Embedding dimension: {len(embedding)}")

        if len(embedding) > 0 and all(isinstance(x, float) for x in embedding):
            print("[SUCCESS] Valid embedding generated with Cohere")
            return True
        else:
            print("[ERROR] Invalid embedding generated")
            return False

    except Exception as e:
        print(f"[ERROR] Error during ingestion test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Check if required API keys are set
    if not os.getenv("COHERE_API_KEY"):
        print("[WARNING] COHERE_API_KEY not found in environment. Please set it before running this test.")
        print("You can set it with: export COHERE_API_KEY='your-api-key'")
        sys.exit(1)

    if not os.getenv("OPENROUTER_API_KEY"):
        print("[WARNING] OPENROUTER_API_KEY not found in environment. Please set it before running this test.")
        print("You can set it with: export OPENROUTER_API_KEY='your-api-key'")
        sys.exit(1)

    print("Running RAG pipeline tests with Cohere and OpenRouter...\n")

    rag_success = test_rag_service()
    ingestion_success = test_ingestion_service()

    if rag_success and ingestion_success:
        print("\n[SUCCESS] All RAG pipeline tests passed!")
        print("Cohere embeddings and OpenRouter LLM integration are working properly.")
    else:
        print("\n[ERROR] Some tests failed.")
        sys.exit(1)