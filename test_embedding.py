#!/usr/bin/env python3
"""
Test script to verify that the embedding service works with Cohere
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables from backend/.env
backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
env_path = os.path.join(backend_dir, '.env')
load_dotenv(env_path)

# Add backend/src to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.src.services.embedding_service import EmbeddingService

def test_embedding_service():
    """Test the embedding service to ensure it works with Cohere"""
    try:
        # Initialize the embedding service
        embedding_service = EmbeddingService()

        # Test with a sample text
        test_texts = ["Hello world", "This is a test document", "Machine learning is fascinating"]

        print("Testing embedding generation...")
        embeddings = embedding_service.generate_embeddings(test_texts)

        print(f"Generated {len(embeddings)} embeddings")
        for i, embedding in enumerate(embeddings):
            print(f"Text {i+1}: '{test_texts[i]}' -> Embedding length: {len(embedding)}")

        # Test single embedding
        single_embedding = embedding_service.generate_single_embedding("Single test")
        print(f"Single embedding length: {len(single_embedding)}")

        print("\n[SUCCESS] Embedding service test passed!")
        print(f"[SUCCESS] Embedding dimension: {len(single_embedding)} (should be 1024 for Cohere's embed-english-v3.0)")

        return True

    except Exception as e:
        print(f"\n[ERROR] Error during embedding test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Check if COHERE_API_KEY is set
    if not os.getenv("COHERE_API_KEY"):
        print("⚠️  Warning: COHERE_API_KEY not found in environment. Please set it before running this test.")
        print("You can set it with: export COHERE_API_KEY='your-api-key'")
        sys.exit(1)

    success = test_embedding_service()
    if not success:
        sys.exit(1)