from openai import OpenAI
from src.config.settings import settings
import cohere


def get_openrouter_client():
    """
    Initialize and return the OpenAI client configured for OpenRouter
    """
    client = OpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url
    )
    return client


def get_cohere_client():
    """
    Initialize and return the Cohere client for embeddings
    """
    client = cohere.Client(api_key=settings.cohere_api_key)
    return client


# Initialize clients
openrouter_client = get_openrouter_client()
cohere_client = get_cohere_client()