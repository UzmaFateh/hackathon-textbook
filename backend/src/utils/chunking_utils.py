import re
from typing import List, Dict, Any


def chunk_text(text: str, max_chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Split text into chunks of specified size with overlap
    """
    if not text:
        return []

    # Split text into sentences to avoid breaking in the middle of sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = ""
    chunk_index = 0

    for sentence in sentences:
        # If adding the next sentence would exceed the max chunk size
        if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
            # Save the current chunk
            chunks.append({
                "content": current_chunk.strip(),
                "chunk_index": chunk_index
            })

            # Start a new chunk with overlap from the previous chunk
            if overlap > 0:
                # Get the end portion of the current chunk for overlap
                overlap_text = _get_overlap_text(current_chunk, overlap)
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk = sentence
            chunk_index += 1
        else:
            # Add the sentence to the current chunk
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence

    # Add the last chunk if it has content
    if current_chunk.strip():
        chunks.append({
            "content": current_chunk.strip(),
            "chunk_index": chunk_index
        })

    return chunks


def _get_overlap_text(text: str, overlap_size: int) -> str:
    """
    Get the ending portion of text with specified overlap size
    """
    words = text.split()
    if len(words) <= overlap_size:
        return text

    # Get the last 'overlap_size' words
    overlap_words = words[-overlap_size:]
    return " ".join(overlap_words)