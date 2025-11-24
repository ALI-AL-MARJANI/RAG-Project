import re
from typing import List

def clean_text(text: str) -> str:
    """Nettoyage simple avant chunking."""
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'arXiv:\s*\S+', '', text)   
    return text.strip()


def chunk_text(text: str, max_length: int = 800, overlap: int = 100) -> List[str]:
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = min(start + max_length, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        if end == len(words):
            break
        start += max_length - overlap

    return chunks
