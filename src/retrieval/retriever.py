import numpy as np
from src.embedding.embedder import BGEEmbedder
from src.vectorstore.faiss_store import FaissVectorStore

class RAGRetriever:
    def __init__(self, vectorstore: FaissVectorStore, embedder: BGEEmbedder):
        self.vectorstore = vectorstore
        self.embedder = embedder

    def retrieve(self, query: str, k: int = 5):
        """
        Encode la requête, interroge le vector store FAISS et retourne les meilleurs chunks.
        """
        q_emb = self.embedder.embed_texts([query])
        results = self.vectorstore.search(q_emb, k=k)
        return results
