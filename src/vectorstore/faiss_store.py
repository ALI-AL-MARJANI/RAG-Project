import os
import json
import numpy as np
import faiss

class FaissVectorStore:
    """
    Simple wrapper around FAISS for storing and querying embeddings.
    """

    def __init__(self, dim: int, index_type: str = "Flat"):
        self.dim = dim

        if index_type == "Flat":
            self.index = faiss.IndexFlatL2(dim)
        else:
            raise NotImplementedError(f"Index type '{index_type}' not supported.")

        self.metadata = []

    def add(self, embeddings: np.ndarray, metadata: list):
        """
        Add embeddings + metadata to the index.
        """
        if embeddings.shape[1] != self.dim:
            raise ValueError(f"Dimension mismatch: expected {self.dim}, got {embeddings.shape[1]}")

        self.index.add(embeddings)
        self.metadata.extend(metadata)

    def search(self, query_embedding: np.ndarray, k: int = 5):
        """
        Return top-k nearest neighbors for a query embedding.
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        distances, indices = self.index.search(query_embedding, k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            results.append({
                "distance": float(dist),
                "metadata": self.metadata[idx]
            })

        return results

    def save(self, folder: str):
        os.makedirs(folder, exist_ok=True)

        faiss.write_index(self.index, os.path.join(folder, "index.faiss"))

        with open(os.path.join(folder, "metadata.json"), "w") as f:
            json.dump(self.metadata, f, indent=2)

        print(f"Saved FAISS vector store to {folder}")

    @staticmethod
    def load(folder: str):
        index = faiss.read_index(os.path.join(folder, "index.faiss"))

        with open(os.path.join(folder, "metadata.json"), "r") as f:
            metadata = json.load(f)

        store = FaissVectorStore(dim=index.d)
        store.index = index
        store.metadata = metadata
        return store
