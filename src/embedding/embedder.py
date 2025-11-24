import os
import json
from typing import List, Dict, Tuple, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


class BGEEmbedder:
    """
    Wrapper around a BGE model via sentence-transformer.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: Optional[str] = None):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:

        if len(texts) == 0:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()))

        emb = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=normalize_embeddings,
        )
        return np.array(emb)


def build_embeddings_from_chunks(
    chunks_dir: str,
    output_dir: str,
    embedder: BGEEmbedder,
    batch_size: int = 32,
) -> Tuple[str, str]:

    os.makedirs(output_dir, exist_ok=True)

    all_texts: List[str] = []
    all_meta: List[Dict] = []

    files = [f for f in os.listdir(chunks_dir) if f.endswith(".json")]
    files.sort()

    for fname in files:
        path = os.path.join(chunks_dir, fname)
        with open(path, "r") as f:
            data = json.load(f)

        doc_id = data.get("id", fname.replace(".json", ""))
        chunks = data.get("chunks", [])

        for idx, chunk in enumerate(chunks):
            all_texts.append(chunk)
            all_meta.append(
                {
                    "doc_id": doc_id,
                    "chunk_id": idx,
                    "source_file": path,
                }
            )

    print(f"Nombre total de chunks à encoder : {len(all_texts)}")

    embeddings = embedder.embed_texts(all_texts, batch_size=batch_size)

    emb_path = os.path.join(output_dir, "embeddings.npy")
    meta_path = os.path.join(output_dir, "metadata.json")

    np.save(emb_path, embeddings)
    with open(meta_path, "w") as f:
        json.dump(all_meta, f, indent=2)

    print(f"Embeddings sauvegardés dans : {emb_path}")
    print(f"Métadonnées sauvegardées dans : {meta_path}")
    print(f"Shape des embeddings : {embeddings.shape}")

    return emb_path, meta_path
