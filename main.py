import os
import json
import shutil
from pathlib import Path
import numpy as np
from src.ingestion.arxiv_loader import download_arxiv_papers
from src.processing.parser import batch_parse_pdfs
from src.chunking.chunker import clean_text, chunk_text
from src.embedding.embedder import BGEEmbedder, build_embeddings_from_chunks
from src.vectorstore.faiss_store import FaissVectorStore
from src.retrieval.retriever import RAGRetriever
from src.generation.generator import LocalGenerator

def main():
    
    DATA_DIR = "data"
    ARXIV_CAT = "cs.LG"
    MAX_PAPERS = 2  
    MODEL_NAME = "mistral" 


    # 1. INGESTION
    raw_dir = os.path.join(DATA_DIR, "raw/arxiv")
    download_arxiv_papers(output_dir=raw_dir, category=ARXIV_CAT, max_results=MAX_PAPERS)

    # 2. PARSING
    processed_text_dir = os.path.join(DATA_DIR, "processed/text")
    txt_files = batch_parse_pdfs(input_dir=raw_dir, output_dir=processed_text_dir)

    # 3. CHUNKING (Bridge entre txt et json pour l'embedder)
    chunks_dir = os.path.join(DATA_DIR, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)
    
    for txt_path in txt_files:
        p = Path(txt_path)
        with open(p, "r", encoding="utf-8") as f:
            raw_content = f.read()
        
        cleaned_content = clean_text(raw_content)
        chunks = chunk_text(cleaned_content)
        
        # Save chunks to JSON
        json_path = os.path.join(chunks_dir, f"{p.stem}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"id": p.stem, "chunks": chunks}, f, indent=2)

    # 4. EMBEDDING
    embedder = BGEEmbedder(device="cpu") 
    vectorstore_dir = os.path.join(DATA_DIR, "vectorstore")
    emb_path, meta_path = build_embeddings_from_chunks(chunks_dir, vectorstore_dir, embedder)

    # 5. INDEXING & RETRIEVAL SETUP

    embeddings = np.load(emb_path)
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    
    
    vector_store = FaissVectorStore(dim=384) # BGE-small dimension is 384
    vector_store.add(embeddings, metadata)
    
    retriever = RAGRetriever(vectorstore=vector_store, embedder=embedder)
    generator = LocalGenerator(model_name=MODEL_NAME)

    # 6. INTERACTIVE LOOP
    while True:
        query = input("\nPose ta question (ou 'q' pour quitter) : ")
        if query.lower() in ['q', 'quit', 'exit']:
            break
        
        print(" Searching for relevant documents...")
        retrieved_docs = retriever.retrieve(query, k=3)
        
        print(" Generating answer...")
        answer = generator.generate(query, retrieved_docs)
        print(answer)
        print("----------------")
        print(f" Sources used : {[d['metadata']['source_file'] for d in retrieved_docs]}")
if __name__ == "__main__":
    main()