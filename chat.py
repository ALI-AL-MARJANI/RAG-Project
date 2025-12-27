import os
import sys
from src.embedding.embedder import BGEEmbedder
from src.vectorstore.faiss_store import FaissVectorStore
from src.retrieval.retriever import RAGRetriever
from src.generation.generator import LocalGenerator

def start_chat():
    VECTOR_DIR = "data/vectorstore"
    if not os.path.exists(os.path.join(VECTOR_DIR, "index.faiss")):
        print(f" No vector store found in '{VECTOR_DIR}'")
        sys.exit(1)

    

    # 1. We load the embedding model
    embedder = BGEEmbedder(device="cpu")

    # 2. Load the FAISS knowledge base
    vector_store = FaissVectorStore.load(VECTOR_DIR)

    # 3. Setup RAG components
    retriever = RAGRetriever(vectorstore=vector_store, embedder=embedder)
    generator = LocalGenerator(model_name="mistral")
    # 4. Interactive chat loop
    while True:
        query = input("\n Enter your question (or 'q' to quit): ")
        if query.lower() in ['q', 'quit', 'exit']:
            break
        
        retrieved_docs = retriever.retrieve(query, k=3)
        print("Generating answer...")
        answer = generator.generate(query, retrieved_docs)
        
        print(answer)
        print("----------------")
        sources = [d['metadata'].get('source_file', 'Inconnu') for d in retrieved_docs]
        print(f"Sources : {sources}")

if __name__ == "__main__":
    start_chat()