import os
import sys
import json

from src.embedding.embedder import BGEEmbedder
from src.vectorstore.faiss_store import FaissVectorStore
from src.retrieval.retriever import RAGRetriever
from src.generation.generator import LocalGenerator
from src.evaluation.rag_evaluator import RAGEvaluator

def main():
    VECTOR_DIR = "data/vectorstore"

    
    if not os.path.exists(os.path.join(VECTOR_DIR, "index.faiss")):
        print(f"Error: Index not found in {VECTOR_DIR}.")
        print("👉 Please run 'python main.py' first to ingest and index documents.")
        sys.exit(1)

    
    # 1. Load Components
    embedder = BGEEmbedder(device="cpu")
    vector_store = FaissVectorStore.load(VECTOR_DIR)
    retriever = RAGRetriever(vectorstore=vector_store, embedder=embedder)
    generator = LocalGenerator(model_name="mistral")

    # 2. Initialize Evaluator
    evaluator = RAGEvaluator(retriever, generator)

    # 3. Define Test Set of " Golden Questions "
    # Since we are using arXiv ML papers, these are generic relevant questions:
    test_questions = [
        "What are the main contributions of the paper?",
        "How does the proposed method compare to state-of-the-art?",
        "What dataset was used for evaluation?",
        "Explain the architecture of the model described.",
        "What are the limitations mentioned by the authors?"
    ]

    # 4. Run Benchmark
    metrics, details = evaluator.run_benchmark(test_questions, k=3)

    # 5. Print Report
    print("\n" + "="*40)
    print("   PERFORMANCE REPORT")
    print("="*40)
    print(f"Total Questions        : {metrics['total_questions']}")
    print(f"Avg Retrieval Latency  : {metrics['avg_retrieval_latency']:.4f} sec")
    print(f"Avg Generation Latency : {metrics['avg_generation_latency']:.4f} sec")
    print(f"Relevance Score (0-1)  : {metrics['avg_relevance_score']:.2f}  (1.0 = Perfect)")
    print("="*40)

    # 6. Save Results
    output_file = "benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump({"metrics": metrics, "details": details}, f, indent=4)
    
    print(f"\n Detailed results saved to '{output_file}'")

if __name__ == "__main__":
    main()