import time
import statistics
import requests
from typing import List, Dict, Tuple, Any






from src.retrieval.retriever import RAGRetriever
from src.generation.generator import LocalGenerator

class RAGEvaluator:
    """
    Evaluator class for the RAG pipeline
    --> It measures latency and uses an 'LLM-as-a-Judge' approach to assess 
    the relevance of retrieved documents.
    """

    def __init__(self, retriever: RAGRetriever, generator: LocalGenerator):
        self.retriever = retriever
        self.generator = generator

    def evaluate_context_relevance(self, query: str, context_chunks: List[Dict]) -> float:
        """
        Uses the Local LLM to judge if the retrieved chunks are relevant to the query.
        Returns:
            float: 1.0 if relevant, 0.0 otherwise.
        """
        # Concatenate retrieved text
        context_text = "\n".join([doc['metadata'].get('text', '') for doc in context_chunks])
        
        # 'LLM-as-a-Judge' Prompt
        evaluation_prompt = f"""
        You are a strict evaluator for a RAG system.
        Your task is to judge if the retrieved context is relevant to the user's query.
        
        QUERY: "{query}"
        
        RETRIEVED CONTEXT:
        "{context_text}"
        
        Does the context contain the information needed to answer the query?
        Answer only with "YES" or "NO".
        """
        
        # Direct API call 
        payload = {
            "model": self.generator.model_name,
            "prompt": evaluation_prompt,
            "stream": False
        }
        
        try:
            response = requests.post(self.generator.api_url, json=payload)
            response.raise_for_status()
            result = response.json().get("response", "").strip().upper()
            
            # If the LLM says YES, score is 1.0, otherwise 0.0
            return 1.0 if "YES" in result else 0.0
        except Exception as e:
            print(f"Evaluation error: {e}")
            return 0.0

    def run_benchmark(self, test_set: List[str], k: int = 3) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Runs a full benchmark on a list of questions.
        Args:
            test_set : List of questions to evaluate.
            k : Number of documents to retrieve for each question.
        """
        results = []
        latencies_retrieval = []
        latencies_generation = []
        relevance_scores = []

        

        for i, query in enumerate(test_set):
            print(f"[{i+1}/{len(test_set)}] Testing: {query}")
            
            # 1. Measure Retrieval Time
            start_ret = time.time()
            retrieved_docs = self.retriever.retrieve(query, k=k)
            end_ret = time.time()
            retrieval_time = end_ret - start_ret
            latencies_retrieval.append(retrieval_time)

            # 2. Measure Generation Time
            start_gen = time.time()
            answer = self.generator.generate(query, retrieved_docs)
            end_gen = time.time()
            generation_time = end_gen - start_gen
            latencies_generation.append(generation_time)

            # 3. Evaluate Relevance (LLM Judge)
            score = self.evaluate_context_relevance(query, retrieved_docs)
            relevance_scores.append(score)

            results.append({
                "query": query,
                "retrieval_time_sec": round(retrieval_time, 4),
                "generation_time_sec": round(generation_time, 4),
                "relevance_score": score,
                "answer_preview": answer[:100] + "..." if answer else "No answer"
            })

        # Calculate Aggregated Metrics
        metrics = {
            "avg_retrieval_latency": statistics.mean(latencies_retrieval),
            "avg_generation_latency": statistics.mean(latencies_generation),
            "avg_relevance_score": statistics.mean(relevance_scores),
            "total_questions": len(test_set)
        }

        return metrics, results