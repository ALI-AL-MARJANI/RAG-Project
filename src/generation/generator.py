from huggingface_hub import InferenceClient
import os

class RAGGenerator:
    """
    LLM generator for RAG pipeline using HuggingFace Inference API
    Perfect for lightweight local development.
    """

    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.3", api_key=None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("HF_API_KEY")

        if self.api_key is None:
            raise ValueError("Missing HF_API_KEY environment variable.")

        self.client = InferenceClient(
            model=model_name,
            token=self.api_key
        )

    def format_prompt(self, query: str, retrieved_docs: list):
        """
        Build the context + question prompt for the LLM.
        """

        context_text = "\n\n".join([doc["metadata"]["text"] for doc in retrieved_docs])

        prompt = f"""
You are a knowledgeable assistant. 
Use ONLY the following context to answer the question. 
If the answer is not found, say you don't know.

### Context:
{context_text}

### Question:
{query}

### Answer:
"""
        return prompt

    def generate(self, query: str, retrieved_docs: list, max_tokens=512):
        """
        Generate final answer using the RAG pipeline.
        """

        prompt = self.format_prompt(query, retrieved_docs)

        response = self.client.text_generation(
            prompt,
            max_new_tokens=max_tokens,
            temperature=0.2,
        )

        return response
