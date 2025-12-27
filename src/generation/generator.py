import requests

class LocalGenerator:
    """
    AI assistant that generates responses using a locally hosted Ollama model
    """
    def __init__(self, model_name="mistral", api_url="http://localhost:11434/api/generate"):
        self.model_name = model_name
        self.api_url = api_url

    def generate(self, query: str, retrieved_docs: list) -> str:
        """
        Generate a response based on the query and retrieved documents
        """
        # Construction du contexte (extraction du texte des métadonnées)
        # Note: on suppose que 'retrieved_docs' contient une clé 'text' dans metadata
        context_text = "\n\n".join([doc["metadata"].get("text", "") for doc in retrieved_docs])

        prompt = f"""You are a helpful AI assistant for technical documentation.
Use the following pieces of retrieved context to answer the user's question.
If the answer is not in the context, say that you don't know.

CONTEXT:
{context_text}

USER QUESTION:
{query}

ANSWER:
"""
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }

        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            return response.json().get("response", "No response.")
        except Exception as e:
            return f"Error communicating with Ollama: {e}"