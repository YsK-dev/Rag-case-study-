import ollama
from typing import List, Optional


class OllamaClient:
    def __init__(self, model: str = "qwen3:1.7b"):
        """
        Initialize Ollama client with specified model.
        
        Args:
            model: Name of the Ollama model to use (default: qwen3:1.7b)
        """
        self.model = model
        
    def generate_answer(
        self, 
        query: str, 
        context_chunks: List[str],
        temperature: float = 0.7
    ) -> str:
        """
        Generate an answer using retrieved context chunks.
        
        Args:
            query: User's question
            context_chunks: Relevant text chunks from RAG retrieval
            temperature: Controls randomness (0.0 = deterministic, 1.0 = creative)
            
        Returns:
            Generated answer as a string
        """
        # Build the prompt with context
        context = "\n\n".join(context_chunks)
        
        prompt = f"""You are a helpful technical assistant. Answer the question based on the context provided below.

IMPORTANT INSTRUCTIONS:
- If the context contains mathematical equations or formulas, preserve them exactly as shown
- If the context contains code blocks, preserve the code formatting
- If the context contains tables, describe them clearly
- Be precise and technical when appropriate
- Cite specific details from the context

Context:
{context}

Question: {query}

Answer (be detailed and preserve any equations, code, or technical formatting):"""
        
        try:
            # Call Ollama API
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": temperature,
                }
            )
            
            return response['response'].strip()
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def chat(self, query: str, temperature: float = 0.7) -> str:
        """
        Simple chat without RAG context (for testing).
        
        Args:
            query: User's question
            temperature: Controls randomness
            
        Returns:
            Generated response
        """
        try:
            response = ollama.generate(
                model=self.model,
                prompt=query,
                options={
                    "temperature": temperature,
                }
            )
            return response['response'].strip()
        except Exception as e:
            return f"Error: {str(e)}"


# Quick test
if __name__ == "__main__":
    client = OllamaClient()
    print("Testing Ollama connection...")
    response = client.chat("Hello! Can you introduce yourself?")
    print(f"Response: {response}")
