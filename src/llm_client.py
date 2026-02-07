import ollama
from typing import List, Optional, Generator, Tuple, Dict


class OllamaClient:
    def __init__(self, model: str = "qwen3:1.7b"):
        """
        Initialize Ollama client with specified model.
        
        Args:
            model: Name of the Ollama model to use (default: qwen3:1.7b)
        """
        self.model = model
        self.last_thinking = ""  # Store thinking from last generation
        
    def generate_answer(
        self, 
        query: str, 
        context_chunks: List[str],
        temperature: float = 0.7
    ) -> Tuple[str, str]:
        """
        Generate an answer using retrieved context chunks with inline citations.
        
        Args:
            query: User's question
            context_chunks: Relevant text chunks from RAG retrieval
            temperature: Controls randomness (0.0 = deterministic, 1.0 = creative)
            
        Returns:
            Tuple of (answer, thinking) where thinking contains model's reasoning
        """
        # Build the prompt with numbered sources for citations
        numbered_sources = []
        for i, chunk in enumerate(context_chunks, 1):
            numbered_sources.append(f"[{i}] {chunk}")
        context = "\n\n".join(numbered_sources)
        
        prompt = f"""You are a helpful technical assistant. Answer the question based on the sources provided below.

IMPORTANT INSTRUCTIONS:
- Include inline citations [1], [2], [3] after each claim that references a source
- If the context contains mathematical equations or formulas, preserve them exactly as shown
- If the context contains code blocks, preserve the code formatting
- If the context contains tables, describe them clearly
- Be precise and technical when appropriate

Sources:
{context}

Question: {query}

Answer (include [1], [2], etc. citations after statements that use information from sources):"""
        
        try:
            # Call Ollama API
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": temperature,
                }
            )
            
            answer = response['response'].strip()
            thinking = ""
            
            # Extract thinking from response if present (qwen3 format: <think>...</think>)
            if '<think>' in answer and '</think>' in answer:
                start = answer.find('<think>')
                end = answer.find('</think>') + len('</think>')
                thinking = answer[start+7:end-8].strip()
                answer = answer[:start] + answer[end:]
                answer = answer.strip()
            
            self.last_thinking = thinking
            return answer, thinking
            
        except Exception as e:
            return f"Error generating response: {str(e)}", ""
    
    def generate_answer_stream(
        self, 
        query: str, 
        context_chunks: List[str],
        temperature: float = 0.7
    ) -> Generator[Dict[str, str], None, None]:
        """
        Generate an answer using streaming, yielding tokens and thinking separately.
        
        Args:
            query: User's question
            context_chunks: Relevant text chunks from RAG retrieval
            temperature: Controls randomness
            
        Yields:
            Dict with either {'token': str} or {'thinking': str}
        """
        # Build the prompt with numbered sources for citations
        numbered_sources = []
        for i, chunk in enumerate(context_chunks, 1):
            numbered_sources.append(f"[{i}] {chunk}")
        context = "\n\n".join(numbered_sources)
        
        prompt = f"""You are a helpful technical assistant. Answer the question based on the sources provided below.

IMPORTANT INSTRUCTIONS:
- Include inline citations [1], [2], [3] after each claim that references a source
- If the context contains mathematical equations or formulas, preserve them exactly as shown
- If the context contains code blocks, preserve the code formatting
- If the context contains tables, describe them clearly
- Be precise and technical when appropriate

Sources:
{context}

Question: {query}

Answer (include [1], [2], etc. citations after statements that use information from sources):"""
        
        try:
            # Call Ollama API with streaming
            stream = ollama.generate(
                model=self.model,
                prompt=prompt,
                stream=True,
                options={
                    "temperature": temperature,
                }
            )
            
            in_thinking = False
            thinking_buffer = ""
            
            for chunk in stream:
                if 'response' in chunk:
                    token = chunk['response']
                    
                    # Check for thinking tags (qwen3 uses <think>...</think>)
                    if '<think>' in token:
                        in_thinking = True
                        # Extract any content before the tag
                        before = token.split('<think>')[0]
                        if before:
                            yield {'token': before}
                        continue
                    
                    if '</think>' in token:
                        in_thinking = False
                        # Extract thinking content and any after
                        parts = token.split('</think>')
                        if thinking_buffer:
                            thinking_buffer += parts[0]
                            yield {'thinking': thinking_buffer}
                            thinking_buffer = ""
                        if len(parts) > 1 and parts[1]:
                            yield {'token': parts[1]}
                        continue
                    
                    if in_thinking:
                        thinking_buffer += token
                        # Yield thinking tokens progressively for long thinking
                        if len(thinking_buffer) > 100:
                            yield {'thinking': thinking_buffer}
                            thinking_buffer = ""
                    else:
                        yield {'token': token}
            
            # Yield any remaining thinking
            if thinking_buffer:
                yield {'thinking': thinking_buffer}
                    
        except Exception as e:
            yield {'token': f"Error: {str(e)}"}
    
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
    
    def chat_stream(self, query: str, temperature: float = 0.7) -> Generator[Dict[str, str], None, None]:
        """
        Simple chat without RAG context, with streaming.
        
        Yields:
            Dict with either {'token': str} or {'thinking': str}
        """
        try:
            stream = ollama.generate(
                model=self.model,
                prompt=query,
                stream=True,
                options={
                    "temperature": temperature,
                }
            )
            
            in_thinking = False
            thinking_buffer = ""
            
            for chunk in stream:
                if 'response' in chunk:
                    token = chunk['response']
                    
                    if '<think>' in token:
                        in_thinking = True
                        before = token.split('<think>')[0]
                        if before:
                            yield {'token': before}
                        continue
                    
                    if '</think>' in token:
                        in_thinking = False
                        parts = token.split('</think>')
                        if thinking_buffer:
                            thinking_buffer += parts[0]
                            yield {'thinking': thinking_buffer}
                            thinking_buffer = ""
                        if len(parts) > 1 and parts[1]:
                            yield {'token': parts[1]}
                        continue
                    
                    if in_thinking:
                        thinking_buffer += token
                        if len(thinking_buffer) > 100:
                            yield {'thinking': thinking_buffer}
                            thinking_buffer = ""
                    else:
                        yield {'token': token}
            
            if thinking_buffer:
                yield {'thinking': thinking_buffer}
                
        except Exception as e:
            yield {'token': f"Error: {str(e)}"}


# Quick test
if __name__ == "__main__":
    client = OllamaClient()
    print("Testing Ollama connection...")
    response = client.chat("Hello! Can you introduce yourself?")
    print(f"Response: {response}")


