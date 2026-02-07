import ollama
import re
from typing import List, Optional, Generator, Tuple, Dict


# --------------- Guardrails / prompt injection detection ---------------

_INJECTION_PATTERNS = [
    r"\bignore\s+(all\s+)?previous\s+instructions\b",
    r"\bdisregard\s+(all\s+)?(previous|above|prior)\s+(instructions|context)\b",
    r"\byou\s+are\s+now\s+(a|an|DAN|jailbreak)\b",
    r"\bpretend\s+you\s+are\s+(not\s+)?(an?\s+)?AI\b",
    r"\bact\s+as\s+if\s+you\s+have\s+no\s+restrictions\b",
    r"\boverride\s+(system|safety)\s+(prompt|instructions)\b",
    r"\breveal\s+(your|the)\s+(system\s+)?prompt\b",
    r"^\s*system\s*:\s*",             # raw "system:" at line start
    r"<\|im_start\|>",                # ChatML injection
    r"\[INST\]",                      # Llama-style injection
]

_COMPILED_INJECTION = [
    re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS
]


def sanitize_input(text: str) -> Tuple[str, bool]:
    """Sanitise user input and check for prompt-injection attempts.

    Returns:
        Tuple of (sanitised_text, is_safe).
        If *is_safe* is ``False`` the query should be rejected.
    """
    # Strip control characters (keep newlines / tabs)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    for pattern in _COMPILED_INJECTION:
        if pattern.search(cleaned):
            return cleaned, False

    return cleaned, True


class OllamaClient:
    def __init__(self, model: str = "qwen3:1.7b"):
        """
        Initialize Ollama client with specified model.
        
        Args:
            model: Name of the Ollama model to use (default: qwen3:1.7b)
        """
        self.model = model
        self.last_thinking = ""  # Store thinking from last generation

    # ----------------------------------------------------------------
    #  Prompt builders
    # ----------------------------------------------------------------

    @staticmethod
    def _build_rag_prompt(
        query: str,
        context_chunks: List[str],
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Build a RAG prompt with numbered sources, chat history,
        and a confidence guardrail."""
        numbered_sources = []
        for i, chunk in enumerate(context_chunks, 1):
            numbered_sources.append(f"[{i}] {chunk}")
        context = "\n\n".join(numbered_sources)

        history_block = ""
        if chat_history:
            turns = []
            for turn in chat_history[-6:]:      # last 6 turns max
                role = turn.get("role", "user").capitalize()
                turns.append(f"{role}: {turn['content']}")
            history_block = (
                "Conversation so far:\n"
                + "\n".join(turns)
                + "\n\n"
            )

        return f"""You are a helpful technical assistant. Answer the question based on the sources provided below.

IMPORTANT INSTRUCTIONS:
- Include inline citations [1], [2], [3] after each claim that references a source
- If the context contains mathematical equations or formulas, preserve them exactly as shown
- If the context contains code blocks, preserve the code formatting
- If the context contains tables, describe them clearly
- Be precise and technical when appropriate
- If the provided sources do NOT contain enough information to answer the question confidently, respond with: "I don't have enough information in the provided documents to answer this question."

{history_block}Sources:
{context}

Question: {query}

Answer (include [1], [2], etc. citations after statements that use information from sources):"""

    @staticmethod
    def _build_chat_prompt(
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Build a plain chat prompt (no RAG context) with optional history."""
        history_block = ""
        if chat_history:
            turns = []
            for turn in chat_history[-6:]:
                role = turn.get("role", "user").capitalize()
                turns.append(f"{role}: {turn['content']}")
            history_block = (
                "Conversation so far:\n"
                + "\n".join(turns)
                + "\n\n"
            )
        return f"""{history_block}Question: {query}

Answer:"""
        
    def generate_answer(
        self, 
        query: str, 
        context_chunks: List[str],
        temperature: float = 0.7,
        chat_history: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Generate an answer using retrieved context chunks with inline citations.
        
        Args:
            query: User's question
            context_chunks: Relevant text chunks from RAG retrieval
            temperature: Controls randomness (0.0 = deterministic, 1.0 = creative)
            chat_history: Previous conversation turns for multi-turn context.
            model: Optional model override for this request (thread-safe).
            
        Returns:
            Tuple of (answer, thinking) where thinking contains model's reasoning
        """
        prompt = self._build_rag_prompt(query, context_chunks, chat_history)
        use_model = model or self.model
        
        try:
            # Call Ollama API
            response = ollama.generate(
                model=use_model,
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
        temperature: float = 0.7,
        chat_history: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
    ) -> Generator[Dict[str, str], None, None]:
        """
        Generate an answer using streaming, yielding tokens and thinking separately.
        
        Args:
            query: User's question
            context_chunks: Relevant text chunks from RAG retrieval
            temperature: Controls randomness
            chat_history: Previous conversation turns for multi-turn context.
            model: Optional model override for this request (thread-safe).
            
        Yields:
            Dict with either {'token': str} or {'thinking': str}
        """
        prompt = self._build_rag_prompt(query, context_chunks, chat_history)
        use_model = model or self.model
        
        try:
            # Call Ollama API with streaming
            stream = ollama.generate(
                model=use_model,
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
                        parts = token.split('<think>', 1)
                        if parts[0]:
                            yield {'token': parts[0]}
                        remainder = parts[1] if len(parts) > 1 else ""
                        
                        # Check if closing tag is also in this chunk
                        if '</think>' in remainder:
                            in_thinking = False
                            close_parts = remainder.split('</think>', 1)
                            thinking_buffer += close_parts[0]
                            if thinking_buffer:
                                yield {'thinking': thinking_buffer}
                                thinking_buffer = ""
                            if len(close_parts) > 1 and close_parts[1]:
                                yield {'token': close_parts[1]}
                            continue
                        
                        if remainder:
                            thinking_buffer += remainder
                        continue
                    
                    if '</think>' in token:
                        in_thinking = False
                        parts = token.split('</think>', 1)
                        thinking_buffer += parts[0]
                        if thinking_buffer:
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
    
    def chat(
        self,
        query: str,
        temperature: float = 0.7,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Simple chat without RAG context (for testing).
        
        Args:
            query: User's question
            temperature: Controls randomness
            chat_history: Previous conversation turns.
            
        Returns:
            Generated response
        """
        try:
            prompt = self._build_chat_prompt(query, chat_history)
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
            return answer
        except Exception as e:
            return f"Error: {str(e)}"
    
    def chat_stream(
        self,
        query: str,
        temperature: float = 0.7,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Generator[Dict[str, str], None, None]:
        """
        Simple chat without RAG context, with streaming.
        
        Yields:
            Dict with either {'token': str} or {'thinking': str}
        """
        try:
            prompt = self._build_chat_prompt(query, chat_history)
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
                    
                    if '<think>' in token:
                        in_thinking = True
                        parts = token.split('<think>', 1)
                        if parts[0]:
                            yield {'token': parts[0]}
                        remainder = parts[1] if len(parts) > 1 else ""
                        
                        # Check if closing tag is also in this chunk
                        if '</think>' in remainder:
                            in_thinking = False
                            close_parts = remainder.split('</think>', 1)
                            thinking_buffer += close_parts[0]
                            if thinking_buffer:
                                yield {'thinking': thinking_buffer}
                                thinking_buffer = ""
                            if len(close_parts) > 1 and close_parts[1]:
                                yield {'token': close_parts[1]}
                            continue
                        
                        if remainder:
                            thinking_buffer += remainder
                        continue
                    
                    if '</think>' in token:
                        in_thinking = False
                        parts = token.split('</think>', 1)
                        thinking_buffer += parts[0]
                        if thinking_buffer:
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

