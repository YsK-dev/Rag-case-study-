""" ReAct agent with RAG + basic web search.

1) Decide: rag | web | answer
2) Execute action
3) Answer using gathered context
"""

import os
import sys
from typing import Dict, Generator, List, Optional

import requests
from bs4 import BeautifulSoup

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rag_engine import RAGEngine
from llm_client import OllamaClient, sanitize_input


class WebRetriever:
    """Tiny web search wrapper using DuckDuckGo HTML."""

    def __init__(self, max_results: int = 3, timeout: int = 10) -> None:
        self.max_results = max_results
        self.timeout = timeout
        self.user_agent = (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )

    def search(self, query: str) -> List[Dict[str, str]]:
        url = "https://duckduckgo.com/html/"
        headers = {"User-Agent": self.user_agent}
        try:
            response = requests.get(
                url,
                params={"q": query, "kl": "us-en"},
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except Exception:
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        results: List[Dict[str, str]] = []
        for block in soup.find_all("div", class_="result"):
            title = block.find("a", class_="result__a")
            snippet = block.find("a", class_="result__snippet")
            link = block.find("a", class_="result__url")
            if not title or not snippet:
                continue
            results.append(
                {
                    "title": title.get_text(strip=True),
                    "snippet": snippet.get_text(strip=True),
                    "url": link.get("href", "") if link else "",
                }
            )
            if len(results) >= self.max_results:
                break

        return results


class ReactAgent:
    """Minimal ReAct agent: RAG + Web + Answer."""

    def __init__(
        self,
        rag_engine: RAGEngine,
        llm_client: OllamaClient,
        max_steps: int = 3,
        enable_web: bool = True,
        web_retriever: Optional[WebRetriever] = None,
    ) -> None:
        self.rag_engine = rag_engine
        self.llm_client = llm_client
        self.max_steps = max_steps
        self.enable_web = enable_web
        self.web = web_retriever or (WebRetriever() if enable_web else None)

    def _decide_prompt(self, question: str, context: str) -> str:
        return (
            "You are a tool-using assistant."
            " Decide next action. Choose one of: RAG, WEB, ANSWER.\n"
            "If you already have enough info, choose ANSWER.\n\n"
            f"Question: {question}\n\n"
            f"Context so far:\n{context}\n\n"
            "Respond with exactly one line: ACTION=<RAG|WEB|ANSWER>"
        )

    def _parse_action(self, decision: str) -> str:
        upper = decision.upper()
        if "ACTION=RAG" in upper:
            return "RAG"
        if "ACTION=WEB" in upper:
            return "WEB"
        if "ACTION=ANSWER" in upper:
            return "ANSWER"
        if "RAG" in upper:
            return "RAG"
        if "WEB" in upper:
            return "WEB"
        return "ANSWER"

    def _answer_prompt(self, question: str, context: str) -> str:
        return (
            "Answer using ONLY the context below."
            " If missing, say you don't have enough info.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

    def _rag_search(self, query: str, top_k: int = 3) -> str:
        chunks, _metas = self.rag_engine.retrieve(query, top_k=top_k)
        if not chunks:
            return "[RAG] No relevant documents found."
        return "\n\n".join(f"[RAG {i+1}] {c}" for i, c in enumerate(chunks))

    def _web_search(self, query: str) -> str:
        if not self.web:
            return "[WEB] Web search is disabled."
        results = self.web.search(query)
        if not results:
            return "[WEB] No results."
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"[WEB {i}] {r['title']} - {r['snippet']} ({r['url']})")
        return "\n".join(lines)

    def run(
        self,
        question: str,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> Generator[Dict, None, None]:
        cleaned, safe = sanitize_input(question)
        if not safe:
            yield {"error": "Prompt injection detected.", "finished": True}
            return
        question = cleaned.strip()
        if not question:
            yield {"error": "Empty question.", "finished": True}
            return

        context_parts: List[str] = []
        last_context = ""

        for step in range(1, self.max_steps + 1):
            decide_prompt = self._decide_prompt(question, "\n".join(context_parts))
            decision = self.llm_client.chat(decide_prompt, temperature=0.2)
            action = self._parse_action(decision)

            if action == "WEB" and not self.web:
                action = "ANSWER"

            yield {"step": step, "action": action}

            if action == "RAG":
                context_parts.append(self._rag_search(question))
            elif action == "WEB":
                context_parts.append(self._web_search(question))
            else:
                break

            current_context = "\n".join(context_parts)
            if current_context == last_context:
                break
            last_context = current_context

        if not context_parts:
            yield {
                "answer": "I don't have enough information to answer this question.",
                "finished": True,
            }
            return

        final_prompt = self._answer_prompt(question, "\n".join(context_parts))
        if stream:
            for token in self.llm_client.chat_stream(final_prompt, temperature=temperature):
                yield token
            yield {"finished": True}
            return

        answer = self.llm_client.chat(final_prompt, temperature=temperature)
        yield {"answer": answer, "finished": True}
