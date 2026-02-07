"""Lightweight test stubs for app dependencies."""

from typing import Dict, List, Tuple


class DummyRAGEngine:
    """Lightweight stub for unit tests."""

    def analyze_query(self, query: str) -> dict:
        return {
            "query_rewrite": query.strip(),
            "search_terms": [],
            "retrieval_strategy": "testing stub",
        }

    def retrieve(
        self, query: str, top_k: int = 3, **_kwargs
    ) -> Tuple[List[str], List[Dict]]:
        return [], []

    def ingest_file(self, _file_path: str) -> str:
        return "Test ingest skipped"

    def delete_document(self, _source_name: str) -> str:
        return "Deleted 0 chunks"

    def list_documents(self) -> List[Dict]:
        return []

    def get_stats(self) -> Dict:
        return {"backend": "dummy", "total_vectors": 0}


class DummyLLMClient:
    """Lightweight LLM stub for unit tests."""

    def __init__(self, model: str = "test-model") -> None:
        self.model = model
        self.last_thinking = ""

    def generate_answer(self, *_args, **_kwargs):
        return "Test response", ""

    def generate_answer_stream(self, *_args, **_kwargs):
        yield {"token": "Test response"}

    def chat(self, *_args, **_kwargs):
        return "Test response"

    def chat_stream(self, *_args, **_kwargs):
        yield {"token": "Test response"}
