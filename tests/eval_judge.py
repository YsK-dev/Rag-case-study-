#!/usr/bin/env python3
"""LLM-as-judge evaluation runner for the RAG API.

Usage examples:
  python tests/eval_judge.py --input data/eval_cases.json
  python tests/eval_judge.py --input data/eval_cases.jsonl --base-url http://localhost:8000
  python tests/eval_judge.py --input data/eval_cases.json --models openai/gpt-oss-120b,qwen/qwen3-32b

Input format (JSON or JSONL). Each case can include:
  - id: optional string
  - question: required string
  - answer: optional string (if missing, the script calls /api/chat)
  - top_k, model, temperature, vector_store: optional overrides for /api/chat
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

DEFAULT_GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "moonshotai/kimi-k2-instruct-0905",
    "qwen/qwen3-32b",
    "meta-llama/llama-guard-4-12b",
]
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
JUDGE_PROMPT = """
You will be given a user_question, retrieved_context, and system_answer.
Your task is to evaluate how well the system_answer addresses the user's question.

Evaluate on these criteria:
1. **Faithfulness** - Is the answer grounded in the retrieved context? (no hallucinations)
2. **Relevance** - Does the answer address the user's question?
3. **Completeness** - Is the answer thorough given the available context?

Give your answer as a float on a scale of 0 to 10, where:
- 0 = Answer is completely wrong, hallucinated, or irrelevant
- 5 = Answer is partially correct but has issues
- 10 = Answer is accurate, well-grounded, and fully addresses the question

Provide your feedback as follows:

Feedback:::
Faithfulness: (1-10, is it grounded in context?)
Relevance: (1-10, does it answer the question?)
Completeness: (1-10, is it thorough?)
Total rating: (your overall rating, as a float between 0 and 10)

Now here are the inputs:

Question: {question}

Retrieved Context:
{context}

System Answer: {answer}

Feedback:::
"""


def _load_api_key() -> str:
    env_key = os.getenv("GROQ_API_KEY")
    if env_key:
        return env_key.strip()

    repo_root = Path(__file__).resolve().parents[1]
    
    # Check .env file first
    env_path = repo_root / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("GROQ_API_KEY="):
                return line.split("=", 1)[1].strip()
    
    # Fall back to config/api_keys.txt
    key_path = repo_root / "config" / "api_keys.txt"
    if key_path.exists():
        for line in key_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("api_key="):
                return line.split("=", 1)[1].strip()

    raise RuntimeError(
        "Missing Groq API key. Set GROQ_API_KEY env var, add to .env, or config/api_keys.txt"
    )


def _load_cases(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.lower() == ".jsonl":
        cases: List[Dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
        return cases

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "cases" in data:
        return list(data["cases"])
    raise ValueError("JSON must be a list or an object with a 'cases' key")


def _call_chat_api(base_url: str, case: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "question": case["question"],
        "top_k": case.get("top_k", 3),
        "model": case.get("model", "gemini-1.5-flash"),
        "stream": False,
        "temperature": case.get("temperature", 0.7),
        "vector_store": case.get("vector_store", "chroma"),
    }
    url = base_url.rstrip("/") + "/api/chat"
    resp = requests.post(url, json=payload, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"/api/chat failed ({resp.status_code}): {resp.text}")
    return resp.json()


def _groq_chat(api_key: str, model: str, messages: List[Dict[str, str]]) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 600,
    }
    resp = requests.post(GROQ_ENDPOINT, headers=headers, json=payload, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"Groq API error ({resp.status_code}): {resp.text}")
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _parse_judge_score(text: str) -> float:
    match = re.search(r"Total rating:\s*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
    if not match:
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
    if not match:
        raise ValueError("Judge response did not include a numeric rating")
    score = float(match.group(1))
    if score < 0 or score > 10:
        raise ValueError(f"Judge rating out of range: {score}")
    return score


def _build_judge_messages(case: Dict[str, Any]) -> List[Dict[str, str]]:
    question = case["question"]
    answer = case["answer"]
    # Include retrieved sources/context for faithfulness evaluation
    sources = case.get("sources", [])
    if sources:
        # Extract text content from sources (handle both string and dict formats)
        context_parts = []
        for s in sources[:5]:  # Limit to top 5 sources
            if isinstance(s, dict):
                text = s.get("content", s.get("text", str(s)))
            else:
                text = str(s)
            context_parts.append(text[:500])  # Truncate each source
        context = "\n---\n".join(context_parts)
    else:
        context = "(No context retrieved)"
    
    content = JUDGE_PROMPT.format(question=question, context=context, answer=answer)
    return [{"role": "user", "content": content}]


def _summarize(results: List[Dict[str, Any]], models: List[str]) -> Dict[str, Any]:
    model_stats: Dict[str, Dict[str, Any]] = {}
    for model in models:
        scores = []
        errors = 0
        for r in results:
            judge = r.get("judges", {}).get(model, {})
            if "error" in judge:
                errors += 1
                continue
            if "score" in judge:
                scores.append(judge["score"])
        count = len(results)
        avg_score = sum(scores) / len(scores) if scores else 0.0
        model_stats[model] = {
            "average_score": round(avg_score, 2),
            "error_rate": round(errors / count, 2) if count else 0.0,
            "valid_count": len(scores),
        }

    overall_scores = []
    for r in results:
        agg = r.get("aggregate", {})
        if "score" in agg:
            overall_scores.append(agg["score"])
    overall_avg = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
    return {
        "count": len(results),
        "overall": {
            "average_score": round(overall_avg, 2),
        },
        "models": model_stats,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM-as-judge eval runner")
    parser.add_argument("--input", required=True, help="Path to JSON or JSONL eval cases")
    parser.add_argument("--output", help="Optional output JSONL file")
    parser.add_argument("--base-url", default="http://localhost:8000", help="RAG API base URL")
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_GROQ_MODELS),
        help="Comma-separated Groq model names",
    )
    args = parser.parse_args()

    cases = _load_cases(Path(args.input))
    api_key = _load_api_key()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        raise ValueError("No judge models provided. Use --models to set one or more models.")

    results: List[Dict[str, Any]] = []
    for idx, case in enumerate(cases, start=1):
        if "question" not in case:
            raise ValueError("Each case must include a 'question'")

        if "answer" not in case or not str(case["answer"]).strip():
            chat_resp = _call_chat_api(args.base_url, case)
            case["answer"] = chat_resp.get("answer", "")
            case["sources"] = chat_resp.get("sources", [])

        messages = _build_judge_messages(case)
        judges: Dict[str, Any] = {}
        latencies: Dict[str, float] = {}
        for model in models:
            start = time.time()
            try:
                raw = _groq_chat(api_key, model, messages)
                score = _parse_judge_score(raw)
                judges[model] = {"score": score}
            except Exception as exc:
                judges[model] = {"error": str(exc)}
            latencies[model] = round(time.time() - start, 2)

        valid_scores = [
            j["score"] for j in judges.values()
            if isinstance(j, dict) and "score" in j
        ]
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

        result = {
            "id": case.get("id", f"case_{idx}"),
            "question": case["question"],
            "answer": case["answer"],
            "judges": judges,
            "latency_sec": latencies,
            "aggregate": {
                "score": round(avg_score, 2),
            },
        }
        results.append(result)
        print(
            f"[{idx}/{len(cases)}] {result['id']} avg_score={result['aggregate']['score']}"
        )

    summary = _summarize(results, models)
    print("Summary:")
    print(json.dumps(summary, indent=2))

    if args.output:
        out_path = Path(args.output)
        with out_path.open("w", encoding="utf-8") as fh:
            for item in results:
                fh.write(json.dumps(item) + "\n")
        print(f"Wrote results to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
