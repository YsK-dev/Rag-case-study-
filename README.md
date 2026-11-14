# Local document Q&A (RAG)

Ask questions about your own documents and get answers with their sources. Everything runs on
your own machine: the documents, the search index and the language model (through Ollama).

This repository holds the back end, a FastAPI service. You can use it from the built-in API page
(`/docs`) or from any front end. The web front end shown in the design notes is not part of this
repository.

## What it does

- **Reads many file types:** PDF (with tables), Word, Excel, CSV, PowerPoint, HTML, Markdown and
  plain text.
- **Splits documents into passages:** text is cut into pieces of about 500 characters with 100
  characters of overlap, and small neighbouring pieces are joined up to 1,000 characters. Tables,
  code and formulas are kept whole. A passage that repeats, such as a page header, is kept at
  most twice.
- **Finds the right passages in two steps:**
  1. Candidates come from two searches: by meaning (vector search with `all-MiniLM-L6-v2` in
     ChromaDB, or FAISS) and by keywords (BM25).
  2. A cross-encoder (`ms-marco-MiniLM-L-6-v2`) reads each question–passage pair and puts the
     candidates in order. Passages it is not confident about (below 0.3) are dropped. If nothing
     is left, the app says it cannot answer instead of guessing.
- **Answers with a local model:** Qwen3 1.7B (fast) or Qwen3 4B Thinking, through Ollama. The
  answer streams word by word and lists the passages it used.
- **Basic safety checks:** common prompt-injection phrases in questions are rejected.
- **Conversations:** follow-up questions see the last 20 messages of the conversation.

## Run it

You need Python 3.10 or newer and [Ollama](https://ollama.com/).

```bash
git clone https://github.com/YsK-dev/Rag-case-study-.git
cd Rag-case-study-

# Language models
ollama serve                                      # leave running in its own terminal
ollama pull qwen3:1.7b
ollama pull pielee/qwen3-4b-thinking-2507_q8

# Back end
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload                         # http://localhost:8000, API page at /docs
```

Try it:

```bash
curl -F "file=@my_notes.pdf" http://localhost:8000/api/upload

curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What does the report say about costs?", "model": "qwen3-1.7b", "top_k": 3}'
```

## API

| Endpoint | Method | What it does |
|---|---|---|
| `/api/upload` | POST | Add a document |
| `/api/documents` | GET | List documents |
| `/api/documents/{filename}` | GET, DELETE | Download or remove one document |
| `/api/documents/{filename}/preview` | GET | Text of a non-PDF document, for a preview |
| `/api/chat` | POST | Ask a question (`"stream": true` for streaming) |
| `/api/chat/conversations` | GET | List conversations |
| `/api/chat/history/{conversation_id}` | GET, DELETE | Show or clear one conversation |
| `/api/feedback` | POST | Thumbs up or down for an answer |
| `/api/models` | GET | Available models |
| `/api/vector-stores` | GET | Available search back ends (ChromaDB, FAISS) |
| `/api/health` | GET | Health check |

## Tests

The unit tests check every endpoint in a light test mode that does not load any model, so they
run in about a second.

```bash
pip install -r requirements-dev.txt
RAG_TESTING=1 python -m pytest
```

## Scoring answers with judge models

`tests/eval_judge.py` sends your questions to the running API and asks larger models on Groq to
score each answer from 0 to 10 for faithfulness to the sources, relevance and completeness. You
supply the question file; the API key is read from the `GROQ_API_KEY` environment variable.

```bash
python tests/eval_judge.py --input my_questions.json --output judge_results.jsonl
```

## Settings

| Setting | Where | Default |
|---|---|---|
| Models | `MODEL_REGISTRY` in `main.py` | `qwen3-1.7b`, `qwen3-4b-thinking` |
| Embedding model | `src/rag_engine.py` | `all-MiniLM-L6-v2` |
| Reranker | `src/rag_engine.py` | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Passage size | `src/rag_engine.py` | about 500 characters, 100 overlap, joined up to 1,000 |
| Confidence cut-off | `retrieve()` in `src/rag_engine.py` | 0.3 |
| Passages per answer | `top_k` in the chat request | 3 |

To add an Ollama model, pull it and add an entry to `MODEL_REGISTRY` in `main.py`.

## Design notes

The [`docs/`](docs/) folder has the design notes written during the project: the problem, the
model layer, the search layer, the API, the architecture and a comparison of the tools considered.
