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
  characters of overlap, and small neighbouring pieces are joined up to 1,000 characters. Tables
  are kept as tables, 50 rows per passage with the header repeated. A
  passage that repeats, such as a page header, is kept at most twice.
- **Finds the right passages in two steps:**
  1. Candidates come from two searches: by meaning (vector search with `all-MiniLM-L6-v2` in
     ChromaDB; FAISS also works after `pip install faiss-cpu`) and by keywords (BM25).
  2. A cross-encoder (`ms-marco-MiniLM-L-6-v2`) reads each question–passage pair and puts the
     candidates in order. Passages it is not confident about (below 0.3) are dropped. If nothing
     is left, the app says it cannot answer instead of guessing.
- **Answers with a local model:** Qwen3 1.7B (fast) or Qwen3 4B Thinking, through Ollama. The
  answer streams word by word and lists the passages it used.
- **Basic safety checks:** common prompt-injection phrases in questions are rejected.
- **Conversations:** follow-up questions see the last 6 messages (3 questions and answers).

## How well the search works

Measured on [SciFact](https://huggingface.co/datasets/BeIR/scifact), a public benchmark from
BEIR: 5,183 scientific abstracts and 300 test claims (short scientific statements), each with the
abstracts that support or contradict it. Every abstract is stored whole as one passage in the
app's own search engine (`src/rag_engine.py`); reading files and splitting them into the app's
smaller passages is not part of this test. Laptop CPU (Apple M1), no GPU.

| Search set-up (10 results) | nDCG@10 | Recall@10 | Hit@3 | Time per claim |
|---|---|---|---|---|
| Vector search only | 0.645 | 78.3% | 68.3% | 10 ms |
| BM25 keyword search only | 0.652 | 77.4% | 69.3% | 10 ms |
| Vector search, then reranker (30 candidates) | 0.691 | 81.0% | 71.0% | 1.2 s |
| Vector + BM25, then reranker (30 + 30 candidates, no cut-off) | 0.696 | 83.1% | 72.3% | 1.9 s |

- **nDCG@10** is the usual score for this benchmark: 1.0 means the right abstracts are at the
  top of the first 10 results, 0 means none of them is there. **Recall@10** is the share of
  right abstracts found in the first 10. **Hit@3** is the share of claims with a right abstract
  in the first 3. Time is the median per claim.
- **The reranker is what helps.** It lifts nDCG@10 by 0.046 over vector search alone (95%
  interval +0.016 to +0.075, paired over the 300 claims), at about 1.2 s per claim on this CPU.
- **BM25 adds little on top of the reranker here:** +0.006 nDCG@10 (95% interval −0.009 to
  +0.022), for about 0.6 s more per claim at 10 results. It may help more on other kinds of
  documents; SciFact does not test that.
- **A bug found by this test.** BM25 used to split text on spaces only, so "cancer." and "cancer"
  were different words. Splitting into words raised BM25 alone from 0.560 to 0.652 (95% interval
  +0.066 to +0.119); the full pipeline barely moved (0.694 to 0.696). The numbers before the fix
  are in commit `4350ee6`. The new splitting was chosen while looking at SciFact, so it was also
  checked on a dataset that played no part in that choice: on
  [FiQA](https://huggingface.co/datasets/BeIR/fiqa) (57,600 financial forum posts, 648 test
  questions), BM25 alone went from 0.159 to 0.217 nDCG@10 and Hit@3 from 23.0% to 30.9%
  (`python -m evaluation.bm25_tokens_check --dataset fiqa`, results in
  `evaluation/bm25_tokens_check.json`).

**The app's default call** asks for 3 passages (9 vector + 9 BM25 candidates) and drops passages
with a reranker confidence below 0.3; when nothing is left, the app says it cannot answer. It takes
about 0.7 s per claim.

| Default call | Hit@3 | Claims with no passage |
|---|---|---|
| With the 0.3 cut-off (as the app runs) | 67.0% | 12.0% (36 of 300) |
| Without the cut-off | 72.7% | 0% |

The cut-off costs 17 claims a right passage: 10 that get no passage at all and 7 whose list was
cut to 1 or 2 passages. In all, 156 of the 300 claims get fewer than 3 passages. For the other
26 claims with no passage, the 3 passages the app would have used were all wrong, so not
answering was the better outcome. Every SciFact claim has an answer in the abstracts, so this test
shows what the cut-off costs, not how often it saves the app from answering a question the
documents cannot answer. Whole abstracts are longer than the app's passages, so the share of
dropped passages and the timings may differ on your own documents.

Reproduce (downloads the pinned benchmark and models, about 25 minutes on a laptop CPU):

```bash
pip install -r requirements-eval.txt
python -m evaluation.retrieval_eval --dataset scifact     # results.json, per_question_scifact.csv
python -m evaluation.compare dense+rerank dense            # paired difference with 95% interval
```

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

The first start downloads the embedding model and the reranker from Hugging Face (about 180 MB in all).

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

The unit tests check the main endpoints (chat, streaming, upload, documents, history, feedback)
and input checks in a light test mode that does not load any model, so they run in about a
second. The tests of the search metrics run there too; the BM25 word test runs
when the full requirements are installed.

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
