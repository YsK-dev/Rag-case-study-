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
  1. Candidates come from two searches: by meaning (vector search with `bge-small-en-v1.5` in
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
| Vector search only | 0.713 | 83.6% | 73.0% | 18 ms |
| BM25 keyword search only | 0.652 | 77.4% | 69.3% | 10 ms |
| Vector search, then reranker (30 candidates) | 0.704 | 83.5% | 72.3% | 1.1 s |
| Vector + BM25, then reranker (30 + 30 candidates, no cut-off) | 0.697 | 82.7% | 72.3% | 1.6 s |

- **nDCG@10** is the usual score for this benchmark: 1.0 means the right abstracts are at the
  top of the first 10 results, 0 means none of them is there. **Recall@10** is the share of
  right abstracts found in the first 10. **Hit@3** is the share of claims with a right abstract
  in the first 3. Time is the median per claim. All intervals below are 95% intervals from a
  paired bootstrap over the 300 claims.
- **The embedding model was changed** from `all-MiniLM-L6-v2` to `bge-small-en-v1.5`. The choice
  was made on SciFact's 809 training claims (nDCG@10 0.747 against 0.660), and the test claims
  were run once afterwards. On test, vector search alone went from 0.645 to 0.713 nDCG@10
  (+0.068, interval +0.041 to +0.094) and Hit@3 from 68.3% to 73.0% (+4.7 points, interval
  +1.3 to +8.3). It gives the same 384 numbers per passage, is a somewhat larger download
  (128 MB against 87 MB) and needs a short prefix before each question, which the engine adds.
- **With the new model the reranker adds nothing here.** Vector search alone scores 0.009 above
  vector search plus reranker (interval −0.018 to +0.035). With the old model the reranker
  lifted nDCG@10 by 0.046 (interval +0.016 to +0.075). BM25 adds nothing on top either (−0.007,
  interval −0.020 to +0.006). It may help more on other kinds of documents; SciFact does not
  test that.
- **A bug found by this test.** BM25 used to split text on spaces only, so "cancer." and "cancer"
  were different words. Splitting into words raised BM25 alone from 0.560 to 0.652 (95% interval
  +0.066 to +0.119); the full pipeline barely moved (0.694 to 0.696). The numbers before the fix
  are in commit `4350ee6`. The new splitting was chosen while looking at SciFact, so it was also
  checked on a dataset that played no part in that choice: on
  [FiQA](https://huggingface.co/datasets/BeIR/fiqa) (57,600 financial forum posts, 648 test
  questions), BM25 alone went from 0.159 to 0.217 nDCG@10 and Hit@3 from 23.0% to 30.9%
  (`python -m evaluation.bm25_tokens_check --dataset fiqa`, results in
  `evaluation/bm25_tokens_check.json`).

### The app's default call

The app asks for 3 passages (9 vector + 9 BM25 candidates, put in order by the reranker) and
drops passages with a reranker confidence below 0.3; when nothing is left, it says it cannot
answer. It takes about 0.55 s per claim.

| Default call | Hit@3 | Claims with no passage |
|---|---|---|
| With the 0.3 cut-off (as the app runs) | 67.0% | 12.3% (37 of 300) |
| Without the cut-off | 72.3% | 0% |

- **The new embedding model did not move this number.** With either model the app finds a right
  abstract for 67.0% of the claims (difference 0, interval −1 to +1 point). Here the reranker
  picks the final 3 passages, so a better vector search changes the candidates but not what the
  app returns. Vector search alone is 6 points better on Hit@3 (73.0%, interval +2 to +10).
- **The cut-off costs 16 claims a right passage:** 9 that get no passage at all and 7 whose list
  was cut to 1 or 2 passages. In all, 151 of the 300 claims get fewer than 3 passages. For the
  other 28 claims with no passage, the 3 passages the app would have used were all wrong, so not
  answering was the better outcome.

Whole abstracts are longer than the app's passages, so the share of dropped passages and the
timings may differ on your own documents.

### When should the app say "I cannot answer"?

Every SciFact claim has an answer among the abstracts, so the test above shows only what the
cut-off costs. To see what it saves, the claims were also run on a second index without any
of the abstracts that answer them: there, the right outcome is to give no passage.

The reranker's confidence tells the two cases apart only weakly. On the test claims, a claim
with an answer in the index gets a higher top confidence than a claim without one 70% of the
time (AUROC 0.70; 0.5 would be chance). The median top confidence is 0.97 with an answer and
0.71 without, so most claims without an answer still get confident passages.

| Cut-off | Hit@3 (answer in the index) | No passage given (no answer in the index) |
|---|---|---|
| none | 72.3% | 0% |
| 0.1 | 70.0% | 19.0% |
| **0.3 (the app)** | **67.0%** | **30.3%** |
| 0.5 | 64.0% | 36.0% |
| 0.7 | 58.7% | 49.3% |
| 0.9 | 52.0% | 67.0% |

300 test claims on each index. Each step up refuses more claims without an answer, but also
loses claims that had one.

**The app keeps 0.3.** Before running, the rule for choosing the cut-off on the training claims
was set as the mean of the two columns. That rule picked 0.95, the top of the range tried, which
on test refuses 44% of the claims that do have an answer (Hit@3 47.7%). The rule itself is at
fault: refusing every claim scores 0.5 under it, more than answering every claim (0.38). So its
choice was not adopted. Which cut-off is right depends on how often users ask about things that
are not in their documents, and SciFact cannot say that.

### Two ways that were tried and not adopted

- **Vector search alone, without the reranker.** Its similarity score was given a cut-off,
  chosen on the training claims to refuse as many claims without an answer as the app does
  (0.74 cosine similarity). On test: Hit@3 64.3% against the app's 67.0% (−2.7 points,
  interval −6.7 to +1.3), the same share of refusals (30.0% against 30.3%), and 18 ms per claim
  instead of about 550 ms. Its score separates the two cases about as well (AUROC 0.69). This
  is 30 times faster at similar quality, but the difference leans negative and SciFact is a
  single dataset, so the app keeps the reranker.
- **A larger reranker.** Two candidates went into the app's default call without the cut-off,
  on the 809 training claims. The rule, fixed before running: switch only if the paired
  interval against the current reranker is above 0.

  | Reranker | Hit@3 (training claims) | Time per claim |
  |---|---|---|
  | `ms-marco-MiniLM-L-6-v2` (the app's) | 75.9% | 0.66 s |
  | `ms-marco-MiniLM-L-12-v2` | 75.6% | 1.3 s |
  | `bge-reranker-base` | 77.9% | 3.0 s |

  `bge-reranker-base` is 2.0 points better (interval −0.1 to +4.1): close, but not clearly
  above 0, and 4.5 times slower. So the app's reranker stays and the test claims were not used
  for this.

Reproduce (downloads the pinned benchmark and models; times on a laptop CPU):

```bash
pip install -r requirements-eval.txt
python -m evaluation.retrieval_eval --dataset scifact      # first table, about 25 min
python -m evaluation.compare dense dense+rerank             # paired difference with 95% interval
python -m evaluation.compare dense dense --metric hit3 \
    --csv-b evaluation/per_question_scifact_all-MiniLM-L6-v2.csv   # new against old embedding model
python -m evaluation.cutoff_tuning                          # cut-off table, about 40 min
python -m evaluation.dense_cutoff                           # vector search with a cut-off, about 8 min
python -m evaluation.reranker_check                         # larger rerankers, about 75 min
```

Results for the old embedding model: `python -m evaluation.retrieval_eval --embedder all-MiniLM-L6-v2`.

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

The first start downloads the embedding model and the reranker from Hugging Face (about 220 MB in all).

**Upgrading from an earlier version:** the default embedding model is now `bge-small-en-v1.5`, and
each embedding model has its own index. The old index is kept but not used, so upload your
documents again, or start the app with `RAG_EMBEDDER=all-MiniLM-L6-v2` to keep using it.

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
| Embedding model | `RAG_EMBEDDER` environment variable (`EMBEDDERS` in `src/rag_engine.py`) | `bge-small-en-v1.5` |
| Reranker | `src/rag_engine.py` | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Passage size | `src/rag_engine.py` | about 500 characters, 100 overlap, joined up to 1,000 |
| Confidence cut-off | `retrieve()` in `src/rag_engine.py` | 0.3 |
| Passages per answer | `top_k` in the chat request | 3 |

To add an Ollama model, pull it and add an entry to `MODEL_REGISTRY` in `main.py`.

## Design notes

The [`docs/`](docs/) folder has the design notes written during the project: the problem, the
model layer, the search layer, the API, the architecture and a comparison of the tools considered.
