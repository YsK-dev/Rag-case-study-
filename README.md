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

Choices below were made on SciFact's 809 training claims and then checked once on the 300 test
claims. That check is not fully independent: 196 of the test claims have an answering abstract
that also answers a training claim, and SciFact's test claims are part of the public MTEB
benchmark on which embedding models are usually compared. So the main choice was also checked
on a second dataset, FiQA, that played no part in it.

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
  paired bootstrap: the claims are drawn again at random 5,000 times, and both set-ups are always
  scored on the same claims. They are over the 300 test claims unless stated otherwise.
- **The embedding model was changed** from `all-MiniLM-L6-v2` to `bge-small-en-v1.5`. Three small
  models were compared with vector search alone. The rule, fixed in advance: the best nDCG@10 on
  SciFact's 809 training claims wins.

  | Embedding model | Training claims | Test claims | FiQA (648 questions) |
  |---|---|---|---|
  | `all-MiniLM-L6-v2` (the old one) | 0.660 | 0.645 | 0.369 |
  | `e5-small-v2` | 0.708 | 0.687 | not run |
  | **`bge-small-en-v1.5`** | **0.747** | **0.713** | **0.403** |

  nDCG@10 with vector search alone. On the SciFact test claims, the new model is +0.068 above the
  old one (interval +0.041 to +0.094) and Hit@3 goes from 68.3% to 73.0% (+4.7 points, interval
  +1.3 to +8.3). On [FiQA](https://huggingface.co/datasets/BeIR/fiqa) (57,600 financial forum
  posts and everyday questions about money), which played no part in the choice, it is +0.035
  (interval +0.017 to +0.053) and Hit@3 goes from 51.2% to 54.3% (+3.1 points, interval +0.2 to
  +6.2). The rule for FiQA, also fixed in advance, was to go back to the old model only if the new
  one was clearly worse there.
- **What it costs:** turning documents into vectors takes about 3 to 5 times longer (SciFact:
  6.9 minutes against 2.5; FiQA: 53 against 11, on this laptop), so uploads are slower. Each
  passage is still stored as 384 numbers, so the index does not grow. The model is a larger
  download (about 135 MB against 90 MB) and needs a short prefix before each question, which the
  engine adds. Searching takes 18 ms per claim instead of 10 ms.
- **With the new model the reranker adds nothing here.** Vector search alone scores 0.009 above
  vector search plus reranker (interval −0.018 to +0.035). With the old model the reranker
  lifted nDCG@10 by 0.046 (interval +0.016 to +0.075). BM25 adds nothing on top either (−0.007,
  interval −0.020 to +0.006). Both may help more on other kinds of documents; SciFact does not
  test that.
- **A bug found by this test.** BM25 used to split text on spaces only, so "cancer." and "cancer"
  were different words. Splitting into words raised BM25 alone from 0.560 to 0.652 (95% interval
  +0.066 to +0.119); the full pipeline, then with the old embedding model, barely moved (0.694 to
  0.696). The numbers before the fix
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
  app returns. Without the cut-off, the app's call (72.3%) and vector search alone (73.0%) are
  level (+0.7 points for vector search, interval −3.0 to +4.3).
- **The cut-off costs 16 claims a right passage:** 9 that get no passage at all and 7 whose list
  was cut to 1 or 2 passages. In all, 152 of the 300 claims get fewer than 3 passages. For the
  other 28 claims with no passage, the 3 passages the app would have used were all wrong, so not
  answering was the better outcome.

Whole abstracts are longer than the app's passages, so the share of dropped passages and the
timings may differ on your own documents.

### When should the app say "I cannot answer"?

Every SciFact claim has an answer among the abstracts, so the test above shows only what the
cut-off costs. To see what it saves, the claims were also run on a second index without the
abstracts that SciFact marks as answering them: there, the right outcome is to give no passage.

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

Two things make the refusal shares look worse than they may be for real use. SciFact does not
mark every abstract that answers a claim, so some passages counted as wrong here may in fact
answer it: on the second index, 99 of the 300 claims still get a top passage with a confidence
of 0.9 or more. And the passages left there are on the same subject as the claim, which makes
refusing harder than for a question about something the documents do not cover at all.

**The app keeps 0.3.** Before running, the rule for choosing the cut-off on the training claims
was set as the mean of the two columns. That rule picked 0.95, the top of the range tried, which
on test refuses 44% of the claims that do have an answer (Hit@3 47.7%). The rule itself is at
fault: refusing every claim scores 0.5 under it, more than answering every claim (0.38). So its
choice was not adopted. Which cut-off is right depends on how often users ask about things that
are not in their documents, and SciFact cannot say that.

### Is the reranker still worth it?

With the new embedding model, on SciFact, it does not clearly pay for itself. Here is vector
search alone (the 3 most similar abstracts, no reranker) against the app's call:

| Hit@3 | Training claims (809) | Test claims (300) |
|---|---|---|
| No cut-off: vector search alone | 79.0% | 73.0% |
| No cut-off: the app's call | 75.9% | 72.3% |
| Difference | +3.1 (+0.5 to +5.6) | +0.7 (−3.0 to +4.3) |
| With a cut-off: vector search alone | 70.1% | 64.3% |
| With a cut-off: the app's call (0.3) | 68.9% | 67.0% |
| Difference | +1.2 (−1.5 to +4.0) | −2.7 (−6.7 to +1.3) |

For vector search, the cut-off is a similarity score of 0.74 (1 means the same meaning). It was
chosen on the training claims so that it refuses as many claims without an answer as the app
does: on test, 30.0% against 30.3%. Its score tells the two cases apart about as well as the
reranker's confidence (AUROC 0.69 against 0.70).

- Without a cut-off, vector search alone is better on the training claims and level on test.
  With a cut-off, the two are level on both; on test, vector search found a right abstract a
  little less often, but the interval includes no difference.
- Vector search alone takes 18 ms per claim, against about 550 ms for the app's call: 30 times
  faster.

**The reranker stays the default for now.** The evidence is mixed and comes from one dataset of
scientific claims, while the reranker was trained on everyday web questions, which are closer to
what users ask. Whether to drop it is the next thing to test on a second dataset.

**A larger reranker did not help either.** Two candidates went into the app's call without the
cut-off, on the 809 training claims. The rule, fixed before running: switch only if the paired
interval against the current reranker is above 0.

| Reranker | Hit@3 (training claims) | Time per claim |
|---|---|---|
| `ms-marco-MiniLM-L-6-v2` (the app's) | 75.9% | 0.66 s |
| `ms-marco-MiniLM-L-12-v2` | 75.6% | 1.3 s |
| `bge-reranker-base` | 77.9% | 3.0 s |

`bge-reranker-base` is 2.0 points better (interval −0.1 to +4.1 over the training claims): close,
but not clearly above 0, and 4.5 times slower. So the app's reranker stays, and the test claims
were not used for this. None of the three beats vector search alone on these claims (79.0%).
Timings come from separate runs, so the app's call shows 0.66 s here and 0.55 s above.

Reproduce (downloads the pinned benchmark and models; times on a laptop CPU):

```bash
pip install -r requirements-eval.txt
python -m evaluation.embedder_choice --dataset scifact     # embedding model choice, about 12 min
python -m evaluation.embedder_choice --dataset fiqa        # check on FiQA
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

The first start downloads the embedding model and the reranker from Hugging Face (about 230 MB in all).

**Upgrading from an earlier version:** the default embedding model is now `bge-small-en-v1.5`, and
each embedding model has its own index. The old index is kept but not used, and the app prints a
warning at start-up when the new index is empty but uploaded files exist. Upload your documents
again, or start the app with `RAG_EMBEDDER=all-MiniLM-L6-v2` to keep using the old index. A
document deleted while one model is in use stays in the other model's index. Each index records
the model that built it, and the app refuses to open an index built with a different model.

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
second. The tests of the search metrics run there too.

The engine tests need the full requirements. They use the real ChromaDB and FAISS with a small
fake embedding model, so no model is downloaded. They check the question prefix, unit-length
vectors, one index per embedding model, the refusal of an index built with another model, and
the BM25 word splitting. CI runs both sets.

```bash
pip install -r requirements-dev.txt
RAG_TESTING=1 python -m pytest                        # light tests

pip install -r requirements.txt faiss-cpu
python -m pytest tests/test_embedders.py tests/test_bm25_tokens.py   # engine tests
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
