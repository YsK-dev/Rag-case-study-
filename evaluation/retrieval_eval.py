"""Measure how well the app's search finds the right passage, on public BEIR benchmarks.

Every document of the benchmark is stored as one passage in the app's own engine
(`src/rag_engine.py`, ChromaDB backend). The steps that read files and split them into
chunks are not part of this test. Then every test question goes through these set-ups:

  dense           vector search only (all-MiniLM-L6-v2 in ChromaDB)
  bm25            keyword search only (the app's BM25)
  dense+rerank    the app's retrieve() with BM25 switched off: vector candidates, then the
                  cross-encoder puts them in order
  hybrid+rerank   the app's retrieve() as it is: vector and keyword candidates, then the
                  cross-encoder (top_k=10, so 30 + 30 candidates; no confidence cut-off)

Scores: nDCG@10 (the standard BEIR score), Recall@10, MRR@10, and Hit@3 (a relevant passage
is among the first 3).

Then the app's default call, retrieve(question), which keeps 3 passages and drops passages
whose reranker confidence is below 0.3. When nothing is left the app does not answer. This is
scored with Hit@3 (no passage counts as a miss) and the share of questions with no passage,
and compared with the same call without the cut-off.

Run from the repository root:
    python -m evaluation.retrieval_eval --dataset scifact
"""

import argparse
import json
import platform
import statistics
import sys
import tempfile
import time
from importlib.metadata import version
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from sentence_transformers import CrossEncoder, SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_engine import RAGEngine  # noqa: E402

from evaluation.metrics import hit_at_k, mrr_at_k, ndcg_at_k, recall_at_k  # noqa: E402

# Pinned versions, so the numbers can be reproduced.
DATASETS = {
    "scifact": {
        "corpus": ("BeIR/scifact", "b3b5335604bf5ee3c4447671af975ea25143d4f5"),
        "qrels": ("BeIR/scifact-qrels", "2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029"),
    },
    "fiqa": {
        "corpus": ("BeIR/fiqa", "979c07a7cb5ccc6ca009792241fa1250b98055dd"),
        "qrels": ("BeIR/fiqa-qrels", "252958f2d646e22cab6d0c72dd3f0d5de6d0655a"),
    },
}
EMBEDDER = ("sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41")
RERANKER = ("cross-encoder/ms-marco-MiniLM-L-6-v2", "233902d25c440f23af6f7d6e94d2946bac0bee0a")
K = 10
APP_TOP_K = 3
OUT_DIR = Path(__file__).resolve().parent


def load(name: str, split: str = "test") -> tuple[pd.DataFrame, dict[str, str], dict[str, dict[str, int]]]:
    """Corpus, test questions and their relevant documents for one BEIR dataset."""
    (crepo, crev), (qrepo, qrev) = DATASETS[name]["corpus"], DATASETS[name]["qrels"]

    def get(repo, file, rev):
        return hf_hub_download(repo, file, repo_type="dataset", revision=rev)

    corpus = pd.read_parquet(get(crepo, "corpus/corpus-00000-of-00001.parquet", crev))
    queries = pd.read_parquet(get(crepo, "queries/queries-00000-of-00001.parquet", crev))
    qrels = pd.read_csv(get(qrepo, f"{split}.tsv", qrev), sep="\t", dtype=str)
    qrels["score"] = qrels["score"].astype(int)

    relevant: dict[str, dict[str, int]] = {}
    for row in qrels.itertuples(index=False):
        relevant.setdefault(row[0], {})[row[1]] = row[2]
    qtext = dict(zip(queries["_id"].astype(str), queries["text"]))
    questions = {qid: qtext[qid] for qid in relevant if qid in qtext}
    return corpus, questions, relevant


def build_engine(corpus: pd.DataFrame, embedder, reranker, db_path: str, name: str) -> tuple[RAGEngine, int]:
    """Store every document as one passage, with its id as the source name."""
    engine = RAGEngine(db_path=db_path, collection_name=f"beir_{name}",
                       embedding_model=embedder, reranker=reranker)
    titles = corpus["title"].fillna("").astype(str)
    bodies = corpus["text"].fillna("").astype(str)
    texts = [(t + "\n" + b).strip() for t, b in zip(titles, bodies)]
    ids = corpus["_id"].astype(str).tolist()
    keep = [i for i, t in enumerate(texts) if t]
    texts, ids = [texts[i] for i in keep], [ids[i] for i in keep]
    step = 2000
    for start in range(0, len(texts), step):
        chunk_texts, chunk_ids = texts[start:start + step], ids[start:start + step]
        emb = embedder.encode(chunk_texts, batch_size=64).tolist()
        engine.collection.add(ids=chunk_ids, documents=chunk_texts, embeddings=emb,
                              metadatas=[{"source": i} for i in chunk_ids])
        print(f"  stored {start + len(chunk_texts)}/{len(texts)}", flush=True)
    return engine, len(texts)


def sources(metas) -> list[str]:
    return [m["source"] for m in metas]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="scifact")
    parser.add_argument("--limit", type=int, default=0, help="only the first N questions (quick check)")
    args = parser.parse_args()

    corpus, questions, relevant = load(args.dataset)
    if args.limit:
        questions = dict(list(questions.items())[:args.limit])
    print(f"{args.dataset}: {len(corpus)} documents, {len(questions)} test questions", flush=True)

    embedder = SentenceTransformer(EMBEDDER[0], revision=EMBEDDER[1], device="cpu")
    reranker = CrossEncoder(RERANKER[0], revision=RERANKER[1], device="cpu")

    with tempfile.TemporaryDirectory() as tmp:
        engine, n_stored = build_engine(corpus, embedder, reranker, tmp, args.dataset)

        def dense(q):
            res = engine.collection.query(query_embeddings=embedder.encode([q]).tolist(), n_results=K)
            return res["ids"][0]

        def bm25(q):
            return sources(engine._bm25_search(q, top_n=K)[1])

        def hybrid_rerank(q):
            return sources(engine.retrieve(q, top_k=K, confidence_threshold=0.0, max_tokens=10**9)[1])

        bm25_search = engine._bm25_search

        def dense_rerank(q):
            engine._bm25_search = lambda *_a, **_k: ([], [])
            try:
                return hybrid_rerank(q)
            finally:
                engine._bm25_search = bm25_search

        def app_default(q):
            return sources(engine.retrieve(q)[1])

        def app_no_cutoff(q):
            return sources(engine.retrieve(q, top_k=APP_TOP_K, confidence_threshold=0.0)[1])

        setups = {"dense": dense, "bm25": bm25, "dense+rerank": dense_rerank,
                  "hybrid+rerank": hybrid_rerank, "app_default": app_default,
                  "app_default_no_cutoff": app_no_cutoff}

        # Warm-up, so that one-off loading does not count in the timings.
        for fn in setups.values():
            fn("warm up")

        rows = []
        for setup, fn in setups.items():
            t_all = time.perf_counter()
            for qid, q in questions.items():
                t0 = time.perf_counter()
                ranked = fn(q)
                ms = (time.perf_counter() - t0) * 1000
                rel = relevant[qid]
                rows.append({"setup": setup, "query_id": qid, "returned": len(ranked), "ms": ms,
                             "ndcg10": ndcg_at_k(ranked, rel, K), "recall10": recall_at_k(ranked, rel, K),
                             "mrr10": mrr_at_k(ranked, rel, K), "hit3": hit_at_k(ranked, rel, 3)})
            print(f"  {setup}: {time.perf_counter() - t_all:.0f} s", flush=True)

    per_q = pd.DataFrame(rows)
    summary = {}
    for setup, g in per_q.groupby("setup", sort=False):
        summary[setup] = {
            "ndcg@10": round(g["ndcg10"].mean(), 4),
            "recall@10": round(g["recall10"].mean(), 4),
            "mrr@10": round(g["mrr10"].mean(), 4),
            "hit@3": round(g["hit3"].mean(), 4),
            "no_passage_share": round(float((g["returned"] == 0).mean()), 4),
            "ms_per_question_median": round(statistics.median(g["ms"]), 1),
        }

    result = {
        "dataset": args.dataset,
        "documents_in_corpus": len(corpus),
        "documents_stored": n_stored,
        "test_questions": len(questions),
        "full_test_set": not args.limit,
        "sources": {"corpus": DATASETS[args.dataset]["corpus"], "qrels": DATASETS[args.dataset]["qrels"],
                    "embedder": EMBEDDER, "reranker": RERANKER},
        "machine": f"{platform.machine()}, {platform.system()} {platform.release()}, CPU only",
        "packages": {p: version(p) for p in ["chromadb", "sentence-transformers", "torch", "rank-bm25"]},
        "setups": summary,
    }
    print(json.dumps(summary, indent=2))
    if args.limit:
        print("Quick check only (--limit): nothing saved.")
        return
    all_results_path = OUT_DIR / "results.json"
    all_results = json.loads(all_results_path.read_text()) if all_results_path.exists() else {}
    all_results[args.dataset] = result
    all_results_path.write_text(json.dumps(all_results, indent=2) + "\n")
    per_q.round(5).to_csv(OUT_DIR / f"per_question_{args.dataset}.csv", index=False)
    print(f"Saved {all_results_path} and per_question_{args.dataset}.csv")


if __name__ == "__main__":
    main()
