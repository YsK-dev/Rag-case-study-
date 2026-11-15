"""Choose the app's confidence cut-off on SciFact training claims, then check it on the test claims.

The cut-off decides when the app says "I cannot answer". To measure both sides of that choice,
each split (train, test) is run on two indexes:
  answerable     all 5,183 abstracts: every claim has a right abstract in the index;
  unanswerable   all abstracts except the ones that answer this split's claims: no claim has one.

Each claim goes through the app's default call (3 passages, 9 vector + 9 BM25 candidates,
reranker) with the cut-off switched off, and the reranker confidence of each kept passage is
recorded. Passages are kept in order of confidence, so the result for any cut-off t is simply the
passages with confidence >= t.

Score of a cut-off = mean of
  (a) Hit@3 on the answerable index (a claim with no passage counts as a miss), and
  (b) the share of claims with no passage on the unanswerable index (saying "I cannot answer").
The cut-off with the best score on the training claims is chosen (ties: the lower one) and then
reported on the test claims, next to the app's old 0.3.

    python -m evaluation.cutoff_tuning --embedder bge-small-en-v1.5
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder, SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_engine import EMBEDDERS, RERANKER, RAGEngine  # noqa: E402

from evaluation.retrieval_eval import load  # noqa: E402

CUTOFFS = [0.0, 0.01, 0.02, 0.03] + [round(0.05 * i, 2) for i in range(1, 20)]   # 0 ... 0.95
OLD_CUTOFF = 0.3
OUT_DIR = Path(__file__).resolve().parent


def add_passages(engine: RAGEngine, ids: list[str], texts: list[str], emb: np.ndarray) -> None:
    for start in range(0, len(ids), 2000):
        engine.collection.add(ids=ids[start:start + 2000], documents=texts[start:start + 2000],
                              embeddings=emb[start:start + 2000].tolist(),
                              metadatas=[{"source": i} for i in ids[start:start + 2000]])


def run_claims(engine: RAGEngine, questions: dict, relevant: dict, split: str, index: str) -> list[dict]:
    rows = []
    for qid, q in questions.items():
        _, metas = engine.retrieve(q, confidence_threshold=0.0)   # the app's default call, no cut-off
        rows.append({"split": split, "index": index, "query_id": qid,
                     "confidence": [m["confidence"] for m in metas],
                     "relevant": [relevant[qid].get(m["source"], 0) > 0 for m in metas]})
    return rows


def at_cutoff(rows: list[dict], t: float) -> dict:
    """Hit@3 on answerable, share with no passage on unanswerable, and their mean."""
    ans = [r for r in rows if r["index"] == "answerable"]
    una = [r for r in rows if r["index"] == "unanswerable"]
    hit = np.mean([any(rel for c, rel in zip(r["confidence"], r["relevant"]) if c >= t) for r in ans])
    answered = np.mean([any(c >= t for c in r["confidence"]) for r in ans])
    abstain = np.mean([not any(c >= t for c in r["confidence"]) for r in una])
    return {"cutoff": t, "hit@3_answerable": round(float(hit), 4), "answered_answerable": round(float(answered), 4),
            "no_passage_unanswerable": round(float(abstain), 4), "score": round(float((hit + abstain) / 2), 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--embedder", choices=sorted(EMBEDDERS), default="bge-small-en-v1.5")
    args = parser.parse_args()

    corpus, test_q, test_rel = load("scifact", "test")
    _, train_q, train_rel = load("scifact", "train")
    texts = [(t + "\n" + b).strip() for t, b in zip(corpus["title"].fillna("").astype(str),
                                                    corpus["text"].fillna("").astype(str))]
    ids = corpus["_id"].astype(str).tolist()

    spec = EMBEDDERS[args.embedder]
    embedder = SentenceTransformer(spec["repo"], revision=spec["revision"], device="cpu")
    reranker = CrossEncoder(RERANKER[0], revision=RERANKER[1], device="cpu")

    rows = []
    t_start = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        def engine(name):
            return RAGEngine(db_path=tmp, collection_name=name, embedding_model=embedder,
                             reranker=reranker, embedder=args.embedder)

        full = engine("all")
        emb = full.embed_passages(texts)          # once; the other indexes reuse these vectors
        add_passages(full, ids, texts, emb)
        print(f"index built in {time.time() - t_start:.0f} s", flush=True)
        for split, questions, relevant in (("train", train_q, train_rel), ("test", test_q, test_rel)):
            answers = {d for rel in relevant.values() for d, g in rel.items() if g > 0}
            keep = [i for i, d in enumerate(ids) if d not in answers]
            without = engine(f"without-{split}")
            add_passages(without, [ids[i] for i in keep], [texts[i] for i in keep], emb[keep])
            rows += run_claims(full, questions, relevant, split, "answerable")
            rows += run_claims(without, questions, relevant, split, "unanswerable")
            print(f"{split}: {len(questions)} claims, {len(answers)} answering abstracts removed for the "
                  f"unanswerable index; {time.time() - t_start:.0f} s", flush=True)

    sweep = {split: [at_cutoff([r for r in rows if r["split"] == split], t) for t in CUTOFFS]
             for split in ("train", "test")}
    best = max(sweep["train"], key=lambda x: (x["score"], -x["cutoff"]))
    chosen = best["cutoff"]
    result = {
        "embedder": args.embedder,
        "claims": {"train": len(train_q), "test": len(test_q)},
        "chosen_cutoff_on_train": chosen,
        "test_at_chosen": next(x for x in sweep["test"] if x["cutoff"] == chosen),
        "test_at_old_0.3": next(x for x in sweep["test"] if x["cutoff"] == OLD_CUTOFF),
        "test_without_cutoff": next(x for x in sweep["test"] if x["cutoff"] == 0.0),
        "sweep": sweep,
        "minutes": round((time.time() - t_start) / 60, 1),
    }
    (OUT_DIR / f"cutoff_{args.embedder}.json").write_text(json.dumps(result, indent=2) + "\n")
    pd.DataFrame(rows).to_json(OUT_DIR / f"cutoff_per_claim_{args.embedder}.jsonl", orient="records", lines=True)
    print(json.dumps({k: v for k, v in result.items() if k != "sweep"}, indent=2))


if __name__ == "__main__":
    main()
