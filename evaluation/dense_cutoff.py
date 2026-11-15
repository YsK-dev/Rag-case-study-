"""Vector search alone with a similarity cut-off, against the reranker path with its confidence cut-off.

Run after `evaluation.cutoff_tuning` (it reads that script's per-claim results for the
reranker path). Same claims, same two indexes per split:
  answerable     all 5,183 abstracts;
  unanswerable   all abstracts except the ones that answer this split's claims.

Vector path: the 3 nearest passages by cosine similarity (ChromaDB, the chosen embedding model),
no keyword search, no reranker. A passage is kept if its similarity is >= the cut-off.

1. How well each path's score tells answerable from unanswerable claims, without any cut-off:
   AUROC of the best kept passage's score (0.5 = no better than chance, 1 = perfect).
2. The vector path's cut-off is chosen on the training claims so that it refuses at least as
   many unanswerable training claims as the reranker path at the app's old cut-off (0.3).
   Both are then compared on the test claims: Hit@3 on the answerable index, share of claims
   with no passage on the unanswerable index, each with a paired 95% bootstrap interval.
3. For the record, the same paired differences on the training claims too, and without any
   cut-off on both paths (`difference_vector_minus_reranker`).

    python -m evaluation.dense_cutoff --embedder bge-small-en-v1.5
    python -m evaluation.dense_cutoff --saved   # recompute from the saved per-claim results
"""

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_engine import EMBEDDERS, RAGEngine  # noqa: E402

from evaluation.compare import paired_bootstrap  # noqa: E402
from evaluation.cutoff_tuning import OLD_CUTOFF, add_passages  # noqa: E402
from evaluation.retrieval_eval import load  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
TOP_K = 3


def auroc(pos: list[float], neg: list[float]) -> float:
    """Chance that a random answerable claim scores higher than a random unanswerable one."""
    p, n = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float((p > n).mean() + 0.5 * (p == n).mean())


def kept(row: dict, t: float, key: str) -> list[bool]:
    return [s >= t for s in row[key]]


def hit(row: dict, t: float, key: str) -> bool:
    return any(k and rel for k, rel in zip(kept(row, t, key), row["relevant"]))


def refused(row: dict, t: float, key: str) -> bool:
    return not any(kept(row, t, key))


def summary(rows: list[dict], t: float, key: str) -> dict:
    ans = [r for r in rows if r["index"] == "answerable"]
    una = [r for r in rows if r["index"] == "unanswerable"]
    return {"cutoff": t,
            "hit@3_answerable": round(float(np.mean([hit(r, t, key) for r in ans])), 4),
            "answered_answerable": round(float(np.mean([not refused(r, t, key) for r in ans])), 4),
            "no_passage_unanswerable": round(float(np.mean([refused(r, t, key) for r in una])), 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--embedder", choices=sorted(EMBEDDERS), default="bge-small-en-v1.5")
    parser.add_argument("--saved", action="store_true",
                        help="reuse the saved per-claim results instead of searching again "
                             "(timings are kept from the run that searched)")
    args = parser.parse_args()

    rerank_file = OUT_DIR / f"cutoff_per_claim_{args.embedder}.jsonl"
    rerank_rows = [json.loads(line) for line in open(rerank_file, encoding="utf-8")]

    corpus, test_q, test_rel = load("scifact", "test")
    _, train_q, train_rel = load("scifact", "train")
    out_json = OUT_DIR / f"dense_cutoff_{args.embedder}.json"
    out_rows = OUT_DIR / f"dense_per_claim_{args.embedder}.jsonl"
    if args.saved:
        rows = [json.loads(line) for line in open(out_rows, encoding="utf-8")]
        before = json.loads(out_json.read_text())
        report(args.embedder, rows, rerank_rows, len(train_q), len(test_q),
               before["vector_path_ms_per_claim_median"], before["minutes"], out_json)
        return
    texts = [(t + "\n" + b).strip() for t, b in zip(corpus["title"].fillna("").astype(str),
                                                    corpus["text"].fillna("").astype(str))]
    ids = corpus["_id"].astype(str).tolist()
    position = {d: i for i, d in enumerate(ids)}
    spec = EMBEDDERS[args.embedder]
    embedder = SentenceTransformer(spec["repo"], revision=spec["revision"], device="cpu")

    rows, ms = [], []
    t_start = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        def engine(name):
            return RAGEngine(db_path=tmp, collection_name=name, embedding_model=embedder,
                             reranker=object(), embedder=args.embedder)

        full = engine("all")
        emb = full.embed_passages(texts)
        add_passages(full, ids, texts, emb)
        print(f"index built in {time.time() - t_start:.0f} s", flush=True)
        for split, questions, relevant in (("train", train_q, train_rel), ("test", test_q, test_rel)):
            answers = {d for rel in relevant.values() for d, g in rel.items() if g > 0}
            keep = [i for i, d in enumerate(ids) if d not in answers]
            without = engine(f"without-{split}")
            add_passages(without, [ids[i] for i in keep], [texts[i] for i in keep], emb[keep])
            for index_name, eng in (("answerable", full), ("unanswerable", without)):
                for qid, q in questions.items():
                    t0 = time.perf_counter()
                    q_emb = eng.embed_query(q)
                    res = eng.collection.query(query_embeddings=q_emb.tolist(), n_results=TOP_K,
                                               include=["distances"])
                    ms.append((time.perf_counter() - t0) * 1000)
                    found = res["ids"][0]
                    sims = [1 - d / 2 for d in res["distances"][0]]   # squared L2 on unit vectors
                    exact = emb[[position[f] for f in found]] @ q_emb[0]
                    assert np.allclose(sims, exact, atol=1e-3), (qid, sims, exact)
                    rows.append({"split": split, "index": index_name, "query_id": qid, "similarity": sims,
                                 "relevant": [relevant[qid].get(f, 0) > 0 for f in found]})
            print(f"{split}: {len(questions)} claims done; {time.time() - t_start:.0f} s", flush=True)

    with open(out_rows, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    report(args.embedder, rows, rerank_rows, len(train_q), len(test_q),
           round(statistics.median(ms), 1), round((time.time() - t_start) / 60, 1), out_json)


def report(embedder: str, rows: list[dict], rerank_rows: list[dict], n_train: int, n_test: int,
           ms_median: float, minutes: float, out_json: Path) -> None:
    def part(rs, split):
        return [r for r in rs if r["split"] == split]

    def top(r, key):
        return max(r[key], default=0.0)

    separation = {}
    for split in ("train", "test"):
        separation[split] = {}
        for name, rs, key in (("vector_similarity", rows, "similarity"), ("reranker_confidence", rerank_rows, "confidence")):
            rs = part(rs, split)
            separation[split][name] = round(auroc([top(r, key) for r in rs if r["index"] == "answerable"],
                                                  [top(r, key) for r in rs if r["index"] == "unanswerable"]), 4)

    # Cut-off chosen on train: refuse at least as many unanswerable claims as the reranker path at 0.3.
    target = summary(part(rerank_rows, "train"), OLD_CUTOFF, "confidence")["no_passage_unanswerable"]
    una_train = sorted(top(r, "similarity") for r in part(rows, "train") if r["index"] == "unanswerable")
    candidates = sorted({round(s, 4) for s in una_train} | {round(s + 1e-4, 4) for s in una_train})
    chosen = next(t for t in candidates
                  if summary(part(rows, "train"), t, "similarity")["no_passage_unanswerable"] >= target)

    vec_test, rr_test = part(rows, "test"), part(rerank_rows, "test")

    def differences(split, t_vec, t_rr, measures):
        by_id = {(r["index"], r["query_id"]): r for r in part(rerank_rows, split)}
        out = {}
        for index_name, fn in measures:
            v = [r for r in part(rows, split) if r["index"] == index_name]
            diff = np.array([float(fn(r, t_vec, "similarity"))
                             - float(fn(by_id[(index_name, r["query_id"])], t_rr, "confidence")) for r in v])
            mean, low, high = paired_bootstrap(diff)
            out["hit@3_answerable" if index_name == "answerable" else "no_passage_unanswerable"] = {
                "difference": round(mean, 4), "interval_95": [round(low, 4), round(high, 4)], "claims": len(diff)}
        return out

    both = (("answerable", hit), ("unanswerable", refused))
    paired = differences("test", chosen, OLD_CUTOFF, both)
    every = {split: {"cutoffs_matched": differences(split, chosen, OLD_CUTOFF, both),
                     "no_cutoff": differences(split, 0.0, 0.0, (("answerable", hit),))}
             for split in ("train", "test")}

    result = {
        "embedder": embedder,
        "claims": {"train": n_train, "test": n_test},
        "auroc_answerable_vs_unanswerable": separation,
        "reranker_path_old_cutoff": {"cutoff": OLD_CUTOFF,
                                     "train": summary(part(rerank_rows, "train"), OLD_CUTOFF, "confidence"),
                                     "test": summary(rr_test, OLD_CUTOFF, "confidence")},
        "vector_path_chosen_on_train": {"cutoff": chosen,
                                        "train": summary(part(rows, "train"), chosen, "similarity"),
                                        "test": summary(vec_test, chosen, "similarity")},
        "vector_path_without_cutoff_test": summary(vec_test, 0.0, "similarity"),
        "test_difference_vector_minus_reranker": paired,
        "difference_vector_minus_reranker": every,
        "vector_path_ms_per_claim_median": ms_median,
        "minutes": minutes,
    }
    out_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
