"""Would a stronger reranker help? Chosen on the SciFact training claims, checked once on the test claims.

Rules, fixed before running:
- Candidates: the app's reranker (ms-marco-MiniLM-L-6-v2), the larger ms-marco-MiniLM-L-12-v2,
  and bge-reranker-base.
- Each one goes into the app's default call (3 passages from 9 vector + 9 keyword candidates)
  with the confidence cut-off switched off, on the index with all 5,183 abstracts.
- Choice on the 809 training claims by Hit@3 (a right abstract among the 3 passages). A
  candidate replaces the app's reranker only if its paired 95% bootstrap interval against the
  app's reranker is above 0 on the training claims.
- If one is chosen, the test claims (300) are run once: Hit@3 against the app's reranker, and how
  well its confidence tells answerable from unanswerable claims (AUROC, using the index without
  the answering abstracts, as in `evaluation.cutoff_tuning`).

The app's reranker results on test come from `evaluation.cutoff_tuning` (same call, same index),
so run that first.

    python -m evaluation.reranker_check --embedder bge-small-en-v1.5
"""

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_engine import EMBEDDERS, RERANKER, RAGEngine  # noqa: E402

from evaluation.compare import paired_bootstrap  # noqa: E402
from evaluation.cutoff_tuning import add_passages  # noqa: E402
from evaluation.dense_cutoff import auroc  # noqa: E402
from evaluation.retrieval_eval import load  # noqa: E402

CANDIDATES = {
    "ms-marco-MiniLM-L-6-v2": RERANKER,
    "ms-marco-MiniLM-L-12-v2": ("cross-encoder/ms-marco-MiniLM-L-12-v2", "7b0235231ca2674cb8ca8f022859a6eba2b1c968"),
    "bge-reranker-base": ("BAAI/bge-reranker-base", "2cfc18c9415c912f9d8155881c133215df768a70"),
}
CURRENT = "ms-marco-MiniLM-L-6-v2"
OUT_DIR = Path(__file__).resolve().parent


def load_reranker(name: str) -> CrossEncoder:
    repo, revision = CANDIDATES[name]
    # Raw scores: the app turns them into a confidence itself (sigmoid).
    return CrossEncoder(repo, revision=revision, device="cpu", activation_fn=torch.nn.Identity())


def run(engine: RAGEngine, questions: dict, relevant: dict) -> tuple[dict, list[float]]:
    rows, ms = {}, []
    for qid, q in questions.items():
        t0 = time.perf_counter()
        _, metas = engine.retrieve(q, confidence_threshold=0.0)
        ms.append((time.perf_counter() - t0) * 1000)
        rows[qid] = {"confidence": [m["confidence"] for m in metas],
                     "relevant": [relevant[qid].get(m["source"], 0) > 0 for m in metas]}
    return rows, ms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--embedder", choices=sorted(EMBEDDERS), default="bge-small-en-v1.5")
    args = parser.parse_args()

    current_rows = [json.loads(line) for line in
                    open(OUT_DIR / f"cutoff_per_claim_{args.embedder}.jsonl", encoding="utf-8")]
    corpus, test_q, test_rel = load("scifact", "test")
    _, train_q, train_rel = load("scifact", "train")
    texts = [(t + "\n" + b).strip() for t, b in zip(corpus["title"].fillna("").astype(str),
                                                    corpus["text"].fillna("").astype(str))]
    ids = corpus["_id"].astype(str).tolist()
    spec = EMBEDDERS[args.embedder]
    embedder = SentenceTransformer(spec["repo"], revision=spec["revision"], device="cpu")

    t_start = time.time()
    result = {"embedder": args.embedder, "candidates": {k: list(v) for k, v in CANDIDATES.items()},
              "train": {}, "test": {}}
    with tempfile.TemporaryDirectory() as tmp:
        def engine(name, reranker):
            return RAGEngine(db_path=tmp, collection_name=name, embedding_model=embedder,
                             reranker=reranker, embedder=args.embedder)

        full = engine("all", load_reranker(CURRENT))
        emb = full.embed_passages(texts)
        add_passages(full, ids, texts, emb)
        print(f"index built in {time.time() - t_start:.0f} s", flush=True)

        train_hits = {}
        for name in CANDIDATES:
            full.reranker = load_reranker(name)
            rows, ms = run(full, train_q, train_rel)
            train_hits[name] = np.array([float(any(rows[q]["relevant"])) for q in train_q])
            result["train"][name] = {"hit@3": round(float(train_hits[name].mean()), 4),
                                     "ms_per_claim_median": round(statistics.median(ms), 1)}
            print(name, result["train"][name], f"{time.time() - t_start:.0f} s", flush=True)

        # The app's reranker here must give the same results as in cutoff_tuning.
        stored = {r["query_id"]: any(r["relevant"]) for r in current_rows
                  if r["split"] == "train" and r["index"] == "answerable"}
        assert all(stored[q] == bool(h) for q, h in zip(train_q, train_hits[CURRENT])), "results differ"

        for name in CANDIDATES:
            if name != CURRENT:
                mean, low, high = paired_bootstrap(train_hits[name] - train_hits[CURRENT])
                result["train"][name]["difference_vs_current"] = {
                    "difference": round(mean, 4), "interval_95": [round(low, 4), round(high, 4)]}
        winners = [n for n in CANDIDATES if n != CURRENT
                   and result["train"][n]["difference_vs_current"]["interval_95"][0] > 0]
        chosen = max(winners, key=lambda n: result["train"][n]["hit@3"]) if winners else CURRENT
        result["chosen_on_train"] = chosen

        if chosen != CURRENT:
            full.reranker = load_reranker(chosen)
            answers = {d for rel in test_rel.values() for d, g in rel.items() if g > 0}
            keep = [i for i, d in enumerate(ids) if d not in answers]
            without = engine("without-test", full.reranker)
            add_passages(without, [ids[i] for i in keep], [texts[i] for i in keep], emb[keep])
            ans, ms = run(full, test_q, test_rel)
            una, _ = run(without, test_q, test_rel)
            cur_ans = {r["query_id"]: r for r in current_rows if r["split"] == "test" and r["index"] == "answerable"}
            cur_una = {r["query_id"]: r for r in current_rows if r["split"] == "test" and r["index"] == "unanswerable"}
            new_hit = np.array([float(any(ans[q]["relevant"])) for q in test_q])
            cur_hit = np.array([float(any(cur_ans[q]["relevant"])) for q in test_q])
            mean, low, high = paired_bootstrap(new_hit - cur_hit)

            def top(r):
                return max(r["confidence"], default=0.0)

            result["test"] = {
                chosen: {"hit@3": round(float(new_hit.mean()), 4),
                         "auroc_answerable_vs_unanswerable": round(auroc([top(ans[q]) for q in test_q],
                                                                         [top(una[q]) for q in test_q]), 4),
                         "ms_per_claim_median": round(statistics.median(ms), 1)},
                CURRENT: {"hit@3": round(float(cur_hit.mean()), 4),
                          "auroc_answerable_vs_unanswerable": round(auroc([top(cur_ans[q]) for q in test_q],
                                                                          [top(cur_una[q]) for q in test_q]), 4)},
                "difference": {"difference": round(mean, 4), "interval_95": [round(low, 4), round(high, 4)],
                               "claims": len(test_q)},
            }
    result["minutes"] = round((time.time() - t_start) / 60, 1)
    (OUT_DIR / f"reranker_check_{args.embedder}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
