"""How the embedding model was chosen, and a check on a second dataset. Vector search only.

Rules, fixed before running:
- SciFact (`--dataset scifact`): three small models. The one with the best nDCG@10 on the 809
  training claims becomes the app's default. The 300 test claims are scored in the same run for
  the record; they play no part in the choice.
- FiQA (`--dataset fiqa`): the chosen model against the old one (all-MiniLM-L6-v2) on the 648
  test questions, a dataset that played no part in the choice. If the chosen model is clearly
  worse there (paired 95% interval for nDCG@10 entirely below 0), the app goes back to the old one.

Search here is exact: every passage's vector is compared with the question's (cosine similarity,
top 10). The app uses ChromaDB, whose search is approximate, so its numbers can differ slightly
(`evaluation.retrieval_eval` gives the app's).

    python -m evaluation.embedder_choice --dataset scifact
    python -m evaluation.embedder_choice --dataset fiqa
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_engine import EMBEDDERS  # noqa: E402

from evaluation.compare import paired_bootstrap  # noqa: E402
from evaluation.metrics import hit_at_k, ndcg_at_k  # noqa: E402
from evaluation.retrieval_eval import load  # noqa: E402

CANDIDATES = {
    "all-MiniLM-L6-v2": EMBEDDERS["all-MiniLM-L6-v2"],
    "bge-small-en-v1.5": EMBEDDERS["bge-small-en-v1.5"],
    "e5-small-v2": {"repo": "intfloat/e5-small-v2", "revision": "ffb93f3bd4047442299a41ebb6fa998a38507c52",
                    "query_prefix": "query: ", "passage_prefix": "passage: "},
}
OLD, NEW = "all-MiniLM-L6-v2", "bge-small-en-v1.5"
PLAN = {"scifact": {"models": list(CANDIDATES), "splits": ["train", "test"]},
        "fiqa": {"models": [OLD, NEW], "splits": ["test"]}}
K = 10
OUT_DIR = Path(__file__).resolve().parent


def top_k(q: np.ndarray, docs: np.ndarray, k: int = K) -> np.ndarray:
    """Indexes of the k most similar passages for each question, best first."""
    sim = q @ docs.T
    part = np.argpartition(-sim, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(sim, part, axis=1), axis=1)
    return np.take_along_axis(part, order, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", choices=sorted(PLAN), default="scifact")
    args = parser.parse_args()
    plan = PLAN[args.dataset]

    splits = {}
    for split in plan["splits"]:
        corpus, questions, relevant = load(args.dataset, split)
        splits[split] = (questions, relevant)
    texts = [(t + "\n" + b).strip() for t, b in zip(corpus["title"].fillna("").astype(str),
                                                    corpus["text"].fillna("").astype(str))]
    ids = np.array(corpus["_id"].astype(str).tolist())
    print(f"{args.dataset}: {len(texts)} passages, "
          + ", ".join(f"{s} {len(q)} questions" for s, (q, _) in splits.items()), flush=True)

    result = {"dataset": args.dataset, "passages": len(texts), "candidates": {}}
    rows = []
    for name in plan["models"]:
        spec = CANDIDATES[name]
        model = SentenceTransformer(spec["repo"], revision=spec["revision"], device="cpu")
        t0 = time.time()
        docs = model.encode([spec["passage_prefix"] + t for t in texts], batch_size=64,
                            normalize_embeddings=True)
        entry = {"repo": spec["repo"], "revision": spec["revision"],
                 "minutes_to_embed_passages": round((time.time() - t0) / 60, 1)}
        for split, (questions, relevant) in splits.items():
            qids = list(questions)
            q = model.encode([spec["query_prefix"] + questions[i] for i in qids], batch_size=64,
                             normalize_embeddings=True)
            for qid, best in zip(qids, top_k(q, docs)):
                ranked = ids[best].tolist()
                rows.append({"model": name, "split": split, "query_id": qid,
                             "ndcg10": ndcg_at_k(ranked, relevant[qid], K),
                             "hit3": hit_at_k(ranked, relevant[qid], 3)})
            mine = [r for r in rows if r["model"] == name and r["split"] == split]
            entry[split] = {"ndcg@10": round(float(np.mean([r["ndcg10"] for r in mine])), 4),
                            "hit@3": round(float(np.mean([r["hit3"] for r in mine])), 4),
                            "questions": len(mine)}
        result["candidates"][name] = entry
        print(name, entry, flush=True)

    per_q = pd.DataFrame(rows)
    result["difference_new_minus_old"] = {}
    for split in plan["splits"]:
        a = per_q[(per_q.model == NEW) & (per_q.split == split)].set_index("query_id")
        b = per_q[(per_q.model == OLD) & (per_q.split == split)].set_index("query_id").loc[a.index]
        result["difference_new_minus_old"][split] = {}
        for metric in ("ndcg10", "hit3"):
            mean, low, high = paired_bootstrap((a[metric] - b[metric]).to_numpy())
            result["difference_new_minus_old"][split][metric] = {
                "difference": round(mean, 4), "interval_95": [round(low, 4), round(high, 4)]}

    if args.dataset == "scifact":
        result["chosen_on_train"] = max(plan["models"],
                                        key=lambda n: result["candidates"][n]["train"]["ndcg@10"])
    else:
        upper = result["difference_new_minus_old"]["test"]["ndcg10"]["interval_95"][1]
        result["keep_new_model"] = bool(upper >= 0)
    (OUT_DIR / f"embedder_choice_{args.dataset}.json").write_text(json.dumps(result, indent=2) + "\n")
    per_q.round(5).to_csv(OUT_DIR / f"embedder_choice_per_question_{args.dataset}.csv", index=False)
    print(json.dumps({k: v for k, v in result.items() if k != "candidates"}, indent=2))


if __name__ == "__main__":
    main()
