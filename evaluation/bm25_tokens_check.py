"""BM25 alone with the old word splitting (on spaces) and the new one (words without punctuation).

The new splitting was chosen while looking at SciFact, so this script also runs on other BEIR
datasets as an independent check. It uses rank_bm25 directly, as the app does, and needs no models.

    python -m evaluation.bm25_tokens_check --dataset fiqa
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from evaluation.metrics import hit_at_k, ndcg_at_k, recall_at_k
from evaluation.retrieval_eval import DATASETS, load

TOKENIZERS = {
    "split_on_spaces": lambda text: text.lower().split(),
    "words": lambda text: re.findall(r"\w+", text.lower()),  # the app's bm25_tokens()
}
OUT = Path(__file__).resolve().parent / "bm25_tokens_check.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="fiqa")
    args = parser.parse_args()

    corpus, questions, relevant = load(args.dataset)
    titles = corpus["title"].fillna("").astype(str)
    bodies = corpus["text"].fillna("").astype(str)
    texts = [(t + "\n" + b).strip() for t, b in zip(titles, bodies)]
    ids = corpus["_id"].astype(str).tolist()
    keep = [i for i, t in enumerate(texts) if t]
    texts, ids = [texts[i] for i in keep], [ids[i] for i in keep]
    print(f"{args.dataset}: {len(texts)} documents, {len(questions)} test questions", flush=True)

    result = {"dataset": args.dataset, "documents": len(texts), "test_questions": len(questions), "tokenizers": {}}
    for name, tokenize in TOKENIZERS.items():
        bm25 = BM25Okapi([tokenize(t) for t in texts])
        ndcg, recall, hit = [], [], []
        for qid, q in questions.items():
            s = bm25.get_scores(tokenize(q))
            top = np.argsort(s)[::-1][:10]
            ranked = [ids[i] for i in top if s[i] > 0]
            ndcg.append(ndcg_at_k(ranked, relevant[qid]))
            recall.append(recall_at_k(ranked, relevant[qid]))
            hit.append(hit_at_k(ranked, relevant[qid], 3))
        result["tokenizers"][name] = {"ndcg@10": round(float(np.mean(ndcg)), 4),
                                      "recall@10": round(float(np.mean(recall)), 4),
                                      "hit@3": round(float(np.mean(hit)), 4)}
        print(f"  {name}: {result['tokenizers'][name]}", flush=True)

    saved = json.loads(OUT.read_text()) if OUT.exists() else {}
    saved[args.dataset] = result
    OUT.write_text(json.dumps(saved, indent=2) + "\n")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
