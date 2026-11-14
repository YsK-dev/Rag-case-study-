"""Ranking metrics for one question. Plain Python, no dependencies.

`ranked` is the list of document ids the system returned, best first.
`relevant` maps each relevant document id to its relevance grade (1 or more).
"""

import math


def ndcg_at_k(ranked: list[str], relevant: dict[str, int], k: int = 10) -> float:
    """Normalised discounted cumulative gain at k, as in the BEIR benchmark."""
    dcg = sum(relevant.get(doc, 0) / math.log2(rank + 2) for rank, doc in enumerate(ranked[:k]))
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(grade / math.log2(rank + 2) for rank, grade in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked: list[str], relevant: dict[str, int], k: int = 10) -> float:
    """Share of the relevant documents that appear in the top k."""
    wanted = {doc for doc, grade in relevant.items() if grade > 0}
    if not wanted:
        return 0.0
    return len(wanted & set(ranked[:k])) / len(wanted)


def hit_at_k(ranked: list[str], relevant: dict[str, int], k: int = 3) -> float:
    """1.0 if at least one relevant document is in the top k, else 0.0."""
    return float(any(relevant.get(doc, 0) > 0 for doc in ranked[:k]))


def mrr_at_k(ranked: list[str], relevant: dict[str, int], k: int = 10) -> float:
    """1 / rank of the first relevant document in the top k, or 0.0 if there is none."""
    for rank, doc in enumerate(ranked[:k]):
        if relevant.get(doc, 0) > 0:
            return 1.0 / (rank + 1)
    return 0.0
