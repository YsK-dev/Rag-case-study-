"""Compare two search set-ups question by question, with a 95% bootstrap interval.

The difference A - B is computed per question, then the questions are resampled with
replacement (5,000 times, seed 0) to get the interval.

    python -m evaluation.compare dense+rerank dense
    python -m evaluation.compare hybrid+rerank dense+rerank
    python -m evaluation.compare app_default app_default_no_cutoff --metric hit3

    # The same set-up in two runs, for example before and after a code change:
    git show 4350ee6:evaluation/per_question_scifact.csv > before.csv
    python -m evaluation.compare bm25 bm25 --csv-b before.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_CSV = Path(__file__).resolve().parent / "per_question_scifact.csv"


def paired_bootstrap(diff: np.ndarray, resamples: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of the per-question differences and its 95% interval."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(resamples, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def scores(csv: Path, setup: str, metric: str) -> pd.Series:
    table = pd.read_csv(csv, dtype={"query_id": str})
    rows = table[table["setup"] == setup]
    if rows.empty:
        raise SystemExit(f"No set-up '{setup}' in {csv}. Set-ups: {sorted(table['setup'].unique())}")
    return rows.set_index("query_id")[metric]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--metric", default="ndcg10", choices=["ndcg10", "recall10", "mrr10", "hit3"])
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="per-question results for A (and B)")
    parser.add_argument("--csv-b", type=Path, help="per-question results for B, if from another run")
    args = parser.parse_args()

    a = scores(args.csv, args.a, args.metric)
    b = scores(args.csv_b or args.csv, args.b, args.metric)
    if set(a.index) != set(b.index):
        raise SystemExit("The two result sets do not cover the same questions.")
    diff = (a - b.reindex(a.index)).to_numpy(dtype=float)
    mean, low, high = paired_bootstrap(diff)
    print(f"{args.metric}: {args.a} {a.mean():.4f}, {args.b} {b.mean():.4f}, "
          f"difference {mean:+.4f} (95% interval {low:+.4f} to {high:+.4f}, {len(diff)} questions)")


if __name__ == "__main__":
    main()
