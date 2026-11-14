"""Checks for the ranking metrics on small hand-made cases."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics import hit_at_k, mrr_at_k, ndcg_at_k, recall_at_k  # noqa: E402


def test_perfect_ranking_scores_one():
    relevant = {"a": 1, "b": 1}
    assert ndcg_at_k(["a", "b", "c"], relevant) == pytest.approx(1.0)
    assert recall_at_k(["a", "b", "c"], relevant) == 1.0
    assert hit_at_k(["a", "b", "c"], relevant, k=1) == 1.0
    assert mrr_at_k(["a", "b", "c"], relevant) == 1.0


def test_nothing_relevant_found_scores_zero():
    relevant = {"x": 1}
    assert ndcg_at_k(["a", "b"], relevant) == 0.0
    assert recall_at_k(["a", "b"], relevant) == 0.0
    assert hit_at_k(["a", "b"], relevant) == 0.0
    assert mrr_at_k(["a", "b"], relevant) == 0.0


def test_ndcg_one_relevant_at_rank_two():
    # DCG = 1 / log2(3); the ideal ranking puts it first, IDCG = 1.
    assert ndcg_at_k(["a", "x", "b"], {"x": 1}) == pytest.approx(1 / math.log2(3))


def test_ndcg_uses_grades():
    relevant = {"a": 2, "b": 1}
    worse_first = ndcg_at_k(["b", "a"], relevant)
    better_first = ndcg_at_k(["a", "b"], relevant)
    assert better_first == pytest.approx(1.0)
    assert worse_first < better_first


def test_cut_off_at_k():
    relevant = {"x": 1}
    ranked = ["a", "b", "c", "x"]
    assert hit_at_k(ranked, relevant, k=3) == 0.0
    assert hit_at_k(ranked, relevant, k=4) == 1.0
    assert recall_at_k(ranked, relevant, k=3) == 0.0
    assert mrr_at_k(ranked, relevant, k=4) == pytest.approx(0.25)


def test_recall_counts_share_of_relevant():
    assert recall_at_k(["a", "c"], {"a": 1, "b": 1}) == 0.5


def test_zero_grade_is_not_relevant():
    relevant = {"a": 0, "b": 1}
    assert hit_at_k(["a"], relevant, k=1) == 0.0
    assert recall_at_k(["a", "b"], relevant) == 1.0
