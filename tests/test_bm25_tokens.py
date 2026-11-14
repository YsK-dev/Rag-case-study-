"""Checks for the words BM25 sees. Needs the full requirements (the engine imports its models)."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("chromadb")
pytest.importorskip("sentence_transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_engine import bm25_tokens  # noqa: E402


def test_punctuation_is_not_part_of_a_word():
    assert bm25_tokens("Cancer. (cancer), CANCER!") == ["cancer", "cancer", "cancer"]


def test_turkish_letters_stay_in_words():
    assert bm25_tokens("Şirketin geliri: 5,2 milyon") == ["şirketin", "geliri", "5", "2", "milyon"]


def test_empty_text_gives_no_words():
    assert bm25_tokens("  ...  ") == []
