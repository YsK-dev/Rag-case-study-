"""How the engine handles its embedding model, with a small fake model (nothing is downloaded).

Needs the full requirements (the engine imports its libraries); the FAISS tests also need faiss-cpu.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("chromadb")
pytest.importorskip("sentence_transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import rag_engine  # noqa: E402
from rag_engine import EMBEDDERS, RAGEngine, resolve_embedder  # noqa: E402

BGE, MINILM = "bge-small-en-v1.5", "all-MiniLM-L6-v2"


class FakeModel:
    """Turns a text into a few numbers that are not unit length; remembers what it was given."""

    def __init__(self, dim: int = 4):
        self.dim, self.seen = dim, []

    def encode(self, texts, normalize_embeddings=False, **_):
        self.seen.extend(texts)
        out = np.array([[len(t) % 7 + 1.0, t.count("a") + 1.0, t.count("e") + 2.0, 3.0, 1.0][: self.dim]
                        for t in texts])
        return out / np.linalg.norm(out, axis=1, keepdims=True) if normalize_embeddings else out

    def get_sentence_embedding_dimension(self):
        return self.dim


class FakeReranker:
    def predict(self, pairs, **_):
        return np.zeros(len(pairs))


def engine(tmp_path, embedder=BGE, backend="chroma", model=None, name="docs"):
    return RAGEngine(db_path=str(tmp_path), collection_name=name, backend=backend,
                     embedding_model=model or FakeModel(), reranker=FakeReranker(), embedder=embedder)


def test_query_prefix_goes_on_questions_only(tmp_path):
    model = FakeModel()
    e = engine(tmp_path, model=model)
    e.embed_passages(["a passage"])
    e.embed_query("a question")
    assert model.seen == ["a passage", EMBEDDERS[BGE]["query_prefix"] + "a question"]


def test_vectors_are_unit_length(tmp_path):
    e = engine(tmp_path)
    assert np.allclose(np.linalg.norm(e.embed_passages(["one", "two words"]), axis=1), 1.0)
    assert np.allclose(np.linalg.norm(e.embed_query("three"), axis=1), 1.0)


def test_each_model_has_its_own_collection_and_the_old_name_stays(tmp_path):
    assert engine(tmp_path, BGE).collection.name == f"docs-{BGE}"
    assert engine(tmp_path, MINILM).collection.name == "docs"


def test_collection_records_its_model(tmp_path):
    meta = engine(tmp_path).collection.metadata
    assert meta["embedder"] == BGE and meta["embedder_revision"] == EMBEDDERS[BGE]["revision"]


def test_collection_built_with_another_revision_is_refused(tmp_path):
    e = engine(tmp_path)
    e.collection.modify(metadata={**e.collection.metadata, "embedder_revision": "0" * 40})
    with pytest.raises(ValueError, match="was built with"):
        engine(tmp_path)


def test_mismatch_is_found_before_any_model_loads(tmp_path, monkeypatch):
    e = engine(tmp_path)
    e.collection.modify(metadata={**e.collection.metadata, "embedder_revision": "0" * 40})

    def boom(*_, **__):
        raise AssertionError("a model was loaded")

    monkeypatch.setattr(rag_engine, "SentenceTransformer", boom)
    monkeypatch.setattr(rag_engine, "CrossEncoder", boom)
    with pytest.raises(ValueError, match="was built with"):
        RAGEngine(db_path=str(tmp_path), collection_name="docs", embedder=BGE)


def test_collection_with_distance_settings_gets_its_record(tmp_path):
    import chromadb

    chromadb.PersistentClient(path=str(tmp_path)).get_or_create_collection(
        f"docs-{BGE}", metadata={"hnsw:space": "l2"})
    assert engine(tmp_path).collection.metadata["embedder"] == BGE


def test_old_collection_without_a_record_is_accepted_and_recorded(tmp_path):
    import chromadb

    chromadb.PersistentClient(path=str(tmp_path)).get_or_create_collection("docs").add(
        ids=["1"], documents=["old"], embeddings=[[0.5, 0.5, 0.5, 0.5]])
    e = engine(tmp_path, MINILM)
    assert e.collection.count() == 1
    assert e.collection.metadata["embedder"] == MINILM


def test_unknown_model_is_refused_before_any_model_loads(tmp_path, monkeypatch):
    def boom(*_, **__):
        raise AssertionError("a model was loaded")

    monkeypatch.setattr(rag_engine, "SentenceTransformer", boom)
    monkeypatch.setattr(rag_engine, "CrossEncoder", boom)
    with pytest.raises(ValueError, match="Unknown embedder"):
        RAGEngine(db_path=str(tmp_path), embedder="no-such-model")


def test_model_names_from_the_environment(tmp_path, monkeypatch):
    assert resolve_embedder(None) == BGE
    assert resolve_embedder("  ") == BGE
    assert resolve_embedder("BAAI/bge-small-en-v1.5") == BGE
    monkeypatch.setenv("RAG_EMBEDDER", f" {MINILM} ")
    e = RAGEngine(db_path=str(tmp_path), embedding_model=FakeModel(), reranker=FakeReranker())
    assert e.embedder == MINILM


def test_each_model_is_pinned():
    for spec in EMBEDDERS.values():
        assert len(spec["revision"]) == 40


def test_reranker_gives_raw_scores(monkeypatch):
    import torch

    seen = {}
    monkeypatch.setattr(rag_engine, "CrossEncoder", lambda *a, **kw: seen.update(kw))
    rag_engine.load_reranker()
    assert isinstance(seen["activation_fn"], torch.nn.Identity)


# --- FAISS ---

def faiss_engine(tmp_path, embedder=BGE, dim=4):
    pytest.importorskip("faiss")
    return engine(tmp_path, embedder, backend="faiss", model=FakeModel(dim))


def add_and_save(e, texts):
    e._faiss_add(texts, e.embed_passages(texts).tolist(), [{"source": t} for t in texts], texts)
    e._save_faiss()


def test_faiss_has_a_folder_per_model_and_the_size_of_the_model(tmp_path):
    e = faiss_engine(tmp_path, dim=4)
    assert e.faiss_db_path == str(tmp_path / BGE)
    assert e.faiss_index.d == 4
    assert faiss_engine(tmp_path, MINILM).faiss_db_path == str(tmp_path)


def test_faiss_index_reopens_with_its_passages(tmp_path):
    add_and_save(faiss_engine(tmp_path), ["one", "two"])
    e = faiss_engine(tmp_path)
    assert e.faiss_index.ntotal == 2 and e.faiss_ids == ["one", "two"]


def test_faiss_index_with_another_vector_size_is_refused(tmp_path):
    add_and_save(faiss_engine(tmp_path, dim=4), ["one"])
    with pytest.raises(ValueError, match="vectors of size 4"):
        faiss_engine(tmp_path, dim=5)


def test_faiss_index_built_with_another_revision_is_refused(tmp_path):
    import json

    add_and_save(faiss_engine(tmp_path), ["one"])
    meta = tmp_path / BGE / "faiss_meta.json"
    data = json.loads(meta.read_text())
    data["embedder"]["embedder_revision"] = "0" * 40
    meta.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="was built with"):
        faiss_engine(tmp_path)
