# tests/test_api.py
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from src.api_app import create_app
from src.indexer import build_index


def test_health_endpoint():
    app = create_app(index_dir="data_index")  # should exist for local runs; not required here
    client = TestClient(app)

    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ask_endpoint_returns_schema_and_non_empty_answer(tmp_path: Path):
    # Arrange: create temp docs + temp index
    docs_dir = tmp_path / "data_docs"
    index_dir = tmp_path / "data_index"
    docs_dir.mkdir(parents=True, exist_ok=True)

    (docs_dir / "a.txt").write_text(
        "This is a document about RAG. RAG retrieves top-k chunks and uses them as context.",
        encoding="utf-8",
    )
    (docs_dir / "b.txt").write_text(
        "This is another doc. Embeddings can be built using sentence-transformers.",
        encoding="utf-8",
    )

    build_index(docs_dir=docs_dir, index_dir=index_dir, model_name="sentence-transformers/all-MiniLM-L6-v2")

    app = create_app(index_dir=str(index_dir))
    client = TestClient(app)

    # Act
    r = client.post("/ask", json={"question": "How does RAG work?", "top_k": 3})

    # Assert
    assert r.status_code == 200
    data = r.json()

    assert "question" in data
    assert "answer" in data
    assert "sources" in data

    assert isinstance(data["sources"], list)
    assert len(data["answer"].strip()) > 0

    # Basic source schema check
    if data["sources"]:
        s0 = data["sources"][0]
        for key in ["chunk_id", "source_name", "source_path", "score", "start_char", "end_char", "text_preview"]:
            assert key in s0
