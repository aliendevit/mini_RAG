# tests/test_api.py
from pathlib import Path

from fastapi.testclient import TestClient

from src.api_app import create_app
from src.indexer import build_index


def test_health() -> None:
    app = create_app(index_dir="data_index")  # ok even if empty; /health should work
    client = TestClient(app)

    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ask_schema(tmp_path: Path) -> None:
    docs_dir = tmp_path / "data_docs"
    index_dir = tmp_path / "data_index"
    docs_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    (docs_dir / "a.txt").write_text(
        "Embeddings are vectors. Normalizing makes cosine similarity equal dot product.", encoding="utf-8"
    )
    (docs_dir / "b.txt").write_text(
        "Chunking splits documents. Overlap prevents losing context at chunk boundaries.", encoding="utf-8"
    )

    build_index(docs_dir=docs_dir, index_dir=index_dir, log_level="ERROR")

    app = create_app(index_dir=str(index_dir))
    client = TestClient(app)

    payload = {"question": "Why normalize embeddings?", "top_k": 3}
    r = client.post("/ask", json=payload)
    assert r.status_code == 200

    data = r.json()
    assert "Extractive answer" in data["answer"]
    assert "question" in data and data["question"]
    assert "answer" in data and isinstance(data["answer"], str) and data["answer"].strip()
    assert "sources" in data and isinstance(data["sources"], list)

    if data["sources"]:
        assert data["sources"][0]["rank"] == 1
        assert "score" in data["sources"][0]
        assert "text_preview" in data["sources"][0]
