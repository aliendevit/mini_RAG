# src/indexer.py
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


SUPPORTED_EXTS = {".txt", ".md"}


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source_path: str
    source_name: str
    start_char: int
    end_char: int
    text: str


def iter_doc_paths(docs_dir: Path) -> Iterable[Path]:
    for p in docs_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            yield p


def read_text_file(path: Path) -> str:
    # Robust UTF-8 read for Windows. If a file has odd encoding, replace invalid chars.
    return path.read_text(encoding="utf-8", errors="replace")


def normalize_whitespace(text: str) -> str:
    # Keep it simple: normalize line endings and excessive spaces.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse huge whitespace runs but preserve newlines.
    lines = [ln.strip() for ln in text.split("\n")]
    # Remove repeated blank lines (2+ -> 1)
    out_lines: List[str] = []
    blank = False
    for ln in lines:
        if ln == "":
            if not blank:
                out_lines.append("")
            blank = True
        else:
            out_lines.append(ln)
            blank = False
    return "\n".join(out_lines).strip()


def chunk_text(
    text: str,
    max_chars: int = 900,
    overlap: int = 150,
) -> List[tuple[int, int, str]]:
    """
    Character-window chunking with overlap.
    Simple and dependable for Day 1.
    Returns list of (start_char, end_char, chunk_text).
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= max_chars:
        raise ValueError("overlap must be < max_chars")

    text = text.strip()
    if not text:
        return []

    chunks: List[tuple[int, int, str]] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append((start, end, chunk))
        if end == n:
            break
        start = max(0, end - overlap)

    return chunks


def build_chunks_from_docs(docs_dir: Path) -> List[Chunk]:
    chunks: List[Chunk] = []
    for doc_path in sorted(iter_doc_paths(docs_dir)):
        raw = read_text_file(doc_path)
        cleaned = normalize_whitespace(raw)
        spans = chunk_text(cleaned)

        for i, (s, e, ctext) in enumerate(spans):
            chunk_id = f"{doc_path.name}::chunk_{i:04d}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    source_path=str(doc_path.resolve()),
                    source_name=doc_path.name,
                    start_char=s,
                    end_char=e,
                    text=ctext,
                )
            )
    return chunks


def l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(norms, eps)


def try_build_faiss_index(embeddings: np.ndarray) -> tuple[Optional[Any], str]:
    """
    Try FAISS IndexFlatIP (cosine via normalized vectors).
    Returns (faiss_index_or_none, status_message)
    """
    try:
        import faiss  # type: ignore

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype(np.float32))
        return index, "faiss_ok"
    except Exception as e:
        return None, f"faiss_unavailable: {type(e).__name__}: {e}"


def save_index(
    index_dir: Path,
    model_name: str,
    chunks: List[Chunk],
    embeddings_norm: np.ndarray,
    faiss_index: Optional[Any],
    faiss_status: str,
) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)

    # Save chunks metadata
    chunks_path = index_dir / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as f:
        for ch in chunks:
            rec = {
                "chunk_id": ch.chunk_id,
                "source_path": ch.source_path,
                "source_name": ch.source_name,
                "start_char": ch.start_char,
                "end_char": ch.end_char,
                "text": ch.text,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Save embeddings (normalized, float32)
    emb_path = index_dir / "embeddings.npy"
    np.save(emb_path, embeddings_norm.astype(np.float32))

    # Save FAISS index if available
    faiss_path = index_dir / "index.faiss"
    if faiss_index is not None:
        try:
            import faiss  # type: ignore

            faiss.write_index(faiss_index, str(faiss_path))
        except Exception:
            # If something goes wrong, we still have embeddings.npy fallback.
            pass

    # Save meta
    meta = {
        "model_name": model_name,
        "num_chunks": len(chunks),
        "embedding_dim": int(embeddings_norm.shape[1]) if len(chunks) > 0 else 0,
        "faiss_status": faiss_status,
        "files_supported": sorted(list(SUPPORTED_EXTS)),
    }
    (index_dir / "index_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_index(
    docs_dir: Path,
    index_dir: Path,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
) -> Dict[str, Any]:
    docs_dir = docs_dir.resolve()
    index_dir = index_dir.resolve()

    if not docs_dir.exists():
        raise FileNotFoundError(f"docs_dir not found: {docs_dir}")

    chunks = build_chunks_from_docs(docs_dir)
    if not chunks:
        raise ValueError(
            f"No supported documents found in {docs_dir}. "
            f"Supported extensions: {sorted(SUPPORTED_EXTS)}"
        )

    model = SentenceTransformer(model_name)

    texts = [c.text for c in chunks]
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32)

    emb_norm = l2_normalize(emb)

    faiss_index, faiss_status = try_build_faiss_index(emb_norm)

    save_index(
        index_dir=index_dir,
        model_name=model_name,
        chunks=chunks,
        embeddings_norm=emb_norm,
        faiss_index=faiss_index,
        faiss_status=faiss_status,
    )

    return {
        "docs_dir": str(docs_dir),
        "index_dir": str(index_dir),
        "model_name": model_name,
        "num_chunks": len(chunks),
        "faiss_status": faiss_status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Mini-RAG vector index from data_docs/.")
    parser.add_argument("--docs_dir", type=str, default="data_docs", help="Folder with .txt/.md docs")
    parser.add_argument("--index_dir", type=str, default="data_index", help="Output folder for index files")
    parser.add_argument(
        "--model_name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformers model name",
    )
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()
    result = build_index(
        docs_dir=Path(args.docs_dir),
        index_dir=Path(args.index_dir),
        model_name=args.model_name,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
