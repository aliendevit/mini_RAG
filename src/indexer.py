# src/indexer.py
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from src.text_utils import chunk_text_smart, normalize_text

SUPPORTED_EXTS = {".txt", ".md"}


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source_path: str
    source_name: str
    start_char: int
    end_char: int
    text: str


def l2_normalize(vecs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, eps)


def iter_doc_paths(docs_dir: Path) -> Iterable[Path]:
    for p in docs_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            yield p


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def build_chunks_for_doc(doc_path: Path, max_chars: int, overlap: int) -> List[Chunk]:
    raw = read_text_file(doc_path)
    norm = normalize_text(raw)

    triples: List[Tuple[int, int, str]] = chunk_text_smart(norm, max_chars=max_chars, overlap=overlap)

    chunks: List[Chunk] = []
    for i, (s, e, txt) in enumerate(triples):
        chunk_id = f"{doc_path.name}::chunk_{i:04d}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                source_path=str(doc_path.resolve()),
                source_name=doc_path.name,
                start_char=s,
                end_char=e,
                text=txt,
            )
        )
    return chunks


def try_build_faiss_index(embeddings: np.ndarray, index_path: Path) -> str:
    """
    Try to build a FAISS IndexFlatIP (inner product). Works with normalized embeddings.
    Returns a status string for meta.
    """
    try:
        import faiss  # type: ignore

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype(np.float32))
        faiss.write_index(index, str(index_path))
        return "built"
    except Exception as e:
        return f"unavailable: {type(e).__name__}: {e}"


def build_index(
    docs_dir: Path,
    index_dir: Path,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_chars: int = 900,
    overlap: int = 150,
    batch_size: int = 32,
    log_level: str = "INFO",
) -> None:
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))
    log = logging.getLogger("indexer")

    t0 = time.time()
    docs_dir = docs_dir.resolve()
    index_dir = index_dir.resolve()
    index_dir.mkdir(parents=True, exist_ok=True)

    doc_paths = sorted(list(iter_doc_paths(docs_dir)))
    if not doc_paths:
        raise FileNotFoundError(f"No supported documents found in {docs_dir} (supported: {sorted(SUPPORTED_EXTS)})")

    log.info("Docs dir: %s", docs_dir)
    log.info("Index dir: %s", index_dir)
    log.info("Found %d document(s)", len(doc_paths))

    all_chunks: List[Chunk] = []
    for p in doc_paths:
        ch = build_chunks_for_doc(p, max_chars=max_chars, overlap=overlap)
        log.info("Chunked %-30s -> %d chunk(s)", p.name, len(ch))
        all_chunks.extend(ch)

    if not all_chunks:
        raise RuntimeError("No chunks generated. Check your documents content.")

    texts = [c.text for c in all_chunks]

    log.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name)

    log.info("Embedding %d chunk(s)...", len(texts))
    emb = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    ).astype(np.float32)

    emb = l2_normalize(emb).astype(np.float32)

    # Save artifacts
    chunks_path = index_dir / "chunks.jsonl"
    emb_path = index_dir / "embeddings.npy"
    meta_path = index_dir / "index_meta.json"
    faiss_path = index_dir / "index.faiss"

    with chunks_path.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    np.save(emb_path, emb)

    faiss_status = try_build_faiss_index(emb, faiss_path)

    meta = {
        "model_name": model_name,
        "supported_exts": sorted(SUPPORTED_EXTS),
        "num_docs": len(doc_paths),
        "num_chunks": len(all_chunks),
        "embedding_dim": int(emb.shape[1]),
        "chunking": {"max_chars": max_chars, "overlap": overlap, "mode": "smart-boundaries"},
        "faiss": {"status": faiss_status, "path": str(faiss_path)},
        "generated_at_unix": int(time.time()),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    log.info("Wrote: %s", chunks_path)
    log.info("Wrote: %s", emb_path)
    log.info("Wrote: %s", meta_path)
    log.info("FAISS: %s", faiss_status)
    log.info("Done in %.2fs", time.time() - t0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Mini-RAG index from local documents.")
    parser.add_argument("--docs_dir", type=str, default="data_docs")
    parser.add_argument("--index_dir", type=str, default="data_index")
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--max_chars", type=int, default=900)
    parser.add_argument("--overlap", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--log_level", type=str, default="INFO")
    args = parser.parse_args()

    build_index(
        docs_dir=Path(args.docs_dir),
        index_dir=Path(args.index_dir),
        model_name=args.model_name,
        max_chars=args.max_chars,
        overlap=args.overlap,
        batch_size=args.batch_size,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
