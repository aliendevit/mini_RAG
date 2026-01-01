# src/retriever.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    source_name: str
    source_path: str
    score: float
    text: str
    start_char: int
    end_char: int


def l2_normalize(vecs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, eps)


class Retriever:
    """
    Loads:
      - chunks.jsonl
      - embeddings.npy (normalized)
      - (optional) index.faiss
      - index_meta.json

    Retrieval:
      - If FAISS index is available, use it.
      - Else fallback to cosine via dot product on normalized vectors.
    """

    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir.resolve()
        if not self.index_dir.exists():
            raise FileNotFoundError(f"index_dir not found: {self.index_dir}")

        meta_path = self.index_dir / "index_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing index_meta.json in {self.index_dir}. Run indexer first.")

        self.meta: Dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
        self.model_name: str = self.meta["model_name"]

        self.model = SentenceTransformer(self.model_name)

        self.chunks = self._load_chunks(self.index_dir / "chunks.jsonl")
        self.embeddings = np.load(self.index_dir / "embeddings.npy").astype(np.float32)

        if len(self.chunks) != self.embeddings.shape[0]:
            raise ValueError("Mismatch: chunks count != embeddings rows")

        self.faiss = self._try_load_faiss(self.index_dir / "index.faiss")

    def _load_chunks(self, path: Path) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    def _try_load_faiss(self, path: Path) -> Optional[Any]:
        if not path.exists():
            return None
        try:
            import faiss  # type: ignore

            return faiss.read_index(str(path))
        except Exception:
            return None

    def retrieve(self, question: str, top_k: int = 5) -> List[RetrievedChunk]:
        question = (question or "").strip()
        if not question:
            return []

        q_emb = self.model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        ).astype(np.float32)
        q_emb = l2_normalize(q_emb)

        # Use FAISS inner product on normalized vectors = cosine similarity
        if self.faiss is not None:
            scores, idx = self.faiss.search(q_emb.astype(np.float32), top_k)
            idx_list = idx[0].tolist()
            score_list = scores[0].tolist()
        else:
            # NumPy fallback: cosine via dot product on normalized embeddings
            sims = (self.embeddings @ q_emb[0]).astype(np.float32)  # (N,)
            top_k = min(top_k, sims.shape[0])
            idx_list = np.argsort(-sims)[:top_k].tolist()
            score_list = sims[idx_list].tolist()

        results: List[RetrievedChunk] = []
        for i, s in zip(idx_list, score_list):
            ch = self.chunks[i]
            results.append(
                RetrievedChunk(
                    chunk_id=ch["chunk_id"],
                    source_name=ch["source_name"],
                    source_path=ch["source_path"],
                    score=float(s),
                    text=ch["text"],
                    start_char=int(ch["start_char"]),
                    end_char=int(ch["end_char"]),
                )
            )
        return results


def context_only_answer(question: str, retrieved: List[RetrievedChunk], max_chars: int = 1200) -> str:
    """
    Day-1 answering: no LLM. Provide a compact response derived from top chunks.
    Always returns a non-empty string if we have any retrieved context.
    """
    if not retrieved:
        return "I could not find relevant context in the indexed documents for this question."

    header = f"Answer (context-only) for: {question}\n\nTop context excerpts:\n"
    body_parts: List[str] = []
    total = 0

    for r in retrieved:
        excerpt = r.text.strip()
        if not excerpt:
            continue
        piece = f"- [{r.source_name}] {excerpt}"
        if total + len(piece) > max_chars:
            remaining = max(0, max_chars - total)
            if remaining > 50:
                piece = piece[:remaining].rstrip() + "..."
                body_parts.append(piece)
            break
        body_parts.append(piece)
        total += len(piece) + 1

    if not body_parts:
        return "I retrieved sources, but they contained no usable text."

    return header + "\n".join(body_parts)
