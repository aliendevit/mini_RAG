# src/answerer.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from src.retriever import RetrievedChunk


_SENT_SPLIT = re.compile(r"(?<=[\.\?\!])\s+")


def split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    text = re.sub(r"\s+", " ", text)
    sents = _SENT_SPLIT.split(text)
    # filter too short fragments
    sents = [s.strip() for s in sents if len(s.strip()) >= 25]
    return sents


def l2_normalize(vecs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, eps)


@dataclass(frozen=True)
class ScoredSentence:
    score: float
    sentence: str
    source_name: str
    chunk_id: str


def extractive_answer(
    question: str,
    retrieved: List[RetrievedChunk],
    model: SentenceTransformer,
    max_sentences: int = 4,
    max_total_chars: int = 900,
) -> Tuple[str, List[ScoredSentence]]:
    """
    Extractive answer:
    - split retrieved chunks into sentences
    - embed sentences and question
    - rank sentences by cosine similarity
    - return top sentences as the answer
    """
    question = (question or "").strip()
    if not retrieved:
        return "I could not find relevant context in the indexed documents for this question.", []

    candidates: List[ScoredSentence] = []
    sent_texts: List[str] = []
    sent_meta: List[Tuple[str, str]] = []  # (source_name, chunk_id)

    for r in retrieved:
        for s in split_sentences(r.text):
            sent_texts.append(s)
            sent_meta.append((r.source_name, r.chunk_id))

    if not sent_texts:
        # fallback to context-only style
        fallback = "\n".join([f"- [{r.source_name}] {r.text[:200].strip()}..." for r in retrieved[:3]])
        return f"Top context excerpts:\n{fallback}", []

    # Embed question + sentences
    q_emb = model.encode([question], convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False).astype(
        np.float32
    )
    s_emb = model.encode(sent_texts, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False).astype(
        np.float32
    )
    q_emb = l2_normalize(q_emb)
    s_emb = l2_normalize(s_emb)

    sims = (s_emb @ q_emb[0]).astype(np.float32)
    top_n = min(max_sentences * 3, sims.shape[0])  # take a bit more then filter
    idxs = np.argsort(-sims)[:top_n].tolist()

    # Build scored list
    scored: List[ScoredSentence] = []
    for i in idxs:
        src_name, chunk_id = sent_meta[i]
        scored.append(
            ScoredSentence(score=float(sims[i]), sentence=sent_texts[i], source_name=src_name, chunk_id=chunk_id)
        )

    # Deduplicate near-identical sentences (simple)
    final: List[ScoredSentence] = []
    seen = set()
    for item in scored:
        key = item.sentence.lower()
        if key in seen:
            continue
        seen.add(key)
        final.append(item)
        if len(final) >= max_sentences:
            break

    # Compose answer
    parts: List[str] = []
    total = 0
    for s in final:
        line = f"- {s.sentence}"
        if total + len(line) > max_total_chars:
            break
        parts.append(line)
        total += len(line) + 1

    answer = "Extractive answer (from retrieved sources):\n" + "\n".join(parts)
    return answer, final
