# src/api_app.py
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.retriever import Retriever, RetrievedChunk, context_only_answer


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")
    top_k: int = Field(5, ge=1, le=20, description="How many chunks to retrieve")


class SourceItem(BaseModel):
    chunk_id: str
    source_name: str
    source_path: str
    score: float
    start_char: int
    end_char: int
    text_preview: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceItem]


def _preview(text: str, limit: int = 240) -> str:
    text = (text or "").strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def create_app(index_dir: Optional[str] = None) -> FastAPI:
    app = FastAPI(title="Mini-RAG API", version="0.1.0")

    rag_index_dir = index_dir or os.environ.get("RAG_INDEX_DIR", "data_index")
    rag_index_path = Path(rag_index_dir)

    retriever_holder = {"retriever": None}  # simple mutable holder

    @app.get("/health")
    def health():
        return {"status": "ok"}

    def get_retriever() -> Retriever:
        if retriever_holder["retriever"] is None:
            if not rag_index_path.exists():
                raise HTTPException(
                    status_code=400,
                    detail=f"Index directory not found: {rag_index_path}. Run indexer first.",
                )
            retriever_holder["retriever"] = Retriever(rag_index_path)
        return retriever_holder["retriever"]

    @app.post("/ask", response_model=AskResponse)
    def ask(payload: AskRequest):
        retriever = get_retriever()

        retrieved: List[RetrievedChunk] = retriever.retrieve(payload.question, top_k=payload.top_k)
        answer = context_only_answer(payload.question, retrieved)

        sources: List[SourceItem] = [
            SourceItem(
                chunk_id=r.chunk_id,
                source_name=r.source_name,
                source_path=r.source_path,
                score=r.score,
                start_char=r.start_char,
                end_char=r.end_char,
                text_preview=_preview(r.text),
            )
            for r in retrieved
        ]

        if not answer.strip():
            # Should never happen, but keep API contract safe.
            answer = "No answer could be generated from the retrieved context."

        return AskResponse(question=payload.question, answer=answer, sources=sources)

    return app


# Uvicorn entrypoint:
app = create_app()
