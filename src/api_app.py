# src/api_app.py
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.answerer import extractive_answer
from src.retriever import Retriever, RetrievedChunk, context_only_answer


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=20)


class SourceItem(BaseModel):
    rank: int
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


def create_app(index_dir: Optional[str] = None) -> FastAPI:
    app = FastAPI(title="Mini-RAG API", version="0.2.0")

    rag_index_dir = index_dir or os.getenv("RAG_INDEX_DIR", "data_index")
    rag_index_path = Path(rag_index_dir).resolve()

    retriever_holder = {"retriever": None}

    def get_retriever() -> Retriever:
        if retriever_holder["retriever"] is None:
            try:
                retriever_holder["retriever"] = Retriever(rag_index_path)
            except FileNotFoundError as e:
                # Make it a clean client error instead of 500
                raise HTTPException(status_code=400, detail=str(e))
        return retriever_holder["retriever"]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/ask", response_model=AskResponse)
    def ask(payload: AskRequest) -> AskResponse:
        retriever = get_retriever()

        retrieved = retriever.retrieve(payload.question, top_k=payload.top_k)
        answer, _ = extractive_answer(payload.question, retrieved, retriever.model)
        sources: List[SourceItem] = []
        for idx, r in enumerate(retrieved, start=1):
            preview = r.text.strip().replace("\n", " ")
            if len(preview) > 240:
                preview = preview[:240].rstrip() + "..."
            sources.append(
                SourceItem(
                    rank=idx,
                    chunk_id=r.chunk_id,
                    source_name=r.source_name,
                    source_path=r.source_path,
                    score=float(r.score),
                    start_char=r.start_char,
                    end_char=r.end_char,
                    text_preview=preview,
                )
            )

        return AskResponse(question=payload.question, answer=answer, sources=sources)

    return app


app = create_app()
