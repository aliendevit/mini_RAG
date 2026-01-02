from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.retriever import Retriever


@dataclass
class QItem:
    qid: str
    question: str
    expected_sources: List[str]
    notes: str = ""


def load_questions_jsonl(path: Path) -> List[QItem]:
    items: List[QItem] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = str(obj.get("id", f"q{line_no}"))
            question = str(obj["question"])
            expected = obj.get("expected_sources", [])
            if isinstance(expected, str):
                expected = [expected]
            expected = [str(x) for x in expected]
            notes = str(obj.get("notes", ""))
            items.append(QItem(qid=qid, question=question, expected_sources=expected, notes=notes))
    return items


def first_hit_rank(results: List[Any], expected_sources: List[str]) -> Optional[int]:
    expected = set([s.lower() for s in expected_sources])
    for i, r in enumerate(results, start=1):
        name = getattr(r, "source_name", "") or ""
        if name.lower() in expected:
            return i
    return None


def compute_metrics(hit_ranks: List[Optional[int]], k: int) -> Tuple[float, float]:
    # Hit@k
    hits = [1 for r in hit_ranks if r is not None and r <= k]
    hit_at_k = sum(hits) / max(1, len(hit_ranks))

    # MRR@k (optional): 1/rank if rank<=k else 0
    rr = [(1.0 / r) for r in hit_ranks if r is not None and r <= k]
    mrr_at_k = sum(rr) / max(1, len(hit_ranks))
    return hit_at_k, mrr_at_k


def main() -> int:
    ap = argparse.ArgumentParser(description="Simple retrieval evaluation (Hit@K, MRR@K).")
    ap.add_argument("--index_dir", type=str, default="data_index", help="Path to built index directory.")
    ap.add_argument("--questions", type=str, default="eval/questions.jsonl", help="JSONL questions file.")
    ap.add_argument("--k", type=int, default=5, help="Top-K to retrieve.")
    ap.add_argument("--out", type=str, default="", help="Optional path to save per-question results as JSON.")
    args = ap.parse_args()

    index_dir = Path(args.index_dir).resolve()
    qpath = Path(args.questions).resolve()
    k = int(args.k)

    retriever = Retriever(index_dir)
    questions = load_questions_jsonl(qpath)

    per_q: List[Dict[str, Any]] = []
    hit_ranks: List[Optional[int]] = []

    print(f"Index: {index_dir}")
    print(f"Questions: {qpath}")
    print(f"Top-K: {k}\\n")

    for q in questions:
        results = retriever.retrieve(q.question, top_k=k)

        rank = first_hit_rank(results, q.expected_sources)
        hit_ranks.append(rank)

        top_sources = [getattr(r, "source_name", "") for r in results]
        top_scores = [float(getattr(r, "score", 0.0)) for r in results]

        per_q.append(
            {
                "id": q.qid,
                "question": q.question,
                "expected_sources": q.expected_sources,
                "hit_rank": rank,
                "top_sources": top_sources,
                "top_scores": top_scores,
                "notes": q.notes,
            }
        )

        status = "HIT" if rank is not None else "MISS"
        rank_str = f"rank={rank}" if rank is not None else "rank=None"
        print(f"[{status}] {q.qid}: {rank_str} | expected={q.expected_sources}")
        print(f"      Q: {q.question}")
        print(f"      Top sources: {top_sources}\\n")

    hit_at_k, mrr_at_k = compute_metrics(hit_ranks, k)

    print("=== Summary ===")
    print(f"Questions: {len(questions)}")
    print(f"Hit@{k}:  {hit_at_k:.3f}")
    print(f"MRR@{k}:  {mrr_at_k:.3f}")

    if args.out:
        out_path = Path(args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "index_dir": str(index_dir),
                    "questions": str(qpath),
                    "k": k,
                    "hit_at_k": hit_at_k,
                    "mrr_at_k": mrr_at_k,
                    "results": per_q,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Saved report: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
