# Mini-RAG API (Portfolio)

Mini-RAG project:
- Ingest docs from `data_docs/`
- Chunk + embed (sentence-transformers)
- Vector search (FAISS if available, fallback to NumPy cosine)
- FastAPI endpoints: `GET /health`, `POST /ask`

## Quickstart (Windows / PowerShell)

### 1) Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

Create docs (optional)

Build index

Run API

Run tests

Example request/response
### Current metrics (toy set)
- Hit@5: 1.0
- MRR@5: 0.33

```powershell
python -m src.eval_retrieval --index_dir .\data_index --questions .\eval\questions.jsonl --k 5

Description
1) Kurz (CV / 1–2 Zeilen)

Mini-RAG API (Python/FastAPI): End-to-end RAG-Pipeline für lokale Dokumente (Chunking, Embeddings, Top-K Vector Retrieval) mit deterministischem extractive Answering und optionaler lokaler HF-LLM-Stufe inkl. Grounding-Guardrail (Citations → Fallback). Tests (pytest) und Retrieval-Evaluation (Hit@K/MRR).

2) Mittel (CV / 4–6 Bullet Points)

Mini-RAG API (Portfolio-Projekt)

Entwickelte eine lokale RAG-Pipeline: Dokument-Ingestion aus data_docs/, Chunking, Embedding-Indexing und Top-K Retrieval via Vektorsuche.

Implementierte eine deterministische „extractive“ Answering-Baseline zur Minimierung von Halluzinationen.

Ergänzte optional eine lokale HuggingFace-LLM-Stufe (z. B. Gemma) und ein Grounding-Guardrail: Antworten werden nur akzeptiert, wenn sie Quellen (S1…Sk) referenzieren, sonst Fallback auf extractive Answering.

Baute eine FastAPI-Schnittstelle: GET /health, POST /ask → {question, answer, sources[]}.

Ergänzte automatisierte Tests (pytest), PowerShell-Scripts (Windows) und eine Retrieval-Evaluation (Hit@K, MRR@K) auf einem JSONL-Fragenset.

Tech-Stack: Python 3.12, FastAPI, sentence-transformers, NumPy/Scikit-learn (FAISS optional), pytest, GitHub Actions.

3) Lang (GitHub/Portfolio Beschreibung)

Dieses Projekt ist ein kompakter, praxisnaher Mini-RAG-Service, der lokale Textdokumente indexiert und Fragen mit nachvollziehbaren Quellen beantwortet. Die Pipeline umfasst Chunking, Embeddings (Sentence-Transformers), Top-K Retrieval über Vektorsuche sowie eine robuste Answering-Schicht: Als zuverlässige Baseline kommt ein deterministisches extractive Answering zum Einsatz. Optional kann eine lokale HuggingFace-LLM (z. B. Gemma) zur Formulierung genutzt werden, abgesichert durch ein Grounding-Guardrail (Citations-Check → Fallback). Das Projekt enthält eine FastAPI-API, pytest-Tests, Windows-freundliche Scripts sowie eine einfache Retrieval-Evaluation (Hit@K/MRR@K).