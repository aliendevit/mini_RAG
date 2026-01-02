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

## Projektbeschreibung (Deutsch)

Dieses Projekt ist ein kompakter **Mini-RAG-Service**, der lokale Textdokumente indexiert und Fragen mit nachvollziehbaren Quellen beantwortet.  
Die Pipeline umfasst **Chunking**, **Embeddings** (Sentence-Transformers), **Top-K Retrieval** via Vektorsuche sowie eine robuste Answering-Schicht:

- Deterministisches **extractive Answering** als Baseline zur Minimierung von Halluzinationen  
- Optionale lokale **HuggingFace-LLM** (z. B. Gemma) zur Formulierung  
- **Grounding-Guardrail**: Antworten werden nur akzeptiert, wenn sie Quellen (S1…Sk) referenzieren, sonst Fallback auf extractive Answering  
- **FastAPI**: `GET /health`, `POST /ask` → `{question, answer, sources[]}`  
- **Tests & Evaluation**: pytest + Retrieval-Evaluation (**Hit@K**, **MRR@K**) auf einem JSONL-Fragenset  

Tech-Stack: Python 3.12, FastAPI, sentence-transformers, NumPy/Scikit-learn (FAISS optional), pytest, GitHub Actions.
