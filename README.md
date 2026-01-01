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

