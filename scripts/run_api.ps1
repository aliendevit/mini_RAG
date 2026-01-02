param(
  [string]$BindHost = "127.0.0.1",
  [int]$Port = 8000,
  [string]$IndexDir = "data_index"
)

$ErrorActionPreference = "Stop"

$env:RAG_INDEX_DIR = $IndexDir

# Optional: enable HF local LLM (leave off by default)
# $env:RAG_USE_HF_LLM="1"
# $env:HF_MODEL_DIR="C:\path\to\local\gemma\folder"
# $env:HF_DEVICE="cpu"
# $env:HF_TEMPERATURE="0"
# $env:HF_MAX_NEW_TOKENS="180"

uvicorn src.api_app:app --reload --host $BindHost --port $Port