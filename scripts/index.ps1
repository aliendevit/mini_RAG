param(
  [string]$DocsDir = "data_docs",
  [string]$IndexDir = "data_index"
)

$ErrorActionPreference = "Stop"

Write-Host "Indexing docs from: $DocsDir"
Write-Host "Index output dir:  $IndexDir"

python -m src.indexer --docs_dir ".\$DocsDir" --index_dir ".\$IndexDir"
