# Chunking
Chunking splits large documents into smaller pieces (chunks) for retrieval.

Key parameters:
- chunk_size: how big each chunk is
- overlap: repeated text between chunks to avoid losing context at boundaries

Trade-offs:
- Smaller chunks: better precision, may lose context
- Larger chunks: more context, may reduce precision
