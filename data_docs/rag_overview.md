# RAG Overview
RAG (Retrieval-Augmented Generation) answers questions by retrieving relevant text chunks from a document corpus,
then using those chunks as context for the answer. This reduces hallucinations and enables citations.

Pipeline:
1) Ingest docs
2) Chunk
3) Embed chunks
4) Vector search top-k
5) Answer using retrieved context
