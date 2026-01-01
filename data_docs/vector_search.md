# Vector Search
Vector search retrieves top-k vectors most similar to a query vector.
Steps:
- Embed query text
- Compute similarity between query embedding and document chunk embeddings
- Return top-k chunk indices + scores

FAISS can speed this up. If FAISS isn't available, cosine similarity can be computed via NumPy.
