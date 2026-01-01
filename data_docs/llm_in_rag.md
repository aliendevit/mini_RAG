# LLMs in RAG
LLMs can be used after retrieval to produce a fluent answer using context.
However, RAG can work without an LLM:
- context-only answers can cite relevant chunks directly
- later you can add optional LLM generation locally (e.g., ollama/llama.cpp)

Important: always keep a fallback path when the LLM is unavailable.
