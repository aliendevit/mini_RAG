# Common Failure Modes
- Bad chunking: important info split incorrectly
- Embedding mismatch: wrong model used for query vs index
- Index not rebuilt after docs update
- Too large context: irrelevant text dilutes answer

Mitigations:
- tune chunk size/overlap
- store model name in meta
- add logging and tests
