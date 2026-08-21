"""
Chat with a PDF — RAG pipeline.

This file will eventually contain the following stages, built one at a time:

# Stage 1: PDF text extraction
#   - Load a PDF from data/ and extract raw text (pypdf)

# Stage 2: Chunking
#   - Split extracted text into overlapping chunks (configurable size/overlap)

# Stage 3: Embeddings
#   - Embed each chunk using a local sentence-transformers model

# Stage 4: Vector index / retrieval
#   - Store chunk embeddings in a FAISS index
#   - Given a question, embed it and retrieve the most relevant chunks

# Stage 5: Prompt construction
#   - Build a grounded prompt from the retrieved chunks + the user's question

# Stage 6: Generation
#   - Send the prompt to a local Ollama model and return the answer,
#     along with the source chunks used
"""
