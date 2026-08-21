# Chat with a PDF — RAG from Scratch

A minimal Retrieval-Augmented Generation (RAG) system that lets you ask 
questions about a PDF document and get grounded, source-cited answers.

This isn't a wrapper around a RAG library — it's built stage-by-stage 
(text extraction → chunking → embeddings → vector retrieval → prompt 
construction → generation) to understand each piece of the pipeline 
individually, including the tradeoffs and failure modes at each step.

## What it does
- Loads a PDF and extracts raw text
- Splits the text into chunks (with configurable size/overlap)
- Embeds each chunk and stores it in a simple vector index
- Given a question, retrieves the most relevant chunks
- Constructs a grounded prompt and generates an answer using an LLM
- Shows which source chunks were used for each answer

## Why
Most "chat with your PDF" tutorials skip straight to a working demo. 
This project is built the other way around — one stage at a time, with 
an explanation of what each stage does, why it's needed, and what design 
tradeoff was made (e.g. fixed-size vs. semantic chunking, why an index 
beats brute-force similarity at scale, why retrieval beats fine-tuning 
for grounding).

## Stack
Python · sentence-transformers (all-MiniLM-L6-v2) for embeddings · Ollama (local LLM) for generation · numpy / FAISS for vector search

## Status
🚧 In progress — built incrementally, stage by stage.
