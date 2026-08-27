# Chat with a PDF — RAG from Scratch

A Retrieval-Augmented Generation (RAG) system that lets you ask 
questions about a PDF document and get grounded, source-cited answers.

This isn't a wrapper around a RAG library — it's built stage-by-stage 
(text extraction → chunking → embeddings → vector retrieval → prompt 
construction → generation) to understand each piece of the pipeline 
individually, including the tradeoffs and failure modes at each step. On 
top of that pipeline sits a multi-user web app with its own accounts, 
per-user document storage, and a Docker-based deployment.

## What it does
- Loads a PDF and extracts raw text
- Splits the text into chunks (with configurable size/overlap)
- Embeds each chunk and stores it in a simple vector index
- Given a question, retrieves the most relevant chunks
- Constructs a grounded prompt and generates an answer using an LLM
- Shows which source chunks were used for each answer
- Supports multiple users, each with their own login and their own set 
  of uploaded documents, switchable from the UI

## Why
Most "chat with your PDF" tutorials skip straight to a working demo. 
This project is built the other way around — one stage at a time, with 
an explanation of what each stage does, why it's needed, and what design 
tradeoff was made (e.g. fixed-size vs. semantic chunking, why an index 
beats brute-force similarity at scale, why retrieval beats fine-tuning 
for grounding).

## Stack
Python · FastAPI (web app + auth) · SQLAlchemy/SQLite (user & document metadata) · sentence-transformers (all-MiniLM-L6-v2) for embeddings · Ollama (local LLM) for generation · numpy / FAISS for vector search · Docker + Cloudflare Tunnel for deployment

## Status
All seven pipeline stages are built. On top of them: a FastAPI web app 
with signup/login (bcrypt-hashed passwords, signed session cookies), 
per-user document storage on disk, a multi-document switcher in the UI, 
and a Docker Compose setup for running the app in a container against a 
host-run Ollama instance.

## Running the web app locally (no Docker)
1. Make sure [Ollama](https://ollama.com) is running locally and the model in `.env`/`OLLAMA_MODEL` is pulled (`ollama pull llama3`).
2. Install dependencies: `pip install -r requirements.txt`
3. Start the server: `uvicorn app:app --reload`
4. Open http://localhost:8000, sign up, upload a PDF, and start asking questions.

Each user only sees their own uploaded documents. Uploading a new PDF 
adds it to that user's document list rather than replacing anything — 
switch between documents from the UI.

## Running with Docker
The app runs in a container; Ollama runs natively on the host (Docker 
Desktop on macOS doesn't pass GPU/Metal acceleration through to Linux 
containers, so containerizing Ollama too makes generation dramatically 
slower).

1. Make sure Ollama is installed and running on the host with the model pulled (`ollama pull llama3`).
2. `docker compose up -d --build`
3. Open http://localhost:8000

Document storage and the SQLite database persist in the `app_data` 
Docker volume across restarts. Set `SESSION_SECRET` in your environment 
before exposing this beyond local testing — the compose file falls back 
to an insecure default otherwise.

## Exposing it publicly
For a quick, free way to share a running instance without owning a 
domain, use a [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps):

```
cloudflared tunnel --url http://localhost:8000
```

This prints a temporary `trycloudflare.com` URL that proxies to your 
local app. It's meant for quick testing, not a stable production URL — a 
named tunnel with a persistent URL requires owning a domain.
