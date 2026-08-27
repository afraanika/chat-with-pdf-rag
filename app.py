"""
FastAPI web app around the chat_with_pdf RAG pipeline.

Serves a small upload-and-chat UI (static/) and a JSON API. Each user has
an account (signup/login via a signed session cookie) and their own set of
uploaded documents - chunks and FAISS indices are persisted to disk
(storage.py) and loaded through a small LRU cache (doc_cache.py) rather
than kept in one global slot, so multiple people can use the app
concurrently without stepping on each other's documents.
"""

import io

from fastapi import Depends, FastAPI, File, HTTPException, Response, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

import doc_cache
import storage
from auth import (
    SESSION_COOKIE_NAME,
    SESSION_MAX_AGE,
    COOKIE_SECURE,
    create_session_token,
    get_current_user,
    hash_password,
    verify_password,
)
from chat_with_pdf import (
    OLLAMA_HOST,
    OLLAMA_MODEL,
    build_document,
    build_index,
    build_prompt,
    chunk_text,
    embed_texts,
    extract_page_content,
    generate_answer,
    load_embedding_model,
    remove_repeated_lines,
    retrieve,
)
from db import Document, User, get_db_session, init_db

app = FastAPI(title="Chat with a PDF")

embedding_model = None


@app.on_event("startup")
def on_startup() -> None:
    global embedding_model
    init_db()
    embedding_model = load_embedding_model()


# ---- Auth ----------------------------------------------------------------


class SignupRequest(BaseModel):
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    email: str


def _set_session_cookie(response: Response, user_id: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=create_session_token(user_id),
        max_age=SESSION_MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=COOKIE_SECURE,
    )


@app.post("/api/signup", response_model=UserOut)
def signup(req: SignupRequest, response: Response, db: Session = Depends(get_db_session)) -> UserOut:
    if len(req.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")

    existing = db.query(User).filter(User.email == req.email).first()
    if existing:
        raise HTTPException(status_code=409, detail="An account with this email already exists.")

    user = User(email=req.email, hashed_password=hash_password(req.password))
    db.add(user)
    db.commit()
    db.refresh(user)

    _set_session_cookie(response, user.id)
    return UserOut(email=user.email)


@app.post("/api/login", response_model=UserOut)
def login(req: LoginRequest, response: Response, db: Session = Depends(get_db_session)) -> UserOut:
    user = db.query(User).filter(User.email == req.email).first()
    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password.")

    _set_session_cookie(response, user.id)
    return UserOut(email=user.email)


@app.post("/api/logout")
def logout(response: Response) -> dict:
    response.delete_cookie(SESSION_COOKIE_NAME)
    return {"ok": True}


@app.get("/api/me", response_model=UserOut)
def me(current_user: User = Depends(get_current_user)) -> UserOut:
    return UserOut(email=current_user.email)


# ---- Documents -------------------------------------------------------------


class DocumentOut(BaseModel):
    id: str
    filename: str
    num_pages: int
    num_chunks: int


@app.post("/api/upload", response_model=DocumentOut)
def upload_pdf(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session),
) -> DocumentOut:
    if not file.filename.lower().endswith(".pdf") and file.content_type not in (
        "application/pdf",
        "application/octet-stream",
    ):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    raw_bytes = file.file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    MAX_UPLOAD_BYTES = 20 * 1024 * 1024
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="PDF is larger than the 20MB limit.")

    try:
        raw_pages = extract_page_content(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read PDF: {exc}") from exc

    cleaned_pages = remove_repeated_lines(raw_pages)
    document, page_ranges = build_document(cleaned_pages)
    chunks = chunk_text(document, page_ranges)

    if not chunks:
        raise HTTPException(status_code=400, detail="No extractable text found in this PDF.")

    embeddings = embed_texts(embedding_model, [chunk.text for chunk in chunks])
    index = build_index(embeddings)

    doc = Document(
        user_id=current_user.id,
        filename=file.filename,
        num_pages=len(cleaned_pages),
        num_chunks=len(chunks),
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    storage.save_document(current_user.id, doc.id, chunks, index)

    return DocumentOut(id=doc.id, filename=doc.filename, num_pages=doc.num_pages, num_chunks=doc.num_chunks)


@app.get("/api/documents", response_model=list[DocumentOut])
def list_documents(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session),
) -> list[DocumentOut]:
    docs = (
        db.query(Document)
        .filter(Document.user_id == current_user.id)
        .order_by(Document.created_at.desc())
        .all()
    )
    return [
        DocumentOut(id=d.id, filename=d.filename, num_pages=d.num_pages, num_chunks=d.num_chunks)
        for d in docs
    ]


def _get_owned_document(doc_id: str, current_user: User, db: Session) -> Document:
    doc = db.get(Document, doc_id)
    if not doc or doc.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Document not found.")
    return doc


@app.delete("/api/documents/{doc_id}")
def delete_document(
    doc_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session),
) -> dict:
    doc = _get_owned_document(doc_id, current_user, db)
    storage.delete_document(current_user.id, doc.id)
    doc_cache.evict(current_user.id, doc.id)
    db.delete(doc)
    db.commit()
    return {"ok": True}


# ---- Chat -------------------------------------------------------------


class ChatRequest(BaseModel):
    document_id: str
    question: str
    top_k: int = 3


class SourceOut(BaseModel):
    index: int
    pages: list[int]
    score: float
    text: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceOut]


@app.post("/api/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session),
) -> ChatResponse:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    doc = _get_owned_document(req.document_id, current_user, db)
    chunks, index = doc_cache.get(current_user.id, doc.id)

    results = retrieve(req.question, chunks, index, embedding_model, top_k=req.top_k)
    prompt = build_prompt(req.question, results)

    try:
        answer = generate_answer(prompt, model=OLLAMA_MODEL, host=OLLAMA_HOST)
    except ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    sources = [
        SourceOut(index=i, pages=chunk.pages, score=score, text=chunk.text)
        for i, (chunk, score) in enumerate(results, start=1)
    ]
    return ChatResponse(answer=answer, sources=sources)


app.mount("/", StaticFiles(directory="static", html=True), name="static")
