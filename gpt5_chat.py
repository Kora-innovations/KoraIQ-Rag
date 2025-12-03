"""
chat_agent.py — FastAPI + CLI for a simple internal GPT-5 chat with Postgres history

Quickstart
----------
# 1) Install deps
#    pip install fastapi uvicorn "SQLAlchemy>=2" "psycopg[binary]>=3" pydantic openai httpx python-dotenv

# 2) Set env (examples)
#    export OPENAI_API_KEY=sk-...
#    export OPENAI_MODEL=gpt-5.1
#    export DATABASE_URL=postgresql+psycopg://user:pass@localhost:5432/chatdb
#    # optional
#    export SYSTEM_PROMPT="You are a helpful internal assistant."
#    export MAX_HISTORY_MESSAGES=20

# 3a) Run API server
#    python chat_agent.py serve --port 8080
#    -> Open http://localhost:8080/docs

# 3b) CLI chat (uses same Postgres history)
#    python chat_agent.py chat --session local-dev
"""

import os
import argparse
import uuid
from datetime import datetime, timedelta
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager

from pydantic import BaseModel

from sqlalchemy import (
    create_engine, String, Text, DateTime, ForeignKey, func
)
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker, Session
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import select, delete


# OpenAI SDK (v1+)
from openai import OpenAI

# ---------- Settings ----------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5")
SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    "You are a helpful, concise internal assistant. If a question is unclear, ask for clarification."
)
DATABASE_URL = os.environ.get("POSTGRES_URL")
MAX_HISTORY_MESSAGES = int(os.environ.get("MAX_HISTORY_MESSAGES", "20"))
RETENTION_DAYS = int(os.environ.get("RETENTION_DAYS", "365"))

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

client = OpenAI(api_key=OPENAI_API_KEY)

# ---------- DB setup ----------
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)

class Base(DeclarativeBase):
    pass

class Conversation(Base):
    __tablename__ = "conversations"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    title: Mapped[Optional[str]] = mapped_column(String(255), default="New chat")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    messages: Mapped[List["Message"]] = relationship(
        "Message",
        back_populates="conversation",
        order_by="Message.created_at",
        cascade="all, delete-orphan",
    )

class Message(Base):
    __tablename__ = "messages"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(20))  # "system" | "user" | "assistant"
    content: Mapped[str] = mapped_column(Text)
    model: Mapped[Optional[str]] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    conversation: Mapped[Conversation] = relationship("Conversation", back_populates="messages")

def init_db() -> None:
    Base.metadata.create_all(bind=engine)

# ---------- History helpers ----------
def get_or_create_conversation(db: Session, session_id: str, title: Optional[str] = None) -> Conversation:
    conv = db.query(Conversation).filter(Conversation.session_id == session_id).one_or_none()
    if not conv:
        conv = Conversation(session_id=session_id, title=title or "New chat")
        db.add(conv)
        db.flush()
    return conv

def add_message(db: Session, conv: Conversation, role: str, content: str, model: Optional[str] = None) -> Message:
    m = Message(conversation_id=conv.id, role=role, content=content, model=model)
    db.add(m)
    db.flush()
    return m

def fetch_history(db: Session, conv: Conversation, limit: int = MAX_HISTORY_MESSAGES) -> List[Message]:
    # last N messages ordered by created_at asc
    q = (
        db.query(Message)
        .filter(Message.conversation_id == conv.id)
        .order_by(Message.created_at.desc())
        .limit(limit)
    )
    rows = list(reversed(q.all()))
    return rows

def prune_old_messages(db: Session, conv: Conversation, limit: int = MAX_HISTORY_MESSAGES) -> None:
    subq = (
        select(Message.id)
        .where(Message.conversation_id == conv.id)
        .order_by(Message.created_at.desc())
        .offset(limit)            # ids beyond the newest `limit`
    )
    db.execute(delete(Message).where(Message.id.in_(subq)))

# def prune_old_messages(db: Session, conv: Conversation, limit: int = MAX_HISTORY_MESSAGES) -> None:
#     # Keep only latest `limit` messages (plus any system messages if you add them)
#     subq = (
#         db.query(Message.id)
#         .filter(Message.conversation_id == conv.id)
#         .order_by(Message.created_at.desc())
#         .offset(limit)
#         .subquery()
#     )
#     # Delete everything beyond the tail
#     db.query(Message).filter(Message.id.in_(subq)).delete(synchronize_session=False)

def enforce_retention(db: Session) -> None:
    cutoff = datetime.utcnow() - timedelta(days=RETENTION_DAYS)
    db.execute(delete(Message).where(Message.created_at < cutoff))

# def enforce_retention(db: Session) -> None:
#     cutoff = datetime.utcnow() - timedelta(days=RETENTION_DAYS)
#     db.query(Message).filter(Message.created_at < cutoff).delete(synchronize_session=False)

# ---------- OpenAI chat ----------
def call_openai(messages: List[dict]) -> str:
    """
    messages: list of {"role": "user"|"assistant"|"system", "content": str}
    Returns assistant text (non-streaming for CLI simplicity).
    """
    # Modern Responses API
    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=messages
    )
    # Extract text
    # When using Responses API with multi-turn, the assistant text is in output_text
    # SDK provides a convenience:
    text = resp.output_text  # type: ignore[attr-defined]
    return text or ""

# ---------- FastAPI ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()           # runs once at startup
    yield               

app = FastAPI(title="Internal Chat (no RAG)", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

class CreateConversationIn(BaseModel):
    session_id: str
    title: Optional[str] = None

class ConversationOut(BaseModel):
    id: uuid.UUID
    session_id: str
    title: Optional[str]
    created_at: datetime

class MessageOut(BaseModel):
    id: uuid.UUID
    role: str
    content: str
    created_at: datetime
    model: Optional[str] = None

class ChatIn(BaseModel):
    session_id: str
    message: str

# @app.on_event("startup")
# def _startup():
#     init_db()

@app.get("/healthz")
def healthz():
    with SessionLocal() as db:
        db.execute(select(func.now()))
    return {"status": "ok"}

# @app.get("/healthz")
# def healthz():
#     with SessionLocal() as db:
#         # basic liveness + db check
#         db.execute(func.now().select())
#     return {"status": "ok"}

@app.post("/conversations", response_model=ConversationOut)
def create_conversation(payload: CreateConversationIn):
    with SessionLocal() as db:
        conv = get_or_create_conversation(db, payload.session_id, title=payload.title)
        db.commit()
        return ConversationOut(
            id=conv.id, session_id=conv.session_id, title=conv.title, created_at=conv.created_at
        )

@app.get("/conversations/{session_id}/messages", response_model=List[MessageOut])
def list_messages(session_id: str):
    with SessionLocal() as db:
        conv = db.query(Conversation).filter(Conversation.session_id == session_id).one_or_none()
        if not conv:
            raise HTTPException(404, "Conversation not found")
        rows = fetch_history(db, conv, MAX_HISTORY_MESSAGES)
        return [MessageOut(id=m.id, role=m.role, content=m.content, created_at=m.created_at, model=m.model) for m in rows]

def _sse_format(data: str) -> str:
    # minimal SSE
    return f"data: {data.replace('\n', '\\n')}\n\n"

@app.post("/chat.stream")
def chat_stream(payload: ChatIn):
    """
    Server-Sent Events stream of assistant response. Saves user/assistant messages to Postgres.
    """
    def gen():
        with SessionLocal() as db:
            enforce_retention(db)
            conv = get_or_create_conversation(db, payload.session_id)
            # Save user message
            add_message(db, conv, "user", payload.message)
            db.commit()

            # Build history
            msgs_db = fetch_history(db, conv, MAX_HISTORY_MESSAGES)
            history = [{"role": m.role, "content": m.content} for m in msgs_db]
            full = [{"role": "system", "content": SYSTEM_PROMPT}] + history

        # We'll chunk the model response in small pieces for SSE
        text = call_openai(full)
        chunk_size = 200
        sent = 0
        yield _sse_format('{"type":"start"}')
        while sent < len(text):
            part = text[sent:sent+chunk_size]
            sent += len(part)
            yield _sse_format(f'{{"type":"delta","text":{part!r}}}')
        yield _sse_format('{"type":"done"}')

        # Save assistant message
        with SessionLocal() as db:
            conv = db.query(Conversation).filter(Conversation.session_id == payload.session_id).one()
            add_message(db, conv, "assistant", text, model=OPENAI_MODEL)
            prune_old_messages(db, conv, MAX_HISTORY_MESSAGES * 4)  # allow larger window in DB, app slices last N
            db.commit()

    return StreamingResponse(gen(), media_type="text/event-stream")

# ---------- CLI ----------
# Minimal history helpers for CLI interop (familiar shape from your Supabase Postgres session)
def get_agent_history(session_id: str) -> List[dict]:
    """Return list of dict messages suitable for OpenAI: [{'role', 'content'}, ...] (without system)."""
    with SessionLocal() as db:
        conv = get_or_create_conversation(db, session_id)
        rows = fetch_history(db, conv, MAX_HISTORY_MESSAGES)
        return [{"role": m.role, "content": m.content} for m in rows]

def save_agent_history(session_id: str, role: str, content: str) -> None:
    with SessionLocal() as db:
        conv = get_or_create_conversation(db, session_id)
        add_message(db, conv, role, content, OPENAI_MODEL if role == "assistant" else None)
        prune_old_messages(db, conv, MAX_HISTORY_MESSAGES * 4)
        db.commit()

def prune_chat_history_in_place(messages: List[dict], keep: int = MAX_HISTORY_MESSAGES) -> List[dict]:
    # Keep only the last N (user/assistant); system will be added at call time
    if len(messages) <= keep:
        return messages
    return messages[-keep:]

def interactive_chat(session_id: str = "local-dev"):
    print("GPT 5 Chat — CLI mode")
    print("Type your message. Commands: /reset  /exit")
    print(f"Active session: {session_id}\n")

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user:
            continue
        if user.lower() in {"/exit", "exit", "quit"}:
            break
        if user.lower().startswith("/reset"):
            print("(Resetting chat history for current session)")
            # simplest: mint a fresh session id
            session_id = f"{session_id}-reset-{uuid.uuid4().hex[:4]}"
            print(f"New session: {session_id}")
            continue

        # Fetch + append
        history = get_agent_history(session_id)
        history.append({"role": "user", "content": user})
        history = prune_chat_history_in_place(history, MAX_HISTORY_MESSAGES)

        # Build OpenAI messages
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

        try:
            answer = call_openai(messages)
        except Exception as e:
            print(f"[err] {e}")
            continue

        # Persist both sides to Postgres
        save_agent_history(session_id, "user", user)
        save_agent_history(session_id, "assistant", answer)

        print(f"Agent: {answer}\n")

# ---------- Entrypoint ----------
if __name__ == "__main__":
    init_db()

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    p_serve = subparsers.add_parser("serve", help="Run the FastAPI server")
    p_serve.add_argument("--port", type=int, default=8080)

    p_chat = subparsers.add_parser("chat", help="Run interactive terminal chat")
    p_chat.add_argument("--session", default="local-dev", help="Session id to use for chat")

    args = parser.parse_args()

    if args.command == "serve":
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    elif args.command == "chat":
        interactive_chat(session_id=args.session)
    else:
        parser.print_help()