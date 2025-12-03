# Redirect to Supabase RAG Agent
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the Supabase RAG agent instead
from rag_agent_supabase import app

if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("serve")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    
    print("🚀 Starting Kora RAG Service (Supabase)...")
    uvicorn.run(app, host=args.host, port=args.port)

# Original imports (commented out to avoid conflicts)
# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_openai import OpenAIEmbeddings
# from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import Tool
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor

import os
import argparse
import contextvars
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel
from regex import sub
import uvicorn

import openai
import requests
import json
# from knowledgebase import ingest_to_mongo
from langchain.tools import tool
from pydantic import BaseModel
import tiktoken

from pymongo import MongoClient

import contextvars
from typing import Dict
from langsmith import traceable
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage
from langchain.chains import RetrievalQA
# from langchain.memory import ChatMessageHistory

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
# budget per call (history only)
MAX_HISTORY_TOKENS = int(os.getenv("MAX_HISTORY_TOKENS", "1500"))
# model to be called for responses; used for token counting
TOKEN_MODEL_FOR_COUNT = os.getenv("TOKEN_MODEL_FOR_COUNT", "gpt-4o")

# ContextVar so tools know which session is active during a request
#CURRENT_SESSION_ID = contextvars.ContextVar("session_id", default=None)

# --- Globals ---
current_session_id = contextvars.ContextVar("current_session_id")

# --- MongoDB-backed Session History ---
def get_agent_history(session_id: str) -> ChatMessageHistory:
    doc = collection.find_one({"_id": session_id, "type": "chat_history"})
    hist = ChatMessageHistory()
    if doc and "messages" in doc:
        for msg in doc["messages"]:
            if msg["type"] == "human":
                hist.add_message(HumanMessage(content=msg["content"]))
            elif msg["type"] == "ai":
                hist.add_message(AIMessage(content=msg["content"]))
            # Add other types if needed
    return hist

def save_agent_history(session_id: str, history: ChatMessageHistory):
    # Serialize messages as dicts for MongoDB
    def serialize_message(msg):
        return {
            "type": msg.type,
            "content": msg.content
        }
    # Store messages as dicts for MongoDB
    messages = [serialize_message(msg) for msg in history.messages]
    collection.update_one(
        {"_id": session_id, "type": "chat_history"},
        {"$set": {"messages": messages}},
        upsert=True
    )

# --- Mongo Connection ---
MONGODB_ATLAS_URI = os.getenv("MONGODB_ATLAS_URI")
MONGODB_DB = os.getenv("MONGODB_DB", "korahq_dev")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "ragdb")
MONGODB_INDEX_NAME = os.getenv("MONGODB_INDEX_NAME", "rag_vector_index_3")

client = MongoClient(MONGODB_ATLAS_URI)
collection = client[MONGODB_DB][MONGODB_COLLECTION]

# --- Embeddings ---
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# --- VectorStore ---
vectorstore = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embeddings,
    index_name=MONGODB_INDEX_NAME,
    text_key="text",
    embedding_key="embedding"
)
# PDF_PATHS = ["general_faqs.pdf", "general_faqs_2.pdf"]
# XLSX_PATHS = ["axa_may_2025_provider_list.xlsx", "customised_plan_and_benefits_2025.xlsx"]
# vectorstore = ingest_to_mongo(PDF_PATHS, XLSX_PATHS)


# --- History-Aware RAG Chain ---
from langchain.chains import ConversationalRetrievalChain

@traceable
def setup_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    prompt = ChatPromptTemplate.from_template("""
        You are Kora IQ, a health insurance assistant for Nigerian employees.
        
        Localization: "The Island" = Lagos Island (Lekki, VI, Ikoyi, Ajah, Oniru). "The Mainland" = Yaba, Ikeja, Surulere, Maryland, Gbagada.
        
        Answer based on the context provided. Be concise and accurate.
        
        Context: {context}
        Question: {question}
    """)
    llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY, max_tokens=1000)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain
qa_chain = setup_qa_chain(vectorstore)
# print(qa_chain.input_keys)


# --- Tools ---
search = TavilySearch(api_key=TAVILY_API_KEY)
def search_web(query: str) -> str:
    prefix = ("Nigerian context. When users say 'the Island', interpret as "
              "Lekki, VI, Ikoyi, Ajah, Oniru, etc. ")
    return search.invoke(prefix + query)

# class InternalDocsInput(BaseModel):
#     query: str

def rag_tool_func(query: str) -> str:
    sid = current_session_id.get(None)
    history = get_agent_history(sid)
    result = qa_chain.invoke({"query": query})
    return result["result"] if "result" in result else str(result)

rag_tool = Tool(
    name="search_internal_docs",
    func=rag_tool_func,
    description="Use for questions answerable with internal company docs (insurance coverage, provider lists, internal FAQs, etc)."
    # args_schema=InternalDocsInput
)

web_tool = Tool(
    name="search_web",
    func=search_web,
    description="Search the public web for up-to-date info: hospital reviews, rankings, addresses, recent news."
)

# --- Agent ---
nigerian_system_prompt = """
You are Kora IQ, a helpful, accurate, and professional QnA assistant supporting employees of a Nigerian fintech company with health insurance, HR, and provider-related questions.

IMPORTANT: You have access to two tools:
1. search_internal_docs - Use this for questions about health insurance plans, coverage, benefits, provider lists, and internal company information
2. search_web - Use this for finding current information about hospitals, clinics, reviews, and public information

ALWAYS use the appropriate tool to answer questions. Do not give generic responses.

Localization rules:
- 'The Island' means the Lagos Island axis (Lekki, VI, Ikoyi, Ajah, Oniru).
- 'The Mainland' means places like Yaba, Ikeja, Surulere, Maryland.
- Default all context to Nigerian usage and culture.
- When users ask for hospital or clinic recommendations, interpret requests in the Nigerian context, using recent data where possible.
- Never reveal private information about another employee.
- Always explain the reasoning for your recommendations.
- If you don't have enough information, ask clarifying questions using Nigerian examples.

For health insurance questions, use the search_internal_docs tool to find specific information about plans, benefits, and providers.
For hospital/clinic recommendations or current information, use the search_web tool.
"""

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(nigerian_system_prompt),
    MessagesPlaceholder("chat_history"),
    HumanMessagePromptTemplate.from_template("{input}"),
    MessagesPlaceholder("agent_scratchpad")
])
llm = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=1000)
agent = create_openai_functions_agent(
    llm=llm,
    tools=[rag_tool, web_tool],
    prompt=prompt
)
agent_executor = AgentExecutor(
    agent=agent,
    tools=[rag_tool, web_tool],
    verbose=True,
    handle_parsing_errors=True
)

# --- Ingestion ---
# def ingest_files(files: List[str]):
#     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#     all_docs: List[Document] = []

#     for file in files:
#         if file.endswith(".pdf"):
#             loader = PyPDFLoader(file)
#         elif file.endswith(".xlsx"):
#             loader = UnstructuredExcelLoader(file)
#         else:
#             raise ValueError(f"Unsupported file type: {file}")
#         docs = loader.load()
#         splits = splitter.split_documents(docs)
#         all_docs.extend(splits)

#     vectorstore.add_documents(all_docs)
#     print(f"Ingested {len(all_docs)} chunks into MongoDB Atlas")

# -----------------------------
# Token budgeting (history truncation)
# -----------------------------
def _encoding_for_model(model_name: str):
    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        # Fallback to a sensible default for GPT-4/3.5 era
        return tiktoken.get_encoding("cl100k_base")

# --- Token Counting ---
def _count_message_tokens(messages: List[BaseMessage], model_name: str) -> int:
    """
    Approx token count for a list of LangChain messages using tiktoken.
    This is an estimate, but good enough to enforce a budget.
    """
    enc = _encoding_for_model(model_name)
    total = 0
    # Simple heuristic: count tokens in content (role overhead is minor vs content)
    for m in messages:
        content = m.content if isinstance(m.content, str) else str(m.content)
        total += len(enc.encode(content))
    return total

def prune_chat_history_in_place(
    history: ChatMessageHistory,
    max_tokens: int = MAX_HISTORY_TOKENS,
    model_name: str = TOKEN_MODEL_FOR_COUNT,
) -> None:
    """
    Truncate chat history IN-PLACE to fit under max_tokens by keeping the most recent turns.
    This is pure truncation (no summarization), as requested.
    """
    msgs = history.messages
    if not msgs:
        return
    current_tokens = _count_message_tokens(msgs, model_name)
    if current_tokens <= max_tokens:
        return
    # Walk backward, keep recent messages until we hit the budget
    kept: List[BaseMessage] = []
    running = 0
    enc = _encoding_for_model(model_name)
    # Always consider preserving the most recent turns first
    for m in reversed(msgs):
        content = m.content if isinstance(m.content, str) else str(m.content)
        t = len(enc.encode(content))
        if running + t > max_tokens:
            break
        kept.append(m)
        running += t

    kept.reverse()

    # Optional: you could prepend a tiny “(previous history truncated)” system marker
    # kept = [SystemMessage(content="(Prior conversation truncated for length)")] + kept

    history.messages = kept

# --- FastAPI ---
app = FastAPI()
class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    sid = req.session_id
    current_session_id.set(sid)
    history = get_agent_history(sid)
    # Token-budgeting the chat history before invoking the agent
    prune_chat_history_in_place(history)
    result = agent_executor.invoke({
        "input": req.message,
        "chat_history": history.messages
    })
    history.add_user_message(req.message)
    history.add_ai_message(result["output"])
    save_agent_history(sid, history)
    return {"response": result["output"]}

# -----------------------------
# Interactive CLI Chat
# -----------------------------

def interactive_chat(session_id: str = "local-dev"):
    """
    Simple terminal chat loop wired to the same agent + per-session history.
    Commands:
      /reset          -> clears current session history
      /switch <id>    -> switch to another session id
      /exit           -> quit
    """
    print("HMO RAG Agent (MongoDB Atlas) — CLI mode")
    print("Type your message. Commands: /reset  /switch <id>  /exit")
    print(f"Active session: {session_id}\n")

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not user:
            continue

        # Commands
        if user.lower() in {"/exit", "exit", "quit"}:
            print("Bye!")
            break
        if user.lower() == "/reset":
            # Clear history in MongoDB
            collection.update_one({"_id": session_id, "type": "chat_history"}, {"$set": {"messages": []}}, upsert=True)
            print(f"[ok] history cleared for session '{session_id}'")
            continue
        if user.lower().startswith("/switch "):
            parts = user.split(maxsplit=1)
            if len(parts) == 2 and parts[1].strip():
                session_id = parts[1].strip()
                print(f"[ok] switched to session '{session_id}'")
            else:
                print("[err] usage: /switch <new_session_id>")
            continue

        # Make sure we have a history object
        current_session_id.set(session_id)
        history = get_agent_history(session_id)

        # Token-budgeting the chat history before invoking the agent
        prune_chat_history_in_place(history)

        # Run the agent
        try:
            res = agent_executor.invoke({
                "input": user,
                "chat_history": history.messages
            })
            answer = res.get("output", "")
        except Exception as e:
            print(f"[err] {e}")
            continue

        # Persist to history in MongoDB
        history.add_user_message(user)
        history.add_ai_message(answer)
        save_agent_history(session_id, history)

        print(f"Agent: {answer}\n")

# --- CLI ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    # ingest_parser = subparsers.add_parser("ingest")
    # ingest_parser.add_argument("--files", nargs="+", required=True)

    serve_parser = subparsers.add_parser("serve")
    serve_parser.add_argument("--port", type=int, default=8000)

    p_chat = subparsers.add_parser("chat", help="Run interactive terminal chat")
    p_chat.add_argument("--session", default="local-dev", help="Session id to use for chat")

    args = parser.parse_args()

    # if args.command == "ingest":
    #     ingest_files(args.files)
    if args.command == "serve":
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    elif args.command == "chat":
        interactive_chat(session_id=args.session)
    else:
        parser.print_help()
