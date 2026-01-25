import os
import argparse
import contextvars
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field # <-- TWEAK: Added Field
from regex import sub
import uvicorn

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.chains import ConversationalRetrievalChain # <-- TWEAK: No longer needed
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.documents import Document
# from langchain.agents import Tool, AgentExecutor, create_openai_functions_agent # <-- TWEAK: Replaced with LCEL branch
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langsmith import traceable
# from langchain.chains import RetrievalQA # <-- TWEAK: No longer needed
# from langchain.tools import tool # <-- TWEAK: No longer needed
from pydantic import BaseModel
import tiktoken

# --- TWEAK START: Add new LCEL imports for the router ---
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# --- TWEAK END ---

import contextvars
from typing import Dict
# from langchain.memory import ChatMessageHistory

# =====================================================================================
# --- Pinecone Vector Store Configuration ---
# Using Pinecone for vector storage and retrieval
# =====================================================================================


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
# budget per call (history only)
MAX_HISTORY_TOKENS = int(os.getenv("MAX_HISTORY_TOKENS", "1500"))
# model to be called for responses; used for token counting
TOKEN_MODEL_FOR_COUNT = os.getenv("TOKEN_MODEL_FOR_COUNT", "gpt-4o-mini")

# ContextVar so tools know which session is active during a request
#CURRENT_SESSION_ID = contextvars.ContextVar("session_id", default=None)

# --- Globals ---
current_session_id = contextvars.ContextVar("current_session_id")

# --- In-Memory Session History Storage ---
# Store chat histories in memory keyed by session_id
_session_histories: Dict[str, ChatMessageHistory] = {}

def get_agent_history(session_id: str):
    """Return in-memory chat history for the given session (persisted in memory during runtime)."""
    if session_id not in _session_histories:
        _session_histories[session_id] = ChatMessageHistory()
    return _session_histories[session_id]

def save_agent_history(session_id: str, history):
    """Save chat history for the session (already stored in _session_histories dict)."""
    _session_histories[session_id] = history
    return

# --- Pinecone Config ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "people-team")

if not PINECONE_API_KEY:
    raise RuntimeError("Missing PINECONE_API_KEY in environment.")
if not PINECONE_INDEX_NAME:
    raise RuntimeError("Missing PINECONE_INDEX_NAME in environment.")

# --- Embeddings ---
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# --- VectorStore (Pinecone) ---
vectorstore = PineconeVectorStore(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings
)


# --- TWEAK START: Define the Retriever ---
# This is the base component for RAG. We'll use k=10 to get a good
# number of chunks, which you would later feed into a reranker.
# For now, we'll just use the top 10.
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

def format_docs_for_context(docs: List[Document]) -> str:
    """Combines retrieved documents into a single string for the LLM context."""
    return "\n\n---\n\n".join([d.page_content for d in docs])

def safe_retrieve_docs(query: str) -> List[Document]:
    """Safely retrieve documents from Pinecone with error handling."""
    try:
        return retriever.invoke(query)
    except Exception as e:
        print(f"⚠️  Pinecone retrieval error: {str(e)}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        print(f"   Traceback:\n{traceback.format_exc()}")
        # Return empty list to allow the chain to continue with no context
        return []
# --- TWEAK END ---


# --- TWEAK START: Delete Old RAG Chain ---
# This entire section is the source of your latency.
# The agent will now *be* the RAG chain, not *call* a RAG chain.

# @traceable
# def setup_qa_chain(vectorstore):
#     ... (DELETED) ...
#
# qa_chain = setup_qa_chain(vectorstore)
# --- TWEAK END ---


# --- TWEAK START: Delete Old Tools ---
# The new router will not use these `Tool` objects.
# It will use Pydantic models to classify intent.

# class DocsInput(BaseModel):
#     query: str
#
# def rag_tool_func(query: str) -> str:
#     ... (DELETED) ...
#
# rag_tool = Tool( ... (DELETED) ... )
#
# search = TavilySearch()
# def search_web(query: str) -> str:
#     ... (DELETED) ...
#
# web_tool = Tool( ... (DELETED) ... )
# --- TWEAK END ---


# --- TWEAK START: FAST ROUTER ARCHITECTURE (#5) ---

# 1. Define Router "Tools" (Pydantic Models)
# These define the "routes" our fast router can choose.
class InternalDocsQuery(BaseModel):
    """
    Routes to this tool when the user is asking about internal company knowledge,
    such as HR policies, health insurance benefits, provider lists, coverage details, 
    company-specific information, or employee benefits.
    
    DO NOT use this for: greetings, general questions, weather, news, or questions 
    that don't relate to company policies or health insurance.
    """
    query: str = Field(description="The user's query, rephrased for an internal vector search")

class WebSearchQuery(BaseModel):
    """
    Routes to this tool when the user is asking about:
    - General questions, greetings, or casual conversation
    - External, real-time events, news, or current information
    - Public information, hospital reviews, or general knowledge
    - Questions that don't relate to internal company documents
    
    Use this for ANY greeting, general question, or non-company-specific query.
    """
    query: str = Field(description="The user's query, rephrased for a public web search")

# 2. Define LLMs
# We use a FAST model for routing and a (still fast) model for generation.
# For your <2s goal, you MUST use gpt-3.5-turbo. gpt-4o is too slow.
fast_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
powerful_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0) # Use 3.5-Turbo for speed!

# 3. Define the Router Chain
# This chain's ONLY job is to output *which tool to call*.
# It's a single, fast call to gpt-3.5-turbo.

# --- FIX: ADDED THIS ROUTER PROMPT ---
# This prompt template is the fix. It takes the {'input': ..., 'chat_history': ...}
# dictionary and formats it correctly for the fast_llm.
router_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a routing assistant. Your job is to classify user queries into one of two categories:

1. **InternalDocsQuery**: Use ONLY for questions about:
   - Company health insurance policies, benefits, coverage details
   - HR policies, employee benefits, company-specific information
   - Provider lists, hospital networks, insurance plans
   - Internal company knowledge or documents

2. **WebSearchQuery**: Use for EVERYTHING ELSE, including:
   - Greetings (hello, hi, how are you)
   - General questions or casual conversation
   - Questions about external information, news, weather
   - Any question that doesn't relate to internal company documents

IMPORTANT: When in doubt, choose WebSearchQuery. Only use InternalDocsQuery for clear company/insurance/HR-related questions."""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])
# --- END FIX ---

fast_router_chain = (
    router_prompt  # <-- FIX: Added the prompt here
    | fast_llm.bind_tools(
        tools=[InternalDocsQuery, WebSearchQuery],
        tool_choice="any"  # Forces it to pick one
    )
)

# This chain now correctly takes a dict {'input': ..., 'chat_history': ...}
# and outputs a message with a tool call.
def extract_tool_call(msg):
    """Extract tool call and log the routing decision."""
    if not msg.tool_calls or len(msg.tool_calls) == 0:
        print("⚠️  Router returned no tool calls, defaulting to WebSearchQuery")
        # Return a default WebSearchQuery tool call
        from langchain_core.messages import ToolMessage
        return {
            "name": "WebSearchQuery",
            "args": {"query": ""}
        }
    tool_call = msg.tool_calls[0]
    print(f"🔀 Router selected: {tool_call.get('name', 'unknown')} with query: {tool_call.get('args', {}).get('query', 'N/A')[:50]}...")
    return tool_call

get_tool_call_chain = (fast_router_chain | extract_tool_call)


# 4. Define the FINAL Answer Chains (RAG and Web Search)

# This is your EXCELLENT persona prompt, moved from the old qa_chain.
# This will be the system prompt for BOTH of our final answer chains.
SYSTEM_PERSONA_PROMPT = """
You are a helpful, accurate, and professional QnA assistant designed to support employees of a Nigerian fintech company called Kora with health insurance-related and general HR questions.
Your goal is to provide clear, concise, and context-aware answers based on internal knowledge sources and public information, when required.

### PERSONA 
Your persona is that of a friendly and helpful HR colleague. You are based in Lagos, Nigeria, and your communication style should reflect the modern Nigerian workplace. Your goal is to be approachable and efficient.
Core Principles:
1. Be Warm & Approachable: Always start conversations with a friendly, slightly informal greeting. You're a colleague, not a robot.
2. Be Direct & To the Point: After the warm greeting, get straight to the answer. Avoid corporate jargon or long, fluffy sentences. Your colleagues are busy; respect their time.
3. Be Respectful & Professional: While your tone is friendly, the information you provide is from HR. It must be accurate, clear, and professional.
4. Use Nigerian English Naturally: Incorporate common Nigerian phrases and light Pidgin where it feels natural and conversational. This makes you more relatable to the team.

### AUDIENCE & LOCALISATION
You are interacting with employees based in Nigeria. You must interpret terms, slang, and geographic references according to local Nigerian usage.
- "The Island" refers to the Lagos Island region (Lekki, Victoria Island (VI), Ikoyi, Ajah, Oniru).
- "The Mainland" refers to areas like Yaba, Ikeja, Surulere, Maryland, and Gbagada.
- "My plan" typically refers to the employee’s active health insurance plan.
- "HMO" refers to the employee’s health maintenance organization.

### INSTRUCTIONS
- You will be given 'Context' from internal documents or 'Web Search Results'.
- **You MUST base your answer on this provided context.**
- If the context is missing, clearly state that you cannot find the information.
- Never fabricate coverage details.
- Be concise and precise, but include key details (fees, limits, waiting periods, addresses, contact numbers) when relevant from the context.
"""

# The RAG Answer Chain
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PERSONA_PROMPT + "\n\nHere is the internal 'Context' to answer the user's question:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

rag_chain = (
    RunnablePassthrough.assign(
        # --- TWEAK: Wrap the lambda and function in RunnableLambda ---
        context=(
            RunnableLambda(lambda x: x['tool_call']['args']['query'])
            | RunnableLambda(safe_retrieve_docs)
            | RunnableLambda(format_docs_for_context)
        )
        # --- END TWEAK ---
    )
    | rag_prompt
    | powerful_llm
    | StrOutputParser()
)

# The Web Search Answer Chain
web_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PERSONA_PROMPT + "\n\nHere are the 'Web Search Results' to answer the user's question:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# Initialize Tavily with API key if available
tavily_api_key = os.getenv("TAVILY_API_KEY")
if not tavily_api_key:
    print("⚠️  Warning: TAVILY_API_KEY not set. Web search will fail.")
tavily_search = TavilySearch(api_key=tavily_api_key) if tavily_api_key else None

def safe_tavily_search(query: str) -> str:
    """Safely run Tavily search with error handling."""
    if not tavily_search:
        return "Web search is not available. TAVILY_API_KEY is not configured."
    try:
        return tavily_search.run(query)
    except Exception as e:
        print(f"⚠️  Tavily search error: {str(e)}")
        return f"Web search encountered an error: {str(e)}. Please try rephrasing your question or ask about internal company information instead."

web_search_chain = (
    RunnablePassthrough.assign(
        # --- TWEAK: Wrap the lambda and method in RunnableLambda ---
        context=(
            RunnableLambda(lambda x: x['tool_call']['args']['query'])
            | RunnableLambda(safe_tavily_search)
        )
        # --- END TWEAK ---
    )
    | web_prompt
    | powerful_llm
    | StrOutputParser()
)

# 5. Define the Branch
# This `RunnableBranch` inspects the tool call from the router
# and runs the correct chain (RAG or Web).
def route_to_chain(x):
    """Route based on tool call name with better error handling."""
    try:
        tool_name = x.get('tool_call', {}).get('name', '')
        if tool_name == 'InternalDocsQuery':
            print("📚 Routing to Internal Docs (RAG)")
            return rag_chain
        elif tool_name == 'WebSearchQuery':
            print("🌐 Routing to Web Search (Tavily)")
            return web_search_chain
        else:
            print(f"⚠️  Unknown tool call: {tool_name}, defaulting to Web Search")
            return web_search_chain  # Default to web search for safety
    except Exception as e:
        print(f"⚠️  Error in router branch: {str(e)}, defaulting to Web Search")
        return web_search_chain

router_branch = RunnableBranch(
    (lambda x: x.get('tool_call', {}).get('name') == 'InternalDocsQuery', rag_chain),
    (lambda x: x.get('tool_call', {}).get('name') == 'WebSearchQuery', web_search_chain),
    web_search_chain  # Default to Web Search for safety (better for general questions)
)

# 6. Add a lightweight "conversation memory" path
# This prevents web search from hijacking questions like "what's my name?"
def _is_memory_question(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    # Personal-memory / conversation-state questions
    if "what's my name" in t or "whats my name" in t or "what is my name" in t:
        return True
    if "who am i" in t:
        return True
    if "do you remember" in t and "name" in t:
        return True
    if ("what did i" in t or "what have i" in t) and ("say" in t or "tell you" in t):
        return True
    return False

memory_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PERSONA_PROMPT + "\n\nAnswer ONLY using the chat history. Do NOT use web search. If the answer is not present in the chat history, say you don't know."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

memory_chain = (
    memory_prompt
    | powerful_llm
    | StrOutputParser()
)

# 6. Create the Final Agent
# This is the new `agent_executor`.
# 1. It passes through the original 'input' and 'chat_history'.
# 2. It *adds* a 'tool_call' key by running the fast_router_chain.
# 3. It pipes all of this to the `router_branch`, which selects and runs the correct final chain.
agent_executor = RunnableBranch(
    (lambda x: _is_memory_question(x.get("input", "")), memory_chain),
    (RunnablePassthrough.assign(tool_call=get_tool_call_chain) | router_branch),
)

# --- TWEAK END ---

# --- TWEAK START: Delete Old Agent ---
# We no longer need the old agent definition.
#
# nigerian_system_prompt = """ ... (DELETED) ... """
# prompt = ChatPromptTemplate.from_messages([ ... (DELETED) ... ])
# llm = ChatOpenAI(model="gpt-4o", temperature=0)
# agent = create_openai_functions_agent( ... (DELETED) ... )
# agent_executor = AgentExecutor(agent=agent, tools=[rag_tool, web_tool], verbose=True)
# --- TWEAK END ---


# --- Token Budgeting Utilities ---
# (This section is unchanged, it's perfect)

def _encoding_for_model(model_name: str):
    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        # Fallback to cl100k_base, which supports most models
        return tiktoken.get_encoding("cl100k_base")


def _count_message_tokens(messages: List[BaseMessage], model_name: str) -> int:
    enc = _encoding_for_model(model_name)
    total = 0
    # Just count content tokens; role overhead is minor vs content
    for m in messages:
        content = m.content if isinstance(m.content, str) else str(m.content)
        total += len(enc.encode(content))
    return total


def prune_chat_history_in_place(
    history,
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

    try:
        history.messages = kept
    except Exception:
        # For SQL-backed histories, store pruned copy for downstream usage
        setattr(history, "_pruned_messages", kept)

# --- FastAPI ---
app = FastAPI()

# Add CORS middleware to allow requests from backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific backend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    """Health check endpoint for production deployment"""
    return {
        "status": "healthy",
        "service": "kora-rag-service",
        "version": "2.0.0"
    }

class ChatRequest(BaseModel):
    session_id: str
    message: str
    user_id: str = None  # Optional, for future use

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    sid = req.session_id
    current_session_id.set(sid)

    history = get_agent_history(sid)

    # add the user message to history and prune prior to invoke
    history.add_user_message(req.message)
    prune_chat_history_in_place(history)

    try:
        # TWEAK: The input to our new chain is a simple dictionary
        result = agent_executor.invoke({
            "input": req.message,
            "chat_history": getattr(history, "_pruned_messages", history.messages)
        })
        # The result is now just the final string, not a dict
        answer = result
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Error processing chat request: {str(e)}")
        print(f"   Full traceback:\n{error_details}")
        # Return a user-friendly error message in the expected format
        return {"response": f"I'm sorry, I encountered an error processing your request: {str(e)}. Please try again or rephrase your question."}

    history.add_ai_message(answer)
    save_agent_history(sid, history)
    return {"response": answer}

# -----------------------------
# Interactive CLI Chat
# -----------------------------

def interactive_chat(session_id: str = "local-dev"):
    print("HMO RAG Agent (Supabase + Postgres) — CLI mode")
    print("Type your message. Commands: /reset  /exit")
    print(f"Active session: {session_id}\n")

    while True:
        user = input("You: ").strip()
        if not user:
            continue
        if user.lower() in {"/exit", "exit", "quit"}:
            break
        if user.lower().startswith("/reset"):
            # New session id or clear history for current session
            print("(Resetting chat history for current session)")
            # SQLChatMessageHistory doesn't have a delete all by default here;
            # simplest is to switch to a new session id or handle deletion externally.
            session_id = session_id + "-reset"
            print(f"New session: {session_id}")
            continue

        current_session_id.set(session_id)
        history = get_agent_history(session_id)

        # Add user message & prune
        history.add_user_message(user)
        prune_chat_history_in_place(history)

        # Run the agent
        try:
            # TWEAK: The input to our new chain is a simple dictionary
            res = agent_executor.invoke({
                "input": user,
                "chat_history": getattr(history, "_pruned_messages", history.messages)
            })
            # The result is now just the final string
            answer = res
        except Exception as e:
            print(f"[err] {e}")
            continue

        history.add_ai_message(answer)
        save_agent_history(session_id, history)

        print(f"Agent: {answer}\n")

# --- CLI ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    # ingest_parser = subparsers.add_parser("ingest", help="Ingest docs into the vector store")
    # ingest_parser.add_argument("files", nargs="+", help="PDF/XLSX files to ingest")

    p_serve = subparsers.add_parser("serve", help="Run the FastAPI server")
    p_serve.add_argument("--port", type=int, default=8080)

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