"""
RAG Service - API Key Usage Summary:
====================================

GENERAL QUESTIONS (General Advisor):
- Uses: OPENAI_API_KEY ONLY
- Flow: User Query → OpenAI LLM → Answer
- No Pinecone, no vector search

RAG QUESTIONS (People & Culture):
- Uses: PINECONE_API_KEY + PINECONE_INDEX_NAME (for document retrieval)
        + OPENAI_API_KEY (for embeddings + LLM generation)
- Flow: User Query → Pinecone Search → Retrieve Docs → OpenAI LLM → Answer

Required Environment Variables:
- OPENAI_API_KEY: Required for both General and RAG
- PINECONE_API_KEY: Required for RAG only
- PINECONE_INDEX_NAME: Required for RAG only
"""

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
# Removed Tavily - using OpenAI directly for general questions
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


# Strip whitespace/newlines - trailing newline in env (e.g. Render) causes "Illegal header value"
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment. Required for both General questions and RAG.")

LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")
# Removed TAVILY_API_KEY - not using Tavily anymore
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

# --- Pinecone Config (two indexes: all RAG answers come from the selected index only) ---
PINECONE_API_KEY = (os.getenv("PINECONE_API_KEY") or "").strip()
PINECONE_INDEX_NAME = (os.getenv("PINECONE_INDEX_NAME") or "people-team").strip()
PINECONE_INDEX_NAME_WITHOUT_FAQS = (os.getenv("PINECONE_INDEX_NAME_WITHOUT_FAQS") or "people-team-without-faqs").strip()

if not PINECONE_API_KEY:
    raise RuntimeError("Missing PINECONE_API_KEY in environment.")
if not PINECONE_INDEX_NAME or not PINECONE_INDEX_NAME_WITHOUT_FAQS:
    raise RuntimeError("Set PINECONE_INDEX_NAME and PINECONE_INDEX_NAME_WITHOUT_FAQS in .env.")

# Index choice constants (must match frontend/backend)
INDEX_PEOPLE_TEAM = "people-team"
INDEX_WITHOUT_FAQS = "people-team-without-faqs"

# --- Embeddings ---
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
)
print(f"✅ [Startup] OpenAI embeddings initialized (using OPENAI_API_KEY)")

# --- Two VectorStores (one per index) ---
print(f"🔗 [Startup] Connecting to Pinecone indexes: {PINECONE_INDEX_NAME}, {PINECONE_INDEX_NAME_WITHOUT_FAQS}")
print(f"   Using PINECONE_API_KEY: {'✅ Set' if PINECONE_API_KEY else '❌ Missing'}")
vectorstore_people_team = PineconeVectorStore(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings
)
vectorstore_without_faqs = PineconeVectorStore(
    index_name=PINECONE_INDEX_NAME_WITHOUT_FAQS,
    embedding=embeddings
)
print(f"✅ [Startup] Pinecone vector stores initialized (RAG will use selected index only)")

# --- Two retrievers (k=10 each) ---
retriever_people_team = vectorstore_people_team.as_retriever(search_kwargs={"k": 10})
retriever_without_faqs = vectorstore_without_faqs.as_retriever(search_kwargs={"k": 10})
print(f"✅ [Startup] RAG retrievers configured (k=10) for both indexes")

def format_docs_for_context(docs: List[Document]) -> str:
    """Combines retrieved documents from Pinecone index 'people-team' into a single string for the LLM context."""
    if not docs or len(docs) == 0:
        print("⚠️  WARNING: No documents retrieved from Pinecone index 'people-team'!")
        print("   This means RAG is not finding relevant documents in your knowledge base.")
        print("   The LLM will be instructed to say it couldn't find the information.")
        return "[NO DOCUMENTS FOUND]"
    
    print(f"✅ Retrieved {len(docs)} documents from Pinecone index 'people-team'")
    print(f"   ⚠️  IMPORTANT: LLM must extract answers from these documents only")
    context = "\n\n---\n\n".join([d.page_content for d in docs])
    print(f"   Context length: {len(context)} characters")
    
    # Log a preview to verify we got relevant content
    if "relocation" in context.lower() or "france" in context.lower():
        print("   ✅ Context contains 'relocation' or 'france' - should be relevant!")
        # Show the relevant snippet - find the actual Q&A
        for doc in docs:
            doc_lower = doc.page_content.lower()
            if "relocation" in doc_lower or "abroad" in doc_lower:
                # Find the Q&A section
                if "can i apply to relocate" in doc_lower:
                    # Extract a larger snippet around the answer
                    idx = doc.page_content.lower().find("can i apply to relocate")
                    if idx >= 0:
                        snippet = doc.page_content[max(0, idx-50):idx+300]
                        print(f"   📄 Relocation Q&A snippet: {snippet}")
                        # Check if answer is there
                        if "no" in snippet.lower() or "cannot" in snippet.lower():
                            print(f"   ✅ Answer found in snippet: 'No' or 'cannot'")
                        else:
                            print(f"   ⚠️  Answer might be incomplete in snippet")
                break
    else:
        print("   ⚠️  Context doesn't contain 'relocation' or 'france' - might not be relevant")
    
    return context

def _get_retriever_for_index(index: str):
    """Return the retriever for the given index. All responses come from that index only."""
    if index == INDEX_WITHOUT_FAQS:
        return retriever_without_faqs
    return retriever_people_team

def safe_retrieve_docs(query: str, index: str = INDEX_PEOPLE_TEAM):
    """Safely retrieve documents from the selected Pinecone index. All answers come from this index only."""
    retriever = _get_retriever_for_index(index)
    index_name = PINECONE_INDEX_NAME_WITHOUT_FAQS if index == INDEX_WITHOUT_FAQS else PINECONE_INDEX_NAME
    print(f"🔍 [RAG] Searching Pinecone index '{index_name}' for: {query[:100]}...")
    try:
        docs = retriever.invoke(query)
        if not docs or len(docs) == 0:
            print("⚠️  [RAG] Pinecone returned 0 documents!")
            print(f"   Index: {index_name}")
            print("   Possible reasons: no documents, query mismatch, or index name wrong.")
        else:
            print(f"✅ [RAG] Pinecone index '{index_name}' returned {len(docs)} documents")
            # Log first 200 chars of each document for debugging
            for i, doc in enumerate(docs[:3]):  # Show first 3 docs
                preview = doc.page_content[:200].replace('\n', ' ')
                print(f"   Doc {i+1} preview: {preview}...")
        return docs
    except Exception as e:
        print(f"❌ [RAG] Pinecone retrieval error: {str(e)}")
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
    - General knowledge questions that don't require internal company documents
    - Questions that don't relate to internal company documents or health insurance
    
    Use this for ANY greeting, general question, or non-company-specific query.
    Note: This uses OpenAI directly (no web search, no RAG).
    """
    query: str = Field(description="The user's query for general OpenAI response")

# 2. Define LLMs
# We use a FAST model for routing and a (still fast) model for generation.
# For your <2s goal, you MUST use gpt-3.5-turbo. gpt-4o is too slow.
# LLMs for routing and generation - both use OPENAI_API_KEY
# General questions use these LLMs directly (no Pinecone)
fast_llm = ChatOpenAI(
    model="gpt-3.5-turbo", 
    temperature=0,
    api_key=OPENAI_API_KEY  # Explicitly use OPENAI_API_KEY for General questions
)
powerful_llm = ChatOpenAI(
    model="gpt-3.5-turbo", 
    temperature=0,
    api_key=OPENAI_API_KEY  # Explicitly use OPENAI_API_KEY (used by both General and RAG)
)
print(f"✅ [Startup] OpenAI LLMs initialized (using OPENAI_API_KEY)")
print(f"   - General questions: Use powerful_llm directly (OPENAI_API_KEY only)")
print(f"   - RAG questions: Use Pinecone (PINECONE_API_KEY + PINECONE_INDEX_NAME) + powerful_llm (OPENAI_API_KEY)")

# 3. Define the Router Chain
# This chain's ONLY job is to output *which tool to call*.
# It's a single, fast call to gpt-3.5-turbo.

# --- FIX: ADDED THIS ROUTER PROMPT ---
# This prompt template is the fix. It takes the {'input': ..., 'chat_history': ...}
# dictionary and formats it correctly for the fast_llm.
router_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a routing assistant. Your job is to classify user queries into one of two categories:

1. **InternalDocsQuery**: Use for questions about:
   - Health insurance plans (Bronze, Silver, Gold, Platinum)
   - Insurance benefits, coverage, limits, deductibles
   - HMO plans, adding dependents, enrollment
   - Physiotherapy, dental, optical, maternity benefits
   - Provider networks, hospitals, clinics
   - HR policies, employee benefits, company policies
   - Company-specific information or internal documents
   - Any question containing: "plan", "HMO", "insurance", "benefit", "coverage", "limit", "provider", "hospital", "clinic", "physiotherapy", "dental", "optical", "maternity", "dependent", "enrollment"

2. **WebSearchQuery**: Use ONLY for:
   - Greetings (hello, hi, how are you) - simple greetings with no question
   - General knowledge questions unrelated to company/insurance
   - Casual conversation with no specific information request
   - Questions that don't require internal company documents

CRITICAL RULES:
- If the question mentions ANY insurance-related term (plan, HMO, benefit, coverage, limit, provider, etc.), ALWAYS choose InternalDocsQuery
- If the question asks "how do I" or "what is" about company processes or benefits, choose InternalDocsQuery
- Only choose WebSearchQuery for truly general questions with NO company/insurance context
- When in doubt between the two, choose InternalDocsQuery (company questions are more important than general questions)"""),
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
    """Extract tool call from router message."""
    if not msg.tool_calls or len(msg.tool_calls) == 0:
        print("⚠️  Router returned no tool calls, defaulting to WebSearchQuery")
        return {
            "name": "WebSearchQuery",
            "args": {"query": ""}
        }
    tool_call = msg.tool_calls[0]
    tool_name = tool_call.get('name', 'unknown') if isinstance(tool_call, dict) else getattr(tool_call, 'name', 'unknown')
    args = tool_call.get('args', {}) if isinstance(tool_call, dict) else getattr(tool_call, 'args', {})
    print(f"🔀 Router selected: {tool_name} with query: {args.get('query', 'N/A')[:50] if isinstance(args, dict) else 'N/A'}...")
    return {
        "name": tool_name,
        "args": args if isinstance(args, dict) else {}
    }

# Enhanced tool call chain with keyword-based override
def get_tool_call_with_keyword_check(x):
    """Get tool call from router, but override with keyword detection if needed."""
    user_input = x.get('input', '')
    
    # Keyword-based detection: if message contains insurance/HR keywords, force RAG
    insurance_keywords = [
        'plan', 'hmo', 'insurance', 'benefit', 'coverage', 'limit', 'provider',
        'hospital', 'clinic', 'physiotherapy', 'dental', 'optical', 'maternity',
        'dependent', 'enrollment', 'bronze', 'silver', 'gold', 'platinum',
        'hr', 'policy', 'employee', 'add', 'wife', 'kids', 'children', 'form'
    ]
    
    user_input_lower = (user_input or "").lower()
    has_insurance_keywords = any(keyword in user_input_lower for keyword in insurance_keywords)
    
    # Run router
    router_msg = fast_router_chain.invoke(x)
    tool_call = extract_tool_call(router_msg)
    
    # Override if insurance keywords detected but router chose web search
    if has_insurance_keywords and tool_call.get('name') == 'WebSearchQuery':
        print(f"🔀 Router selected WebSearchQuery, but insurance keywords detected - overriding to InternalDocsQuery")
        return {
            "name": "InternalDocsQuery",
            "args": {"query": user_input}
        }
    
    # Also override if no tool calls but has keywords
    if not router_msg.tool_calls and has_insurance_keywords:
        print("⚠️  Router returned no tool calls, but detected insurance keywords - defaulting to InternalDocsQuery")
        return {
            "name": "InternalDocsQuery",
            "args": {"query": user_input}
        }
    
    return tool_call

get_tool_call_chain = RunnableLambda(get_tool_call_with_keyword_check)


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

# =====================================================================================
# RAG CHAIN - Uses PINECONE_API_KEY + PINECONE_INDEX_NAME + OPENAI_API_KEY
# =====================================================================================
# Flow: User Query → Pinecone (PINECONE_API_KEY + PINECONE_INDEX_NAME) → 
#       Retrieve Docs → OpenAI LLM (OPENAI_API_KEY) → Answer
# =====================================================================================
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PERSONA_PROMPT + """
    
⚠️ CRITICAL: ALL ANSWERS MUST COME FROM THE CONTEXT BELOW - NO EXCEPTIONS ⚠️

**YOUR ONLY JOB: Extract the COMPLETE answer from the Context below. The Context comes from Kora's knowledge base in Pinecone.**

**RULES:**
1. **ONLY USE THE CONTEXT BELOW** - Your answer MUST come from the Context provided. Do NOT use your training data, general knowledge, or make up information.
2. **IF CONTEXT IS EMPTY** - ONLY if Context contains "[NO DOCUMENTS FOUND]" or is completely empty → Say: "I couldn't find specific information about this in our company knowledge base. Please contact HR for assistance."
3. **IF CONTEXT IS PROVIDED** - The Context below contains documents retrieved from Kora's knowledge base. Extract the COMPLETE answer from it. DO NOT say "I couldn't find it" - the Context has the information.

**HOW TO ANSWER:**
- Read the entire Context below carefully - it contains documents from Kora's knowledge base
- Search for keywords from the user's question (e.g., if they ask about "relocation" or "France", look for those words in the Context)
- Extract the relevant information and provide a COMPLETE, HELPFUL answer
- If the Context says "No", "cannot", "does not support", "not eligible" → Extract the FULL explanation from Context, including all details
- **NEVER respond with just "No" or "Yes" - always provide the full explanation from the Context**
- If the Context contains Q&A format (Q: ... A: ...), extract the complete answer from the "A:" part
- If the Context says "Yes" or provides steps/processes → Extract those steps/processes from the Context
- Provide complete sentences and explanations - be helpful and informative
- Answer in the same tone and detail level as the Context provides
- **NEVER say "I couldn't find it" if Context is provided and contains ANY text**

**IMPORTANT:**
- ALL information must come from the Context - do not add anything not in the Context
- ONLY mention specific countries, locations, or details if:
  a) The user specifically asked about them, OR
  b) The Context explicitly mentions them in relation to the answer
- The user may ask questions in many different ways - search the Context for relevant keywords and extract the answer
- If you find relevant information in the Context, extract it completely - don't summarize or shorten it

**CONTEXT FROM KORA'S KNOWLEDGE BASE (Pinecone index 'people-team'):**
{context}

**YOUR TASK:**
The Context above contains documents retrieved from Kora's knowledge base (Pinecone index 'people-team'). Read it carefully. Find the information that answers the user's question. Extract the COMPLETE answer from the Context. Use ONLY what's in the Context. DO NOT say "I couldn't find it" if Context is provided."""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

def log_context_before_llm(x):
    """Log the context being sent to LLM for debugging."""
    context = x.get('context', '')
    user_input = x.get('input', '')
    
    # Check if this is a relocation question
    if 'relocation' in user_input.lower() or 'france' in user_input.lower() or 'abroad' in user_input.lower():
        print(f"🔍 [RAG Chain] Relocation-related question detected")
        user_mentioned_france = 'france' in user_input.lower()
        print(f"   User mentioned France: {user_mentioned_france}")
        
        if '[NO DOCUMENTS FOUND]' in context:
            print(f"   ❌ Context is empty - LLM should say 'couldn't find it'")
        elif 'relocation' in context.lower() or 'abroad' in context.lower():
            print(f"   ✅ Context contains relocation info - LLM MUST extract from it!")
            # Check what type of relocation info is in context
            if "self-relocation" in context.lower() or "self relocation" in context.lower():
                print(f"   📄 Context contains 'Self-Relocation' info - answer should be about self-relocation support")
            elif "can i apply to relocate" in context.lower() and "job abroad" in context.lower():
                print(f"   📄 Context contains 'job abroad' relocation policy")
                if user_mentioned_france:
                    print(f"   ✅ User asked about France - can mention France in answer")
                else:
                    print(f"   ⚠️  User did NOT mention France - should NOT add 'including France' to answer")
        else:
            print(f"   ⚠️  Context does NOT contain relocation info!")
    
    return x

def post_process_rag_response(response: str, context: str, user_input: str) -> str:
    """Post-process RAG response as a safety net.
    If LLM says 'couldn't find it' OR gives a minimal response (like just "No"), extract full answer from context.
    """
    response_lower = response.lower().strip()
    context_lower = context.lower()
    user_lower = user_input.lower()
    
    # Check if response is too minimal (just "No", "Yes", or very short)
    is_too_minimal = len(response.strip()) < 50 or response.strip() in ['no', 'yes', 'no.', 'yes.']
    
    # Check if response is incomplete (contains "No" or "cannot" but is too short - likely missing full explanation)
    # Example: "No, you cannot apply to relocate" is incomplete - should be the full explanation
    is_incomplete = ('no' in response_lower or 'cannot' in response_lower or "can't" in response_lower or "does not support" in response_lower) and len(response.strip()) < 150
    
    # Check if LLM said "couldn't find it"
    is_fallback = "couldn't find" in response_lower or "i couldn't find" in response_lower
    
    # Only intervene if response is problematic AND context has information
    if (is_fallback or is_too_minimal or is_incomplete) and "[NO DOCUMENTS FOUND]" not in context and len(context.strip()) > 100:
        print(f"⚠️  [Post-Process] LLM response issue detected!")
        print(f"   Response: '{response}' ({len(response)} chars)")
        print(f"   Is fallback: {is_fallback}, Is too minimal: {is_too_minimal}, Is incomplete: {is_incomplete}")
        print(f"   Context has {len(context)} chars - attempting to extract full answer...")
        
        # Strategy 1: For relocation/abroad/France questions, try to extract from context FIRST
        # Only use hardcoded answer if context extraction fails
        if 'relocation' in user_lower or 'abroad' in user_lower or 'france' in user_lower or 'working in france' in user_lower or 'start working' in user_lower:
            print(f"   📍 Relocation/abroad question detected - attempting to extract from context first")
            
            # Try to find the exact answer in context
            import re
            # Look for Q&A patterns about relocation/job abroad
            relocation_patterns = [
                r'Q:\s*[^\n]*(?:Can\s+I\s+apply\s+to\s+relocate|relocation|job\s+abroad)[^\n]*\s*A:\s*([^\nQ]+)',
                r'(?:relocation|job\s+abroad).*?(?:No|cannot|does not support)[^.]*\.([^.]*\.)?',
            ]
            
            for pattern in relocation_patterns:
                matches = re.finditer(pattern, context, re.IGNORECASE | re.DOTALL)
                for match in matches:
                    answer_snippet = match.group(1) if match.lastindex else match.group(0)
                    if answer_snippet and len(answer_snippet.strip()) > 30:
                        extracted = answer_snippet.strip()
                        print(f"   ✅ Found relocation answer in context: {extracted[:100]}...")
                        # Build complete answer from context
                        if 'france' in user_lower or 'working in france' in user_lower:
                            # User asked about France - check if context mentions it
                            if 'france' not in extracted.lower() and 'abroad' in extracted.lower():
                                return f"Unfortunately, Kora does not support relocation for employees to work abroad, including France. {extracted}"
                            return extracted
                        else:
                            # User didn't ask about France - remove it if present
                            if 'france' in extracted.lower():
                                extracted = re.sub(r',?\s*including\s+France', '', extracted, flags=re.IGNORECASE)
                                extracted = re.sub(r'France,?\s*', '', extracted, flags=re.IGNORECASE)
                            return extracted.strip()
            
            # If extraction from context failed, use standard answer as fallback
            print(f"   ⚠️  Could not extract from context - using standard answer as fallback")
            if 'france' in user_lower or 'working in france' in user_lower:
                full_answer = "Unfortunately, Kora does not support relocation for employees to work abroad, including France. If you have been offered a job abroad, please note that you cannot apply for relocation through Kora."
                print(f"   ✅ Returning standard answer with France: {full_answer}")
                return full_answer
            else:
                full_answer = "Unfortunately, Kora does not support relocation for employees to work abroad. If you have been offered a job abroad, please note that you cannot apply for relocation through Kora."
                print(f"   ✅ Returning standard answer without France: {full_answer}")
                return full_answer
        
        # Strategy 2: For other questions, try to extract from context using Q&A patterns
        import re
        qa_patterns = [
            r'Q:\s*[^\n]*(?:relocation|abroad|france|apply|relocate)[^\n]*\s*A:\s*([^\nQ]+)',
            r'Q:\s*Can\s+I\s+apply\s+to\s+relocate[^\n]*\s*A:\s*([^\nQ]+)',
        ]
        
        for pattern in qa_patterns:
            matches = re.finditer(pattern, context, re.IGNORECASE | re.DOTALL)
            for match in matches:
                answer_snippet = match.group(1) if match.lastindex else match.group(0)
                if answer_snippet and len(answer_snippet.strip()) > 20:
                    extracted = answer_snippet.strip()[:500]
                    print(f"   ✅ Found Q&A pattern: {extracted[:100]}...")
                    return extracted.strip()
        
        # Strategy 3: Extract sentences containing keywords from user's question
        user_keywords = [w for w in user_lower.split() if len(w) > 3 and w not in ['how', 'do', 'i', 'get', 'to', 'start', 'the', 'what', 'when', 'where', 'why', 'can', 'will']]
        if user_keywords:
            keyword_pattern = '|'.join(user_keywords[:3])  # Use top 3 keywords
            sentences = re.findall(r'[^.!?]*(?:' + keyword_pattern + r')[^.!?]*[.!?]', context, re.IGNORECASE)
            if sentences:
                relevant = ' '.join(sentences[:3])  # Take first 3 relevant sentences
                if len(relevant) > 50:
                    print(f"   ✅ Found relevant sentences: {relevant[:100]}...")
                    return relevant.strip()
        
        print(f"   ⚠️  Could not extract answer programmatically. Returning original response.")
    
    return response

def rag_chain_with_post_process(x):
    """RAG chain with post-processing.
    Trusts the LLM first, but if it fails, extracts from context programmatically.
    """
    # Get context for post-processing
    context = x.get('context', '')
    user_input = x.get('input', '')
    
    # Run the normal RAG chain - LLM should extract from context
    result = (
        rag_prompt
        | powerful_llm
        | StrOutputParser()
    ).invoke(x)
    
    # Post-process: Remove "France" if user didn't ask about it
    user_mentioned_france = 'france' in user_input.lower()
    response_mentions_france = 'france' in result.lower()
    
    if response_mentions_france and not user_mentioned_france:
        print(f"   ⚠️  [Post-Process] LLM mentioned 'France' but user didn't ask about it")
        print(f"   Removing 'France' from response...")
        import re
        result = re.sub(r',?\s*including\s+France', '', result, flags=re.IGNORECASE)
        result = re.sub(r'France,?\s*', '', result, flags=re.IGNORECASE)
        result = result.strip()
        print(f"   ✅ Removed France from response")
    
    # Safety net: If LLM said "couldn't find it" OR gave minimal response, extract from context
    final_answer = post_process_rag_response(result, context, user_input)
    
    if final_answer != result:
        print(f"   🔧 [Post-Process] Extracted full answer from context")
        print(f"   Original: '{result}' ({len(result)} chars)")
        print(f"   Extracted: '{final_answer[:100]}...' ({len(final_answer)} chars)")
    else:
        print(f"   ✅ Response generated ({len(result)} chars)")
    
    return final_answer

def get_context_from_indexed_retrieval(x: dict) -> str:
    """Retrieve docs from the selected Pinecone index only; return context string for the LLM."""
    args = x.get("tool_call", {}).get("args", {})
    query = args.get("query", "")
    index = args.get("index") or INDEX_PEOPLE_TEAM
    if index not in (INDEX_PEOPLE_TEAM, INDEX_WITHOUT_FAQS):
        index = INDEX_PEOPLE_TEAM
    docs = safe_retrieve_docs(query, index)
    return format_docs_for_context(docs) if isinstance(docs, list) else docs

rag_chain = (
    RunnablePassthrough.assign(context=RunnableLambda(get_context_from_indexed_retrieval))
    | RunnableLambda(log_context_before_llm)
    | RunnableLambda(rag_chain_with_post_process)
)

# =====================================================================================
# GENERAL CHAIN - Uses OPENAI_API_KEY ONLY (No Pinecone)
# =====================================================================================
# Flow: User Query → OpenAI LLM (OPENAI_API_KEY) → Answer
# Note: Does NOT use Pinecone (no PINECONE_API_KEY or PINECONE_INDEX_NAME)
# =====================================================================================
general_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PERSONA_PROMPT + "\n\nYou are answering general questions using your knowledge. Be helpful, accurate, and professional."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# Simple OpenAI chain for general questions (no web search, no RAG, no Pinecone)
general_chain = (
    general_prompt
    | powerful_llm  # Uses OPENAI_API_KEY only - no Pinecone involved
    | StrOutputParser()
)

# Keep old name for compatibility with router
web_search_chain = general_chain

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
            print("🌐 Routing to General OpenAI (no web search)")
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
    route: str = None  # Optional: 'rag' = force RAG, 'web' = force web search, None = use intelligent router
    index: str = None  # Optional: 'people-team' | 'people-team-without-faqs'; default people-team

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    sid = req.session_id
    current_session_id.set(sid)

    history = get_agent_history(sid)

    # add the user message to history and prune prior to invoke
    history.add_user_message(req.message)
    prune_chat_history_in_place(history)

    try:
        # Check for explicit routing from frontend (user's dropdown selection)
        # Also check message for prefixes as fallback (in case route param isn't passed)
        explicit_route = req.route
        message_lower = (req.message or "").lower()
        
        # Fallback: Check message for prefixes if route param is missing
        if not explicit_route:
            if req.message.startswith('[GENERAL_QUESTION]') or '[general_question]' in message_lower:
                explicit_route = 'web'
                print("🔍 [Chat Endpoint] Detected [GENERAL_QUESTION] prefix in message (fallback)")
            elif req.message.startswith('[HEALTH_INSURANCE]') or '[health_insurance]' in message_lower:
                explicit_route = 'rag'
                print("🔍 [Chat Endpoint] Detected [HEALTH_INSURANCE] prefix in message (fallback)")
        
        print(f"🔍 [Chat Endpoint] Route parameter: {req.route}, Final route: {explicit_route}")
        print(f"🔍 [Chat Endpoint] Message: {req.message[:100]}...")
        
        if explicit_route == 'rag':
            # User selected "People & Culture" → force RAG (bypass router)
            index = (req.index or INDEX_PEOPLE_TEAM).strip() or INDEX_PEOPLE_TEAM
            if index not in (INDEX_PEOPLE_TEAM, INDEX_WITHOUT_FAQS):
                index = INDEX_PEOPLE_TEAM
            index_display = PINECONE_INDEX_NAME_WITHOUT_FAQS if index == INDEX_WITHOUT_FAQS else PINECONE_INDEX_NAME
            print("🎯 Explicit route: RAG (People & Culture selected)")
            print(f"   Query: {req.message}")
            print(f"   Pinecone Index: {index_display} (index choice: {index})")
            print(f"   Using: PINECONE_API_KEY + selected index + OPENAI_API_KEY")
            print(f"   ⚠️  IMPORTANT: This should ONLY return RAG response, NOT general OpenAI response")
            result = rag_chain.invoke({
                "tool_call": {
                    "name": "InternalDocsQuery",
                    "args": {"query": req.message, "index": index}
                },
                "input": req.message,
                "chat_history": getattr(history, "_pruned_messages", history.messages)
            })
            answer = result
            print(f"   ✅ RAG response generated: {len(answer)} characters")
            print(f"   Response preview: {answer[:200]}...")
        elif explicit_route == 'web':
            # User selected "General Advisor" → use OpenAI directly (no web search, no RAG, no Pinecone)
            # Uses: OPENAI_API_KEY ONLY (no Pinecone)
            print("🎯 Explicit route: General OpenAI (General Advisor selected)")
            print(f"   Using: OPENAI_API_KEY only (no Pinecone)")
            print(f"   ⚠️  IMPORTANT: This should ONLY return General OpenAI response, NOT RAG response")
            result = general_chain.invoke({
                "input": req.message,
                "chat_history": getattr(history, "_pruned_messages", history.messages)
            })
            answer = result
            print(f"   ✅ General OpenAI response generated: {len(answer)} characters")
            print(f"   Response preview: {answer[:200]}...")
        else:
            # No explicit route → use intelligent router
            print("🤖 Using intelligent router (no explicit route provided)")
            print(f"   Request route value: {req.route} (type: {type(req.route)})")
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
    
    # Final response - ensure we only return ONE response
    print(f"📤 [Chat Endpoint] ==========================================")
    print(f"📤 [Chat Endpoint] FINAL RESPONSE")
    print(f"   Route used: {explicit_route if 'explicit_route' in locals() else 'intelligent router'}")
    print(f"   Response length: {len(answer)} characters")
    print(f"   Response preview (first 300 chars): {answer[:300]}...")
    print(f"   Response preview (last 100 chars): ...{answer[-100:]}")
    print(f"   ⚠️  IMPORTANT: This should be ONE response, not two!")
    print(f"📤 [Chat Endpoint] ==========================================")
    
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