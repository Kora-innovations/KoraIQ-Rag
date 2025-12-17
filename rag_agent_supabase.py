import os
import argparse
import contextvars
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from regex import sub
import uvicorn

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader
from langchain_core.vectorstores import VectorStore
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory, SQLChatMessageHistory
from langchain_core.documents import Document
from langchain_core.tools import Tool
# In LangChain 1.0.5, use langgraph for agents
from langgraph.prebuilt import create_react_agent
from typing import Any, Dict

# Compatibility wrapper for AgentExecutor using langgraph
class AgentExecutor:
    def __init__(self, agent, tools, verbose=False):
        self.agent = agent
        self.tools = tools
        self.verbose = verbose
    
    def invoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # The agent from langgraph expects messages format
            # Convert input to messages format if needed
            input_text = input_dict.get("input", "")
            if not input_text or not input_text.strip():
                return {"output": "I didn't receive a valid message. Please try again."}
            
            chat_history = input_dict.get("chat_history", [])
            
            # Build messages list
            messages = []
            # Add chat history if provided
            if chat_history:
                try:
                    # Ensure chat_history is a list of message objects
                    for msg in chat_history:
                        if hasattr(msg, 'content'):
                            messages.append(msg)
                        elif isinstance(msg, dict):
                            # Convert dict to message if needed
                            if msg.get("type") == "human" or msg.get("role") == "user":
                                messages.append(HumanMessage(content=str(msg.get("content", ""))))
                            elif msg.get("type") == "ai" or msg.get("role") == "assistant":
                                messages.append(AIMessage(content=str(msg.get("content", ""))))
                except Exception as hist_error:
                    print(f" [AgentExecutor] Warning: Error processing chat history: {hist_error}")
                    # Continue without history if there's an error
            
            # Add current input (HumanMessage imported at top of file)
            messages.append(HumanMessage(content=str(input_text)))
            
            # Invoke the agent (langgraph agent expects messages)
            try:
                result = self.agent.invoke({"messages": messages})
            except Exception as agent_error:
                error_str = str(agent_error)
                print(f" [AgentExecutor] Agent invocation error: {error_str}")
                import traceback
                print(f" [AgentExecutor] Traceback: {traceback.format_exc()}")
                
                # Return a user-friendly error message
                if "timeout" in error_str.lower():
                    return {"output": "The request timed out. Please try again with a shorter question."}
                elif "rate limit" in error_str.lower():
                    return {"output": "The service is experiencing high demand. Please try again in a moment."}
                else:
                    return {"output": "I encountered an error processing your request. Please try rephrasing your question."}
            
            # Extract output from result
            if isinstance(result, dict):
                if "messages" in result and result["messages"]:
                    # Get the last message content
                    try:
                        last_msg = result["messages"][-1]
                        output = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
                    except Exception as extract_error:
                        print(f" [AgentExecutor] Error extracting message content: {extract_error}")
                        output = str(result.get("output", "I couldn't process the response. Please try again."))
                else:
                    output = result.get("output", str(result))
            else:
                output = str(result)
            
            # Ensure we have a valid output
            if not output or not output.strip():
                output = "I couldn't generate a response. Please try rephrasing your question."
            
            return {"output": output}
            
        except Exception as e:
            error_str = str(e)
            print(f" [AgentExecutor] Unexpected error: {type(e).__name__}: {error_str}")
            import traceback
            print(f" [AgentExecutor] Traceback: {traceback.format_exc()}")
            return {"output": "I encountered an unexpected error. Please try again, and if the problem persists, contact support."}

# Compatibility function for create_openai_functions_agent
def create_openai_functions_agent(llm, tools, prompt):
    # Use langgraph's create_react_agent which is the replacement
    # Note: create_react_agent doesn't use prompt parameter the same way
    # but it should work with the tools and llm
    return create_react_agent(llm, tools)
# Temporarily commented out - langchain-tavily incompatible with langchain 1.0.0+
# Will use Tavily API directly if needed
# from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langsmith import traceable
from langchain_classic.chains import RetrievalQA
# from knowledgebase import ingest_to_mongo
from langchain_core.tools import tool
from pydantic import BaseModel
import tiktoken
from openai import OpenAI

from supabase import create_client, Client

import contextvars
from typing import Dict
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

# GPT-5 client for general questions
# Configure with longer timeout for complex queries
gpt5_client = OpenAI(
    api_key=OPENAI_API_KEY,
    timeout=60.0,  # 60 second timeout for API calls
    max_retries=2  # Retry up to 2 times on transient failures
)
GPT5_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a helpful, knowledgeable AI assistant. You provide accurate and well-researched answers to general questions. You are good at conversation, remember context, and can discuss a wide range of topics including technology, business, current events, and general knowledge. If a question is unclear, ask for clarification. Always be helpful, accurate, and engaging in your responses.")

# Health insurance keywords for routing
HEALTH_KEYWORDS = [
    'health', 'insurance', 'hmo', 'provider', 'coverage', 'benefits',
    'medical', 'hospital', 'clinic', 'doctor', 'physician', 'nurse',
    'pharmacy', 'prescription', 'medication', 'treatment', 'diagnosis',
    'premium', 'deductible', 'copay', 'coinsurance', 'out-of-pocket',
    'network', 'in-network', 'out-of-network', 'referral', 'authorization',
    'claim', 'claims', 'billing', 'payment', 'reimbursement',
    'wellness', 'preventive', 'screening', 'vaccination', 'immunization',
    'maternity', 'pregnancy', 'prenatal', 'postnatal', 'delivery',
    'mental health', 'therapy', 'counseling', 'psychology', 'psychiatry',
    'dental', 'vision', 'optical', 'hearing', 'audiology',
    'emergency', 'urgent care', 'ambulance', 'er', 'emergency room',
    'specialist', 'specialty', 'cardiology', 'dermatology', 'orthopedics',
    'pediatrics', 'geriatrics', 'oncology', 'neurology', 'gastroenterology'
]

# RAG service only - no routing needed

def call_gpt5_system(message: str, session_id: str, user_id: str = None) -> str:
    """Call GPT-5 system for general questions - Direct OpenAI integration"""
    try:
        print(f" [GPT-5] Processing: {message[:50]}...")
        
        # Get conversation history
        try:
            history = get_agent_history(session_id, user_id)
        except Exception as hist_error:
            print(f" [GPT-5] Warning: History retrieval failed, continuing without history: {hist_error}")
            history = ChatMessageHistory()
        
        # Build messages for GPT-5
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Add conversation history (messages are already clean now)
        # Limit to last 10 messages to avoid token limits
        try:
            recent_messages = history.messages[-10:] if len(history.messages) > 10 else history.messages
            
            for msg in recent_messages:
                try:
                    if msg.type == "human":
                        messages.append({"role": "user", "content": str(msg.content)})
                    elif msg.type == "ai":
                        messages.append({"role": "assistant", "content": str(msg.content)})
                except Exception as msg_error:
                    print(f" [GPT-5] Warning: Error processing message: {msg_error}")
                    continue
        except Exception as hist_process_error:
            print(f" [GPT-5] Warning: Error processing history: {hist_process_error}")
        
        # Add current message (already clean)
        messages.append({"role": "user", "content": str(message)})
        
        # Call GPT-5 with standard Chat Completions API
        try:
            # Use longer timeout for complex queries - client already configured with 60s timeout
            response = gpt5_client.chat.completions.create(
                model=GPT5_MODEL,
                messages=messages,
                max_completion_tokens=2000,
                stream=False
                # Timeout is set at client level (60 seconds)
            )
            
            if not response or not response.choices:
                raise ValueError("Empty response from OpenAI")
            
            answer = response.choices[0].message.content
            if not answer or not answer.strip():
                answer = "I apologize, but I couldn't generate a response. Please try again."
                
            print(f" [GPT-5] Response generated: {len(answer)} characters")
            return answer
            
        except Exception as api_error:
            error_str = str(api_error)
            print(f" [GPT-5] OpenAI API Error: {error_str}")
            
            # Provide more specific error messages
            if "timeout" in error_str.lower():
                return "I'm experiencing a timeout while processing your request. Please try again in a moment."
            elif "rate limit" in error_str.lower() or "quota" in error_str.lower():
                return "The service is currently experiencing high demand. Please try again in a few minutes."
            elif "authentication" in error_str.lower() or "api key" in error_str.lower():
                return "There's an authentication issue with the AI service. Please contact support."
            else:
                return "I'm having trouble processing your request right now. Please try again in a moment."
        
    except Exception as e:
        error_str = str(e)
        print(f" [GPT-5] Unexpected Error: {type(e).__name__}: {error_str}")
        import traceback
        print(f" [GPT-5] Traceback: {traceback.format_exc()}")
        return "I encountered an unexpected error. Please try again, and if the problem persists, contact support."

# --- Postgres-backed Session History (via SQLChatMessageHistory) ---
# Mongo-specific code removed. We now rely on SQLChatMessageHistory, which persists directly to Postgres.

# In-memory history store as fallback when PostgreSQL fails
_in_memory_histories: Dict[str, ChatMessageHistory] = {}

def get_agent_history(session_id: str, user_id: str = None):
    """Return a Postgres-backed chat history.
    Uses SQLChatMessageHistory under the hood so adds are auto-persisted.
    In production, uses user_id to create user-specific session IDs.
    """
    # Create user-specific session ID for production
    if user_id:
        effective_session_id = f"user_{user_id}_{session_id}"
        print(f" [History] Getting history for user {user_id}, session: {session_id}")
        print(f" [History] Effective session ID: {effective_session_id}")
    else:
        effective_session_id = session_id
        print(f" [History] Getting history for session: {session_id}")
    
    # Get connection string
    postgres_url = os.getenv("POSTGRES_URL") or os.getenv("DATABASE_URL")
    
    if not postgres_url:
        print(f" [History] Warning: No POSTGRES_URL or DATABASE_URL found, using in-memory history")
        # Use persistent in-memory store
        if effective_session_id not in _in_memory_histories:
            _in_memory_histories[effective_session_id] = ChatMessageHistory()
        return _in_memory_histories[effective_session_id]
    
    # Validate connection string format (for Supabase)
    # Supabase format: postgresql://postgres.[project-ref]:[password]@[host]:[port]/postgres
    # Or: postgresql+psycopg://postgres.[project-ref]:[password]@[host]:[port]/postgres
    if "postgres." not in postgres_url and "postgresql" in postgres_url:
        # Try to construct Supabase connection string if we have SUPABASE_URL
        supabase_url = os.getenv("SUPABASE_URL", "")
        if supabase_url and "supabase.co" in supabase_url:
            print(f" [History] Warning: POSTGRES_URL may be in wrong format for Supabase")
            print(f" [History] Expected format: postgresql://postgres.[project-ref]:[password]@aws-0-[region].pooler.supabase.com:6543/postgres")
    
    try:
        history = SQLChatMessageHistory(session_id=effective_session_id, connection=postgres_url)
        
        # SQLChatMessageHistory doesn't automatically load messages, so we need to fetch them
        # The messages are stored in the database but not loaded into the history object
        # We'll use a workaround by checking if messages exist and loading them
        
        # Try to get the first message to see if history exists
        try:
            # This will trigger a database query to load messages
            messages = history.messages
            if messages:
                print(f" [History] Loaded {len(messages)} messages for session {effective_session_id}")
            else:
                print(f" [History] No existing messages for session {effective_session_id}")
        except Exception as load_error:
            print(f" [History] Error loading messages: {load_error}")
            
        return history
    except Exception as e:
        error_msg = str(e)
        # Mask password in error message for security
        if "password" in error_msg.lower() or "@" in error_msg:
            # Don't print full connection string
            print(f" [History] Warning: PostgreSQL connection failed, using in-memory history")
            print(f" [History] Error type: {type(e).__name__}")
            if "Tenant or user not found" in error_msg:
                print(f" [History] Fix: Check POSTGRES_URL format. For Supabase, use: postgresql://postgres.[project-ref]:[password]@aws-0-[region].pooler.supabase.com:6543/postgres")
            elif "connection" in error_msg.lower():
                print(f" [History] Fix: Check POSTGRES_URL host, port, and network connectivity")
        else:
            print(f" [History] Warning: PostgreSQL connection failed, using in-memory history: {error_msg}")
        # Fallback to persistent in-memory history
        if effective_session_id not in _in_memory_histories:
            _in_memory_histories[effective_session_id] = ChatMessageHistory()
            print(f" [History] Created new in-memory history for session: {effective_session_id}")
        else:
            existing_messages = len(_in_memory_histories[effective_session_id].messages)
            print(f" [History] Using existing in-memory history with {existing_messages} messages for session: {effective_session_id}")
        return _in_memory_histories[effective_session_id]

def save_agent_history(session_id: str, history):
    # No-op: SQLChatMessageHistory persists messages on .add_* calls
    return

# --- Supabase & Postgres Config ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "hmo_documents")
SUPABASE_QUERY_NAME = os.getenv("SUPABASE_QUERY_NAME", "match_documents")

POSTGRES_URL = os.getenv("POSTGRES_URL") or os.getenv("DATABASE_URL")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("Missing SUPABASE_URL / SUPABASE_SERVICE_KEY in environment.")
if not POSTGRES_URL:
    raise RuntimeError("Missing POSTGRES_URL (e.g., 'postgresql+psycopg://user:pass@host:5432/db').")

# Verify Supabase URL format
if not SUPABASE_URL.startswith("https://"):
    print(f" [Config] WARNING: SUPABASE_URL should start with https://")
    print(f" [Config] Current URL: {SUPABASE_URL}")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
print(f" [Config] Supabase client initialized")
print(f" [Config] URL: {SUPABASE_URL}")
print(f" [Config] Table: {SUPABASE_TABLE}, Query function: {SUPABASE_QUERY_NAME}")

# Test connection by trying to access the table
try:
    test_response = supabase.table(SUPABASE_TABLE).select("id").limit(1).execute()
    print(f" [Config] ✓ Connection test successful - table is accessible")
except Exception as conn_test_error:
    error_str = str(conn_test_error)
    if "Name or service not known" in error_str or "Errno -2" in error_str:
        print(f" [Config] ✗ Connection test failed: DNS error")
        print(f" [Config] ✗ This suggests SUPABASE_URL in Render environment might be incorrect")
        print(f" [Config] ✗ Expected format: https://[project-ref].supabase.co")
    else:
        print(f" [Config] ⚠️  Connection test warning: {conn_test_error}")

# --- Embeddings ---
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# --- Custom VectorStore Wrapper for supabase 2.3.0+ compatibility ---
class CompatibleSupabaseVectorStore(VectorStore):
    """Wrapper around SupabaseVectorStore to handle supabase 2.3.0+ API changes"""
    
    def __init__(self, supabase_client, table_name, embedding, query_name):
        self.supabase_client = supabase_client
        self.table_name = table_name
        self.embedding = embedding
        self.query_name = query_name
        # Try to initialize the original SupabaseVectorStore
        # Even if it initializes, it might fail at runtime due to API changes
        # So we'll always try the original first, then fall back to direct calls
        try:
            self._vectorstore = SupabaseVectorStore(
                client=supabase_client,
                table_name=table_name,
                embedding=embedding,
                query_name=query_name
            )
            self._use_wrapper = False
            print(f" [VectorStore] SupabaseVectorStore initialized successfully")
        except Exception as e:
            print(f" [VectorStore] Failed to initialize SupabaseVectorStore: {e}")
            print(f" [VectorStore] Will use direct supabase calls as fallback")
            self._vectorstore = None
            self._use_wrapper = True
    
    def similarity_search(self, query: str, k: int = 4, **kwargs) -> List:
        """Perform similarity search"""
        print(f" [VectorStore] similarity_search called with query: '{query[:50]}...', k={k}")
        
        if not self._use_wrapper and self._vectorstore:
            try:
                print(f" [VectorStore] Attempting search with SupabaseVectorStore...")
                results = self._vectorstore.similarity_search(query, k=k, **kwargs)
                print(f" [VectorStore] ✓ SupabaseVectorStore returned {len(results)} results")
                return results
            except (AttributeError, TypeError) as e:
                error_str = str(e)
                if "params" in error_str or "SyncRPCFilterRequestBuilder" in error_str:
                    print(f" [VectorStore] ✗ Compatibility error detected: {e}")
                    print(f" [VectorStore] Switching to direct supabase calls")
                    self._use_wrapper = True
                else:
                    print(f" [VectorStore] ✗ SupabaseVectorStore error: {e}")
                    raise
        
        # Fallback: Use direct supabase client calls
        if self._use_wrapper:
            print(f" [VectorStore] Using direct supabase RPC call...")
            return self._similarity_search_direct(query, k, **kwargs)
        raise RuntimeError("VectorStore not properly initialized")
    
    def _similarity_search_direct(self, query: str, k: int, **kwargs) -> List:
        """Direct similarity search using supabase client"""
        try:
            # Generate embedding for query
            print(f" [VectorStore] Generating embedding for query...")
            query_embedding = self.embedding.embed_query(query)
            print(f" [VectorStore] ✓ Generated embedding (length: {len(query_embedding)})")
            
            # Use lower threshold to be less restrictive (0.5 instead of 0.7)
            match_threshold = kwargs.get("match_threshold", 0.5)
            
            # Call the match_documents function in Supabase
            # For supabase 2.24.0, try different RPC call methods
            data = None
            last_error = None
            
            # Method 1: Direct rpc() call (supabase 2.24.0)
            # Function signature: match_documents(filter, match_count, query_embedding)
            try:
                print(f" [VectorStore] Calling RPC function '{self.query_name}' with match_count={k}, threshold={match_threshold}...")
                # Try with filter parameter (may be optional or empty dict)
                response = self.supabase_client.rpc(
                    self.query_name,
                    {
                        "filter": {},  # Empty filter or can include threshold logic
                        "match_count": k,
                        "query_embedding": query_embedding
                    }
                ).execute()
                print(f" [VectorStore] ✓ RPC call successful")
                
                # Extract data from response
                if hasattr(response, 'data'):
                    data = response.data
                elif isinstance(response, dict) and 'data' in response:
                    data = response['data']
                elif isinstance(response, list):
                    data = response
                else:
                    print(f" [VectorStore] ⚠️  Unexpected response format: {type(response)}")
                    data = []
                
                print(f" [VectorStore] RPC returned {len(data) if isinstance(data, list) else 'non-list'} results")
                    
            except Exception as e1:
                last_error = e1
                print(f" [VectorStore] Method 1 failed: {e1}")
                # Method 2: Try without filter (in case it's optional)
                try:
                    print(f" [VectorStore] Trying without filter parameter...")
                    response = self.supabase_client.rpc(
                        self.query_name,
                        {
                            "match_count": k,
                            "query_embedding": query_embedding
                        }
                    ).execute()
                    data = response.data if hasattr(response, 'data') else (response if isinstance(response, list) else [])
                    print(f" [VectorStore] ✓ Method 2 (no filter) successful")
                except Exception as e2:
                    last_error = e2
                    print(f" [VectorStore] Method 2 failed: {e2}")
                    # Method 3: Try via postgrest with filter
                    try:
                        print(f" [VectorStore] Trying via postgrest with filter...")
                        response = self.supabase_client.postgrest.rpc(
                            self.query_name,
                            {
                                "filter": {},
                                "match_count": k,
                                "query_embedding": query_embedding
                            }
                        ).execute()
                        data = response.data if hasattr(response, 'data') else (response if isinstance(response, list) else [])
                        print(f" [VectorStore] ✓ Method 3 (postgrest with filter) successful")
                    except Exception as e3:
                        last_error = e3
                        print(f" [VectorStore] Method 3 failed: {e3}")
                        # Method 4: Try with explicit URL construction
                        try:
                            # Re-create client to ensure proper initialization
                            supabase_url = os.getenv("SUPABASE_URL")
                            supabase_key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
                            if supabase_url and supabase_key:
                                from supabase import create_client
                                temp_client = create_client(supabase_url, supabase_key)
                                response = temp_client.rpc(
                                    self.query_name,
                                    {
                                        "filter": {},
                                        "match_count": k,
                                        "query_embedding": query_embedding
                                    }
                                ).execute()
                                data = response.data if hasattr(response, 'data') else (response if isinstance(response, list) else [])
                            else:
                                raise Exception("SUPABASE_URL or SUPABASE_SERVICE_KEY not set")
                        except Exception as e5:
                            last_error = e5
                            error_str = str(e5)
                            print(f" [VectorStore] All RPC call methods failed")
                            print(f" [VectorStore] Last error: {e5}")
                            # Check if it's a connection/DNS error
                            if "Name or service not known" in error_str or "Errno -2" in error_str:
                                print(f" [VectorStore] DNS/Connection error detected")
                                print(f" [VectorStore] SUPABASE_URL: {os.getenv('SUPABASE_URL', 'NOT SET')}")
                                print(f" [VectorStore] This might be a network issue from Render to Supabase")
                                print(f" [VectorStore] Or the SUPABASE_URL environment variable in Render might be incorrect")
                            raise
            
            # Convert results to Document objects
            from langchain_core.documents import Document
            documents = []
            if isinstance(data, list):
                print(f" [VectorStore] Processing {len(data)} results...")
                for i, row in enumerate(data):
                    if isinstance(row, dict):
                        content = row.get("content", row.get("text", row.get("page_content", "")))
                        if content:
                            documents.append(Document(
                                page_content=content,
                                metadata=row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
                            ))
                        else:
                            print(f" [VectorStore] ⚠️  Row {i} has no content field")
                print(f" [VectorStore] ✓ Converted {len(documents)} results to Documents")
            elif data:
                print(f" [VectorStore] ⚠️  Unexpected data format: {type(data)}")
            else:
                print(f" [VectorStore] ⚠️  No data returned from RPC call")
            
            if not documents:
                print(f" [VectorStore] ⚠️  WARNING: No documents retrieved! This might mean:")
                print(f" [VectorStore]   1. The table is empty (need to ingest documents)")
                print(f" [VectorStore]   2. No documents match the query (threshold too high)")
                print(f" [VectorStore]   3. The RPC function returned empty results")
            
            return documents
        except Exception as e:
            print(f" [VectorStore] Direct supabase search failed: {e}")
            import traceback
            print(f" [VectorStore] Traceback: {traceback.format_exc()}")
            return []
    
    def as_retriever(self, **kwargs):
        """Return a retriever interface"""
        from langchain_core.vectorstores import VectorStoreRetriever
        return VectorStoreRetriever(vectorstore=self, **kwargs)
    
    @classmethod
    def from_texts(cls, texts, embedding, metadatas=None, **kwargs):
        """Create vectorstore from texts (required abstract method)"""
        # This is typically used for initialization, not needed for our use case
        # But required by VectorStore abstract class
        raise NotImplementedError("from_texts not implemented - use SupabaseVectorStore directly for ingestion")
    
    def add_texts(self, texts, metadatas=None, **kwargs):
        """Add texts to the vectorstore"""
        if self._vectorstore and not self._use_wrapper:
            return self._vectorstore.add_texts(texts, metadatas, **kwargs)
        raise NotImplementedError("Direct supabase add_texts not implemented")
    
    def delete(self, ids=None, **kwargs):
        """Delete documents"""
        if self._vectorstore and not self._use_wrapper:
            return self._vectorstore.delete(ids, **kwargs)
        raise NotImplementedError("Direct supabase delete not implemented")

# --- VectorStore ---
# Use compatible wrapper that handles supabase 2.3.0+ API changes
vectorstore = CompatibleSupabaseVectorStore(
    supabase_client=supabase,
    table_name=SUPABASE_TABLE,
    embedding=embeddings,
    query_name=SUPABASE_QUERY_NAME
)
print(" [VectorStore] CompatibleSupabaseVectorStore initialized")
# PDF_PATHS = ["general_faqs.pdf", "general_faqs_2.pdf"]
# XLSX_PATHS = ["axa_may_2025_provider_list.xlsx", "customised_plan_and_benefits_2025.xlsx"]
# vectorstore = ingest_to_mongo(PDF_PATHS, XLSX_PATHS)


# --- History-Aware RAG Chain ---
# ConversationalRetrievalChain already imported at top

@traceable
def setup_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    prompt = ChatPromptTemplate.from_template("""
        You are a helpful, accurate, and professional QnA assistant designed to support employees of a Nigerian fintech company called Kora with health insurance-related and general HR questions.
        
        Your goal is to provide clear, concise, and context-aware answers based on internal knowledge sources and public information, when required.
        
        ### PERSONA 
        Your persona is that of a friendly and helpful HR colleague. You are based in Lagos, Nigeria, and your communication style should reflect the modern Nigerian workplace. Your goal is to be approachable and efficient.
        Core Principles:
        1. Be Warm & Approachable: Always start conversations with a friendly, slightly informal greeting. You're a colleague, not a robot.
        2. Be Direct & To the Point: After the warm greeting, get straight to the answer. Avoid corporate jargon or long, fluffy sentences. Your colleagues are busy; respect their time.
        3. Be Respectful & Professional: While your tone is friendly, the information you provide is from HR. It must be accurate, clear, and professional.
        4. Use Nigerian English Naturally: Incorporate common Nigerian phrases and light Pidgin where it feels natural and conversational. This makes you more relatable to the team.
        Specific Instructions & Phrasing:
        * Greetings (Use these interchangeably):
            * "Hi hi, [Employee Name]"
            * "Hi [Employee Name], how you dey?"
            * "Hiya [Employee Name], how's it going?"
            * "Morning, [Employee Name]!"
            * "Hi [Employee Name], trust you're good?"
        * General Tone Words & Phrases to Use:
            * "No wahala" (No problem)
            * "Sharp sharp" (Quickly, right away)
            * "Well done" (Used as a thank you or acknowledgement of effort)
            * "Abeg" (Please, used informally to add emphasis)
        What to Avoid:
        * Overly Formal Language: Do not use phrases like "Dear Sir/Madam," "To whom it may concern," or "I hope this email finds you well."
        * Robotic Apologies: Avoid generic chatbot phrases like "I apologize for the inconvenience." Instead, say something like, "Ah, my bad on that. Let me sort it out now."
        * Forcing Slang: Use Nigerian English and Pidgin naturally. Don't force it into every sentence. The goal is clarity and relatability, not to become a caricature.
        Example Interactions:
        * User: "Hi, I haven't received my payslip for this month."
        * Ideal Response: " Thanks for flagging this. Let me check with Finance and get back to you sharp sharp."
        * User: "Can you remind me how to request leave?"
        * Ideal Response: "How far Aisha, how you dey? To request leave, just head over to the HR portal and fill the 'Leave Request' form. Let me know if you hit any roadblocks."                                      

        ### AUDIENCE & LOCALISATION
        You are interacting with employees based in Nigeria. You must interpret terms, slang, and geographic references according to local Nigerian usage.
        
        Here are important localization rules to follow:
        - "The Island" refers to the Lagos Island region, including Lekki, Victoria Island (VI), Ikoyi, Ajah, and Oniru.
        - "The Mainland" refers to areas like Yaba, Ikeja, Surulere, Maryland, and Gbagada.
        - "My plan" typically refers to the employee’s active health insurance plan.
        - "HMO" refers to the employee’s health maintenance organization.
        - Always default to Nigerian interpretations unless a user explicitly specifies an international location or context.
        
        If a user uses ambiguous terms, assume they are referring to the Nigerian context. If needed, ask clarifying questions using Nigerian-specific options (e.g., “Do you mean Lagos Island like Lekki or VI?” instead of “Long Island or Bali?”).

        ### KNOWLEDGE BASES
        You have access to internal documents:
        
        - Health plan tiers, benefit details, and coverage limits (e.g., dental, optical, maternity).
        - List of in-network hospitals, clinics, their locations, and contact details.
        - HR policies, onboarding information, and general employee guides.
        
        Always reference the appropriate document when formulating answers. You may summarize, but do not invent facts that are not present in the source.
        
        ### WEB SEARCH
        When a user asks for a recommendation based on public sentiment , feel free to perform a web search to gather recent information like ratings, reviews, or operational status. Return a summary and include the reasoning for your recommendation.
        
        ### INSTRUCTIONS
        - Always use the **retrieved context** first. If context is missing or insufficient, be honest about it.
        - Be concise and precise, but include key details (fees, limits, waiting periods, addresses, contact numbers) when relevant.
        - Clarify ambiguities. If a user asks for hospitals "on the island", assume they mean Lagos Island (Lekki/VI/Ikoyi/Ajah/Oniru). If they say "mainland", assume Yaba/Ikeja/Surulere/Maryland. Ask a quick follow-up if needed.
        - If a user asks for a recommendation based on public sentiment (e.g., "best hospitals for maternity in VI"), summarize retrieved info and optionally use the web tool.
        - **IMPORTANT**: Never fabricate coverage details. If you don't have specific information in the knowledge base, clearly state this and recommend contacting HR or the insurance provider directly.
        - If a user has the Rhodium plan, return hospital information relevant to that plan, the Platinum, the Gold, the Customised Gold, the Silver, and the Bronze.
        - If a user has the Platinum plan, return hospital information relevant to that plan, the Gold, the Customised Gold, the Silver, and the Bronze.
        - If a user has the Customised Gold plan, return hospital information relevant to that plan, the Gold plan, the Silver, and the Bronze.
        - If a user has the Silver plan, return hospital information relevant to that plan and the Bronze plan.
        - If a user has the Bronze plan, return hospital information relevant to ONLY the Bronze plan.
        - **When information is not available**: Say something like "I don't have specific information about [topic] in our current knowledge base. For the most accurate details, I recommend contacting our HR team or your health insurance provider directly."

        ### PROMPT FORMAT
        Context: {context}
        Question: {question}

    """)
    llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={
            "prompt": prompt,
        },
        return_source_documents=True
    )
    return qa_chain

qa_chain = setup_qa_chain(vectorstore)

# --- Tools ---

# Internal RAG tool
class DocsInput(BaseModel):
    query: str


def rag_tool_func(query: str) -> str:
    print(f"\n{'='*60}")
    print(f" [RAG Tool] Called with query: '{query}'")
    print(f"{'='*60}")
    
    sid = current_session_id.get(None)
    history = get_agent_history(sid)
    try:
        print(f" [RAG Tool] Invoking QA chain...")
        result = qa_chain.invoke({"query": query, "chat_history": getattr(history, "_pruned_messages", history.messages)})
        
        # Log source documents if available
        if "source_documents" in result:
            print(f" [RAG Tool] Retrieved {len(result['source_documents'])} source documents")
            for i, doc in enumerate(result['source_documents'][:3]):  # Show first 3
                print(f" [RAG Tool]   Doc {i+1}: {doc.page_content[:100]}...")
        else:
            print(f" [RAG Tool] ⚠️  No source_documents in result")
        
        answer = result["answer"] if "answer" in result else result.get("result", "")
        print(f" [RAG Tool] Generated answer (length: {len(answer)})")
        print(f" [RAG Tool] Answer preview: {answer[:200]}...")
        print(f"{'='*60}\n")
        
        return answer
    except (AttributeError, TypeError) as e:
        error_str = str(e)
        if ("params" in error_str and "SyncRPCFilterRequestBuilder" in error_str) or \
           ("object has no attribute" in error_str and "params" in error_str):
            print(f" [RAG Tool] SupabaseVectorStore compatibility error: {e}")
            print(f" [RAG Tool] This is due to supabase 2.3.0+ API changes incompatible with current langchain-community version")
            print(f" [RAG Tool] Solution: Update langchain-community to latest version or use supabase 2.0.0")
            return "I'm experiencing a technical issue accessing the knowledge base. The system administrators have been notified. Please try again later or contact support."
        raise
    except Exception as e:
        error_str = str(e)
        print(f" [RAG Tool] Error in RAG tool: {type(e).__name__}: {e}")
        import traceback
        print(f" [RAG Tool] Traceback: {traceback.format_exc()}")
        # Check if it's a supabase-related error
        if "supabase" in error_str.lower() or "SyncRPCFilterRequestBuilder" in error_str:
            return "I'm experiencing a technical issue accessing the knowledge base. Please try again later or contact support."
        return f"I encountered an error while searching the knowledge base: {str(e)[:200]}"

rag_tool = Tool(
    name="search_internal_docs",
    func=rag_tool_func,
    description="ALWAYS use this tool FIRST for ANY health insurance, HMO, or HR-related questions. This searches the internal knowledge base containing: health plan benefits, coverage limits, provider lists (hospitals, clinics, optical centers), locations, addresses, HR policies, forms, and procedures. Use this for questions about: plan limits, adding dependents, hospital locations, optical centers, dental clinics, maternity coverage, physiotherapy limits, and any other health insurance or HR information."
)

# Web search tool
# Temporarily disabled - langchain-tavily incompatible with langchain 1.0.0+
# TODO: Re-enable when langchain-tavily supports langchain 1.0.0+ or use Tavily API directly
def search_web(query: str) -> str:
    return "Web search is temporarily unavailable. Please contact support for the most up-to-date information."

web_tool = Tool(
    name="search_web",
    func=search_web,
    description="ONLY use if search_internal_docs returns no results. This tool is currently unavailable. For health insurance questions, provider locations, or HR information, ALWAYS use search_internal_docs first."
)

# --- Agent ---
nigerian_system_prompt = """
You are a helpful, accurate, and professional QnA assistant supporting employees of a Nigerian company with health insurance, HR, and provider-related questions.

CRITICAL: For ANY health insurance, HMO, HR, or provider-related question, you MUST use the search_internal_docs tool FIRST. This includes:
- Plan limits and coverage (physiotherapy, dental, optical, etc.)
- Hospital and clinic locations (including "optical hospitals in Ajah")
- Adding dependents to plans
- HR policies and procedures
- Provider lists and addresses

Localization rules:
- 'The Island' means the Lagos Island axis (Lekki, VI, Ikoyi, Ajah, Oniru).
- 'The Mainland' means places like Yaba, Ikeja, Surulere, Maryland.
- Default all context to Nigerian usage and culture.
When users ask for hospital or clinic recommendations, interpret locations within Lagos unless explicitly stated otherwise.

Do not guess if a plan covers something—say what you can verify from context and note what is unknown.
Avoid leaking personal or sensitive data. If you cannot verify the user's identity or entitlement, explain the limitation.
Always explain the reasoning for your recommendations. If you don't have enough information, ask a concise follow-up question.

NEVER use search_web for health insurance questions - always use search_internal_docs first.
"""

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(nigerian_system_prompt),
    MessagesPlaceholder("chat_history"),
    HumanMessagePromptTemplate.from_template("{input}"),
    MessagesPlaceholder("agent_scratchpad")
])
llm = ChatOpenAI(model="gpt-4o", temperature=0)
agent = create_openai_functions_agent(
    llm=llm,
    tools=[rag_tool, web_tool],
    prompt=prompt
)
agent_executor = AgentExecutor(agent=agent, tools=[rag_tool, web_tool], verbose=True)

# --- Token Budgeting Utilities ---

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

# Add validation error handler to log detailed errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body = await request.body()
    print(f" [Validation Error] Request path: {request.url.path}")
    print(f" [Validation Error] Request method: {request.method}")
    print(f" [Validation Error] Validation errors: {exc.errors()}")
    print(f" [Validation Error] Request body: {body.decode('utf-8') if body else 'Empty'}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "body": body.decode('utf-8') if body else 'Empty'}
    )

class ChatRequest(BaseModel):
    session_id: str
    message: str
    user_id: Optional[str] = None  # Optional user ID from KoraID backend

@app.get("/health")
def health_check():
    """Health check endpoint for production deployment"""
    return {
        "status": "healthy",
        "service": "kora-rag-service",
        "timestamp": str(datetime.now()),
        "version": "1.0.0"
    }

@app.post("/chat/rag")
def rag_chat_endpoint(req: ChatRequest):
    """Health insurance questions - RAG system only"""
    sid = req.session_id
    user_id = req.user_id
    current_session_id.set(sid)

    print(f" [RAG CHAT START] ==========================================")
    print(f" [RAG Chat] Processing message for session: {sid}")
    if user_id:
        print(f" [RAG Chat] User ID: {user_id}")
    print(f" [RAG Chat] User message: {req.message}")
    
    # Clean the message (remove prefixes if any)
    clean_message = req.message
    if req.message.startswith('[HEALTH_INSURANCE]'):
        clean_message = req.message.replace('[HEALTH_INSURANCE]', '').strip()
    
    # Create user-specific session ID for history - RAG system
    if user_id:
        effective_session_id = f"rag_user_{user_id}_{sid}"
        print(f" [RAG Chat] Using RAG-specific session: {effective_session_id}")
    else:
        effective_session_id = f"rag_{sid}"
        print(f" [RAG Chat] Using RAG anonymous session: {effective_session_id}")
    
    history = get_agent_history(effective_session_id, user_id)
    
    # Debug: Check what's in the history before adding new message
    print(f" [RAG Chat] History before adding message: {len(history.messages)} messages")
    if history.messages:
        print(f" [RAG Chat] Last message: {history.messages[-1].content[:50]}...")

    # Add the CLEAN user message to history and prune prior to invoke
    history.add_user_message(clean_message)
    print(f" [RAG Chat] Added clean user message: {clean_message[:50]}...")
    
    prune_chat_history_in_place(history)
    print(f" [RAG Chat] After pruning: {len(history.messages)} messages")

    try:
        print(f" [RAG Chat] PURE RAG MODE - Invoking RAG agent only")
        result = agent_executor.invoke({
            "input": clean_message,
            "chat_history": getattr(history, "_pruned_messages", history.messages)
        })
        answer = result.get("output", "")
        
        # Check if the response indicates no relevant information was found
        if not answer or len(answer.strip()) < 10:
            answer = "I couldn't find specific information about that in our knowledge base. Could you please rephrase your question or provide more details about what you're looking for?"
        elif "I don't know" in answer.lower() or "I cannot" in answer.lower() or "I'm not sure" in answer.lower():
            answer = "I don't have specific information about that in our current knowledge base. For the most accurate and up-to-date information, I recommend contacting our HR team or your health insurance provider directly."
        
        print(f" [RAG Chat] PURE RAG response: {answer[:100]}...")
            
    except Exception as e:
        print(f" [RAG Chat] Error in RAG system: {e}")
        # Provide more specific error messages based on the type of error
        if "connection" in str(e).lower() or "timeout" in str(e).lower():
            answer = "I'm having trouble connecting to the knowledge base. Please try again in a moment."
        elif "rate limit" in str(e).lower() or "quota" in str(e).lower():
            answer = "The system is currently experiencing high demand. Please try again in a few minutes."
        else:
            answer = "I'm having trouble accessing the health insurance knowledge base. Please try again later or contact support if the issue persists."

    history.add_ai_message(answer)
    print(f" [RAG Chat] Added AI response to history")
    save_agent_history(sid, history)
    
    print(f" [RAG Chat] Chat completed for session {sid} using RAG system")
    print(f" [RAG CHAT END] ============================================")
    return {"response": answer, "system": "rag"}

@app.post("/chat/gpt5")
def gpt5_chat_endpoint(req: ChatRequest):
    """General questions - GPT-5 system only"""
    sid = req.session_id
    user_id = req.user_id
    current_session_id.set(sid)

    print(f" [GPT5 CHAT START] ==========================================")
    print(f" [GPT5 Chat] Processing message for session: {sid}")
    if user_id:
        print(f" [GPT5 Chat] User ID: {user_id}")
    print(f" [GPT5 Chat] User message: {req.message}")
    
    # Clean the message (remove prefixes if any)
    clean_message = req.message
    if req.message.startswith('[GENERAL_QUESTION]'):
        clean_message = req.message.replace('[GENERAL_QUESTION]', '').strip()
    
    # Create user-specific session ID for history - GPT-5 system
    if user_id:
        effective_session_id = f"gpt5_user_{user_id}_{sid}"
        print(f" [GPT5 Chat] Using GPT5-specific session: {effective_session_id}")
    else:
        effective_session_id = f"gpt5_{sid}"
        print(f" [GPT5 Chat] Using GPT5 anonymous session: {effective_session_id}")
    
    history = get_agent_history(effective_session_id, user_id)
    
    # Debug: Check what's in the history before adding new message
    print(f" [GPT5 Chat] History before adding message: {len(history.messages)} messages")
    if history.messages:
        print(f" [GPT5 Chat] Last message: {history.messages[-1].content[:50]}...")

    # Add the CLEAN user message to history and prune prior to invoke
    history.add_user_message(clean_message)
    print(f" [GPT5 Chat] Added clean user message: {clean_message[:50]}...")
    
    prune_chat_history_in_place(history)
    print(f" [GPT5 Chat] After pruning: {len(history.messages)} messages")

    try:
        print(f" [GPT5 Chat] PURE GPT-5 MODE - Invoking GPT-5 system only")
        answer = call_gpt5_system(clean_message, effective_session_id, user_id)
        
        # Validate answer
        if not answer or not answer.strip():
            answer = "I couldn't generate a response. Please try rephrasing your question."
        
        print(f" [GPT5 Chat] PURE GPT-5 response: {answer[:100]}...")
            
    except Exception as e:
        error_str = str(e)
        print(f" [GPT5 Chat] Error in GPT-5 system: {type(e).__name__}: {error_str}")
        import traceback
        print(f" [GPT5 Chat] Traceback: {traceback.format_exc()}")
        
        # Provide more specific error message
        if "timeout" in error_str.lower():
            answer = "The request timed out. Please try again with a shorter question."
        elif "rate limit" in error_str.lower() or "quota" in error_str.lower():
            answer = "The service is experiencing high demand. Please try again in a few minutes."
        else:
            answer = "I'm having trouble processing your request right now. Please try again."

    history.add_ai_message(answer)
    print(f" [GPT5 Chat] Added AI response to history")
    save_agent_history(sid, history)
    
    print(f" [GPT5 Chat] Chat completed for session {sid} using GPT5 system")
    print(f" [GPT5 CHAT END] ============================================")
    return {"response": answer, "system": "gpt5"}

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
            res = agent_executor.invoke({
                "input": user,
                "chat_history": getattr(history, "_pruned_messages", history.messages)
            })
            answer = res.get("output", "")
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
