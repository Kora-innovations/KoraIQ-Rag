import os
import time
import logging
import contextvars
import argparse
import asyncio
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_core.documents import Document

load_dotenv()
# --- Configurations ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-agent")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "pinecone-test-index")

llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0, streaming=True)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# --- 1. VS & Hybrid Retrieval Setup ---
vectorstore = PineconeVectorStore(
    index_name=PINECONE_INDEX_NAME, 
    embedding=embeddings
)

# Note: In a real production environment, you would persist BM25 
# or use Pinecone's native hybrid (DotProduct + Sparse). 
# For this script, we initialize BM25 from the vectorstore content merely as a fallback demonstration.
# To make this truly fast without reloading docs, rely on Pinecone Hybrid or 
# assume the vector search is primary and BM25 is supplementary for keywords.
# Here we define the standard vector retriever:
# dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 50}) #changed from k=10
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# To achieve >90% accuracy, we add a Reranker.
# Flashrank is a lightweight, local reranker (no API call latency).
compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2")
rerank_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, 
    base_retriever=retriever
)

# --- 2. Caching Layer (Semantic) ---
# Simple in-memory semantic cache to achieve <10ms for repeated queries.
# In production, use Redis or GPTCache.
class SimpleSemanticCache:
    def __init__(self, threshold=0.9):
        self.cache = [] # List of (embedding, response)
        self.threshold = threshold

    async def check(self, query: str) -> Optional[str]:
        if not self.cache: 
            return None
        
        q_embed = await embeddings.aembed_query(query)
        import numpy as np
        
        # Calculate cosine similarity
        scores = []
        for cached_emb, ans in self.cache:
            score = np.dot(q_embed, cached_emb) / (np.linalg.norm(q_embed) * np.linalg.norm(cached_emb))
            scores.append((score, ans))
        
        scores.sort(key=lambda x: x[0], reverse=True)
        best_score, best_ans = scores[0]
        
        if best_score > self.threshold:
            logger.info(f"Cache Hit! Score: {best_score}")
            return best_ans
        return None

    async def store(self, query: str, response: str):
        q_embed = await embeddings.aembed_query(query)
        self.cache.append((q_embed, response))
        # Keep cache small for this demo
        if len(self.cache) > 100: 
            self.cache.pop(0)

semantic_cache = SimpleSemanticCache()

# --- 3. Chains & Prompts ---
# Contextualize Question Prompt (Rewrites user query based on history)
contextualize_q_system_prompt = """Given a chat history and the latest user question 
which might reference context in the chat history, formulate a standalone question 
that can be understood without the chat history. Do NOT answer the question, 
just reformulate it if needed or return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

history_aware_retriever = create_history_aware_retriever(
    llm, rerank_retriever, contextualize_q_prompt
)

# Answer Generation Prompt (Nigerian Context)
system_prompt = """
You are a fast, helpful HR assistant for Kora (Nigeria).
Tone: Friendly, efficient, professional. Use Nigerian English naturally (e.g., "No wahala", "Sharp sharp").

Localization:
- "The Island" = Lekki, VI, Ikoyi, Ajah, Chevron.
- "Mainland" = Agege, Ajeromi-Ifelodun, Alimosho, Amuwo-Odofin, Badagry, Epe, Ifako-Ijaiye, Ikeja, Ikorodu, Kosofe, Lagos Mainland, Mushin, Ojo, Oshodi-Isolo, Shomolu, Surulere.

Data Handling:
- Context: {context}
- If the answer isn't in the context, say "I don't have that info right now" and advise contacting the People Team.
- Do not make up hospitals or benefits.

Logic:
- For "Rhodium" users: show Rhodium + Platinum + Gold + Silver + Bronze.
- For "Bronze" users: show ONLY Bronze.
- Never fabricate coverage details. If unsure, respond with what you know and indicate what is unknown.
    - If a user has the Rhodium plan, return hosptital information relevant to that plan, the Platinum, the Gold, the Customised Gold, the Silver, and the Bronze.
    - If a user has the Platinum plan, return hosptital information relevant to that plan, the Gold, the Customised Gold, the Silver, and the Bronze.
    - If a user has the Customised Gold plan, return hosptital information relevant to that plan, the Gold plan, the Silver, and the Bronze.
    - If a user has the Silver plan, return hosptital information relevant to that plan and the Bronze plan.
    - If a user has the Bronze plan, return hosptital information relevant to ONLY the Bronze plan.
        
Be concise. < 150 words.
"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# --- 4. Router (Replacing ReAct Agent) ---
# Agents are slow (Reasoning Loops). Routers are fast (Classification).
async def intent_router(query: str) -> str:
    """Classifies query to determine if we need RAG or just ChitChat."""
    # Simple keyword heuristic for speed, fallback to LLM for complex cases
    general_greetings = ["hi", "hello", "how are you", "morning", "hey"]
    if query.lower().strip() in general_greetings:
        return "chitchat"
    
    # If not simple, ask a tiny LLM
    router_prompt = f"""Classify this query: "{query}". 
    Return 'rag' if it asks about insurance, hospitals, HR, benefits, or company policy.
    Return 'chitchat' if it's a greeting or general pleasantry.
    Return ONLY the word."""
    
    classification = await llm.ainvoke(router_prompt)
    return classification.content.strip().lower()

# --- FastAPI Setup ---
app = FastAPI()

class ChatRequest(BaseModel):
    session_id: str
    message: str
    history: List[dict] = [] # Expecting [{"role": "user", "content": "..."}]

# @app.post("/chat")
async def generate_response(message: str, history: List[BaseMessage]):
    """
    Shared logic for generating a response. 
    Used by both the FastAPI endpoint and the Terminal CLI.
    """
    start_time = time.time()
    
    # 1. Check Semantic Cache
    cached_response = await semantic_cache.check(message)
    if cached_response:
        return cached_response, "cache", time.time() - start_time
    # 2. Route
    route = await intent_router(message)
    # 3. Process
    response_text = ""
    if "chitchat" in route:
        # Fast chitchat (Context is less critical here, but we pass history for flow)
        res = await llm.ainvoke(
            [("system", "You are a friendly Nigerian HR assistant. Respond warmly.")] + 
            history + 
            [HumanMessage(content=message)]
        )
        response_text = res.content
    else:
        # RAG Flow
        result = await rag_chain.ainvoke({
            "input": message,
            "chat_history": history
        })
        response_text = result["answer"]
    # 4. Update Cache
    await semantic_cache.store(message, response_text)
    return response_text, "rag" if "chitchat" not in route else "llm", time.time() - start_time

# --- 5. FastAPI Endpoint ---
app = FastAPI()

class ChatRequest(BaseModel):
    session_id: str
    message: str
    history: List[dict] = [] 

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    # Convert dict history to LangChain messages
    chat_history = []
    for msg in req.history:
        if msg['role'] == 'user':
            chat_history.append(HumanMessage(content=msg['content']))
        else:
            chat_history.append(AIMessage(content=msg['content']))

    answer, source, latency = await generate_response(req.message, chat_history)

    return {
        "response": answer, 
        "source": source,
        "latency": latency
    }

# --- 6. Interactive Terminal Chat (CLI) ---
async def interactive_chat():
    print("\n--- Kora RAG Agent (Pinecone + GPT-4o-mini) ---")
    print("Type '/exit' to quit. Type '/clear' to reset history.\n")
    
    history = [] # Local session history
    
    while True:
        try:
            user_input = input("\033[94mYou:\033[0m ").strip() # Blue text for user
            if not user_input:
                continue
            if user_input.lower() in ["/exit", "exit", "quit"]:
                print("Goodbye!")
                break
            if user_input.lower() in ["/clear", "clear"]:
                history = []
                print("(History cleared)")
                continue

            # Run the async logic
            answer, source, latency = await generate_response(user_input, history)

            # Print output with metadata
            print(f"\033[92mAgent:\033[0m {answer}") # Green text for Agent
            print(f"\033[90m[Source: {source.upper()} | Latency: {latency:.2f}s]\033[0m\n") # Grey metadata

            # Update History
            history.append(HumanMessage(content=user_input))
            history.append(AIMessage(content=answer))
            
            # Keep history short for this test (sliding window of last 6 turns)
            if len(history) > 6:
                history = history[-6:]

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\033[91mError: {e}\033[0m")

# --- 7. Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kora RAG Agent")
    subparsers = parser.add_subparsers(dest="command", help="Mode to run")

    # Command: serve (Run API)
    serve_parser = subparsers.add_parser("serve", help="Start FastAPI server")
    serve_parser.add_argument("--port", type=int, default=8080)

    # Command: chat (Run Terminal)
    chat_parser = subparsers.add_parser("chat", help="Start interactive terminal chat")

    args = parser.parse_args()

    if args.command == "chat":
        # Asyncio run is needed to execute async function from sync main
        asyncio.run(interactive_chat())
    elif args.command == "serve":
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    else:
        # Default to chat if no arg provided (for convenience)
        asyncio.run(interactive_chat())