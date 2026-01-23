import os
import time
import re
import asyncio
import argparse
from dotenv import load_dotenv
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()
# --- Config ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "kora-rag-prod")

# Initialize Models
# Streaming=True is essential for the "instant" feel
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# --- 1. Vector Store (No Reranker for Speed) ---
vectorstore = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
# We trust the Top 8 results. With good metadata, this is usually accurate enough.
retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

# --- 2. Prompts ---
system_prompt = """
# ROLE
You are the "Kora People Navigator," an expert HR Advisor for Kora employees in Nigeria. You are resourceful, empathetic, and highly organized. Your goal is to provide accurate information regarding benefits and policies while maintaining the vibrant professional culture of Kora.

# TONE & STYLE
- **Professional Naija:** Use "Standard Nigerian English." Be professional but warm. Use local context naturally (e.g., "Good morning," "Kindly find," "Please note"). 
- **Efficiency:** Get to the point quickly, but don't be cold.
- **Localization:** Understand that for the Lagos locations, "The Island" refers to Lekki, VICTORIA ISLAND (VI), IKOYI, AJAH, EPE, LAGOS ISLAND; and "The Mainland" refers to Yaba, Ikeja, SURULERE, GBAGADA, BARIGA, SOMOLU, OWOROSHONKI, ANTHONY VILLAGE, MARYLAND, OBANIKORO, ILUPEJU, OGUDU, OJOTA, ISOLO, AJAO ESTATE, IKORODU, IKOTUN, IGANDO, ISHERI OSUN, ALAGBADO, OKOTA, FESTAC TOWN, SATELLITE TOWN, AGBARA, BADAGRY, OJO, OKOKOMAIKO, OJODU BERGER, KETU, MAGODO, AKOWONJO/EGBEDA/ABULE EGBA/IDIMU/AGEGE/IYANA IPAJA/DOPEMU/AGBADO, AGEGE, AYOBO, APAPA/AJEGUNLE, AKOKA, YABA/EBUTE METTA, OGBA/IKEJA, OSHODI, MUSHIN, EJIGBO. Account for local nuances like traffic or proximity when relevant.

# THE REASONING ENGINE (TIERED ACCESS LOGIC)
You must apply "Downward Compatibility" logic for all provider/benefit queries. Before answering, mentally verify the user's tier and follow this hierarchy:

1. **Rhodium:** (Highest) Access to EVERYTHING (Rhodium, Platinum, Custom Gold, Gold, Silver, Bronze).
2. **Platinum:** Access to Platinum, Gold, Silver, and Bronze.
3. **Customised Gold:** Access to Customised Gold, standard Gold, Silver, and Bronze.
4. **Gold:** Access to Gold, Silver, and Bronze.
5. **Silver:** Access to Silver and Bronze.
6. **Bronze:** (Base) Access to Bronze providers ONLY.

**Constraint:** If a user asks for a provider in a tier above theirs, politely explain that it is outside their current plan.

# DATA GROUNDING
- Use ONLY the provided context: {context}.
- If the specific information is not in the context, say: "I don't have that specific information right now. Please reach out to the People team for further clarification."
- Never hallucinate provider names, tier levels, policy information, or processes.

# RESPONSE STRUCTURE
1. Acknowledge the query warmly.
2. State the relevant policy or provider clearly (if applicable).
3. (If applicable) Mention localization context (Island vs. Mainland).
4. End with a helpful closing.

"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Contextualize Prompt (Only used when history exists)
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given a chat history and the latest user question, formulate a standalone question. Do NOT answer it, just rewrite it if needed."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Create Chains
history_chain = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
qa_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_chain, qa_chain)

# --- 3. Optimized Logic ---
class InMemoryCache:
    def __init__(self):
        self.cache = {} # Map query -> response
    
    def get(self, query):
        return self.cache.get(query.strip().lower())
    
    def set(self, query, response):
        self.cache[query.strip().lower()] = response

exact_cache = InMemoryCache()

async def get_response_stream(message: str, history: List):
    """
    Generator that yields chunks of the response for true streaming.
    """
    start_time = time.time()
    
    # 1. Exact Cache Check (0ms latency)
    cached = exact_cache.get(message)
    if cached:
        yield f"{cached}"
        yield f"\n\n\033[90m[Source: EXACT CACHE | Latency: {time.time() - start_time:.2f}s]\033[0m"
        return

    # 2. Fast Router (Regex - 0ms latency)
    # Don't waste an LLM call on "Hi"
    greetings = r"^(hi|hello|hey|how far|morning|afternoon|evening)\b"
    if re.match(greetings, message.lower()):
        # Stream a static response instantly
        greeting_resp = "Hi there! How can the People team help you today?"
        for word in greeting_resp.split():
            yield word + " "
            await asyncio.sleep(0.05) # Fake typing effect
        yield f"\n\n\033[90m[Source: FAST ROUTER | Latency: {time.time() - start_time:.2f}s]\033[0m"
        return

    # 3. Smart RAG Selection
    # If NO history, we don't need to "contextualize" the question. 
    # We can skip the first LLM call and go straight to retrieval.
    if not history:
        # Fast Lane: Retrieve -> Generate
        docs = await retriever.ainvoke(message)
        # Manually invoke the answer chain with docs
        stream = qa_chain.astream({
            "context": docs,
            "chat_history": [],
            "input": message
        })
    else:
        # Slow Lane: Rewrite Question -> Retrieve -> Generate
        stream = rag_chain.astream({
            "input": message,
            "chat_history": history
        })

    # 4. Stream the Output
    full_answer = ""
    async for chunk in stream:
        # LangChain's .stream returns chunks differently depending on the chain
        if isinstance(chunk, dict) and "answer" in chunk:
            token = chunk["answer"]
            full_answer += token
            yield token
        elif isinstance(chunk, str):
            full_answer += chunk
            yield chunk
        # Handle simple chain output
        elif hasattr(chunk, 'content'): 
            full_answer += chunk.content
            yield chunk.content

    # Cache the full result for next time
    exact_cache.set(message, full_answer)
    
    yield f"\n\n\033[90m[Source: RAG | Latency: {time.time() - start_time:.2f}s]\033[0m"

# --- 4. Interactive Chat (Streaming Enabled) ---
async def interactive_chat():
    print("\n--- Kora RAG Agent (Optimized Stream) ---")
    print("Type '/exit' to quit. \n")
    
    history = []
    
    while True:
        try:
            user_input = input("\n\033[94mYou:\033[0m ").strip()
            if user_input.lower() in ["/exit", "exit"]: break
            
            print("\033[92mAgent:\033[0m ", end="", flush=True)
            
            full_response = ""
            # Consume the stream
            async for token in get_response_stream(user_input, history):
                print(token, end="", flush=True)
                # Filter out the metadata for the history log
                if not token.startswith("\n\n\033"): 
                    full_response += token
            
            print("") # Newline
            
            # Update history
            history.append(HumanMessage(content=user_input))
            history.append(AIMessage(content=full_response))
            if len(history) > 6: history = history[-6:]
            
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    asyncio.run(interactive_chat())