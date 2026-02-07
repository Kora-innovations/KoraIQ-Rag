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
# Both indexes: answers come from the selected index only
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "people-team")
PINECONE_INDEX_NAME_WITHOUT_FAQS = os.getenv("PINECONE_INDEX_NAME_WITHOUT_FAQS", "people-team-without-faqs")

# Validate index names
if not PINECONE_INDEX_NAME or not PINECONE_INDEX_NAME_WITHOUT_FAQS:
    raise ValueError("Set PINECONE_INDEX_NAME and PINECONE_INDEX_NAME_WITHOUT_FAQS in .env")

# Initialize Models
# Streaming=True is essential for the "instant" feel
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# --- 1. Two Vector Stores (one per index); all responses come from the chosen index only ---
vectorstore_people_team = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
vectorstore_without_faqs = PineconeVectorStore(index_name=PINECONE_INDEX_NAME_WITHOUT_FAQS, embedding=embeddings)

# Two retrievers: top 8 from the selected index
retriever_people_team = vectorstore_people_team.as_retriever(search_kwargs={"k": 8})
retriever_without_faqs = vectorstore_without_faqs.as_retriever(search_kwargs={"k": 8})

# Index choice constants (used by caller to select which index to query)
INDEX_PEOPLE_TEAM = "people-team"
INDEX_WITHOUT_FAQS = "people-team-without-faqs"

# --- 2. Prompts ---
system_prompt = """
# ROLE
You are the "Kora People Navigator," an expert HR Advisor for Kora employees in Nigeria. You are resourceful, empathetic, and highly organized. Your goal is to provide accurate information regarding benefits and policies while maintaining the vibrant professional culture of Kora.

# TONE & STYLE
- **Tone**: We usually keep it friendly, direct, and respectful and always aim to be warm but straight to the point. We don’t overdo the fluff, but we keep the tone approachable.
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

# DATA GROUNDING (CRITICAL)
- ALL answers MUST come ONLY from the provided context: {context}. Do not use external knowledge.
- If the specific information is not in the context, say: "I don't have that specific information right now. Please reach out to the People team for further clarification."
- Never hallucinate provider names, tier levels, policy information, or processes.
- If there's any chance that the answer to a question contains a link, ALWAYS provide the exact URL from the context. Even if the user does not ask for it.
- The context is retrieved from Kora's knowledge base; your reply must be grounded 100% in that context.

# RESPONSE STRUCTURE
1. Acknowledge the query warmly.
2. State the relevant policy or provider clearly (if applicable).
3. (If applicable) Mention localization context (Island vs. Mainland).
4. End with a helpful closing.
5. For words you want to give a bold formatting, format the output using standard Markdown 
bolding for keys. Ensure there are no backslashes before the asterisks and no 
leading spaces before the bullet points. Example: * **Key:** Value
The goal is for the end user to not experience words visibly wrapped in asterisks.
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

# Create Chains per index (so retrieval always uses the selected index)
history_chain_people_team = create_history_aware_retriever(llm, retriever_people_team, contextualize_q_prompt)
history_chain_without_faqs = create_history_aware_retriever(llm, retriever_without_faqs, contextualize_q_prompt)
qa_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain_people_team = create_retrieval_chain(history_chain_people_team, qa_chain)
rag_chain_without_faqs = create_retrieval_chain(history_chain_without_faqs, qa_chain)

# --- 3. Optimized Logic ---
def _get_retriever(index_choice: str):
    """Return the retriever for the selected index. All responses come from this index only."""
    if index_choice == INDEX_WITHOUT_FAQS:
        return retriever_without_faqs
    return retriever_people_team  # default: people-team

def _get_rag_chain(index_choice: str):
    """Return the RAG chain for the selected index."""
    if index_choice == INDEX_WITHOUT_FAQS:
        return rag_chain_without_faqs
    return rag_chain_people_team  # default: people-team

class InMemoryCache:
    def __init__(self):
        self.cache = {}  # Map (query_lower, index_choice) -> response

    def get(self, query: str, index_choice: str) -> str | None:
        key = (query.strip().lower(), index_choice)
        return self.cache.get(key)

    def set(self, query: str, index_choice: str, response: str):
        key = (query.strip().lower(), index_choice)
        self.cache[key] = response

exact_cache = InMemoryCache()

async def get_response_stream(message: str, history: List, index_choice: str = INDEX_PEOPLE_TEAM):
    """
    Generator that yields chunks of the response for true streaming.
    index_choice: "people-team" or "people-team-without-faqs" — retrieval uses only that index.
    All answers come from the selected index only.
    """
    start_time = time.time()
    # Normalize index choice
    if index_choice not in (INDEX_PEOPLE_TEAM, INDEX_WITHOUT_FAQS):
        index_choice = INDEX_PEOPLE_TEAM

    # 1. Exact Cache Check (0ms latency) — keyed by message and index
    cached = exact_cache.get(message, index_choice)
    if cached:
        yield cached
        yield f"\n\n\033[90m[Source: EXACT CACHE | Index: {index_choice} | Latency: {time.time() - start_time:.2f}s]\033[0m"
        return

    # 2. Fast Router (Regex - 0ms latency)
    greetings = r"^(Hi hi|how you dey|how are you|how far|hiya|how's it going)\b"
    if re.match(greetings, message.lower()):
        greeting_resp = "Hi there! How can the People team help you today?"
        for word in greeting_resp.split():
            yield word + " "
            await asyncio.sleep(0.05)
        yield f"\n\n\033[90m[Source: FAST ROUTER | Latency: {time.time() - start_time:.2f}s]\033[0m"
        return

    # 3. RAG from selected index only
    retriever = _get_retriever(index_choice)
    rag_chain = _get_rag_chain(index_choice)

    if not history:
        # Fast Lane: Retrieve from selected index -> Generate (context only from that index)
        docs = await retriever.ainvoke(message)
        stream = qa_chain.astream({
            "context": docs,
            "chat_history": [],
            "input": message
        })
    else:
        # Slow Lane: Rewrite -> Retrieve from selected index -> Generate
        stream = rag_chain.astream({
            "input": message,
            "chat_history": history
        })

    # 4. Stream the Output
    full_answer = ""
    async for chunk in stream:
        if isinstance(chunk, dict) and "answer" in chunk:
            token = chunk["answer"]
            full_answer += token
            yield token
        elif isinstance(chunk, str):
            full_answer += chunk
            yield chunk
        elif hasattr(chunk, 'content'):
            full_answer += chunk.content
            yield chunk.content

    exact_cache.set(message, index_choice, full_answer)
    yield f"\n\n\033[90m[Source: RAG | Index: {index_choice} | Latency: {time.time() - start_time:.2f}s]\033[0m"

# --- 4. Interactive Chat (Streaming Enabled) ---
async def interactive_chat():
    print("\n--- Kora RAG Agent (Optimized Stream) ---")
    print("Two indexes: all answers come from the selected index only.")
    print(f"  1 = {INDEX_PEOPLE_TEAM}")
    print(f"  2 = {INDEX_WITHOUT_FAQS}")
    print("Type '/switch' to change index, '/exit' to quit.\n")

    index_choice = INDEX_PEOPLE_TEAM
    history = []

    while True:
        try:
            prompt = f"\n\033[94mYou [{index_choice}]:\033[0m "
            user_input = input(prompt).strip()
            if user_input.lower() in ["/exit", "exit"]:
                break
            if user_input.lower() == "/switch":
                index_choice = INDEX_WITHOUT_FAQS if index_choice == INDEX_PEOPLE_TEAM else INDEX_PEOPLE_TEAM
                print(f"\033[93mSwitched to index: {index_choice}\033[0m")
                continue

            print("\033[92mAgent:\033[0m ", end="", flush=True)
            full_response = ""
            async for token in get_response_stream(user_input, history, index_choice):
                print(token, end="", flush=True)
                if not token.startswith("\n\n\033"):
                    full_response += token
            print("")

            history.append(HumanMessage(content=user_input))
            history.append(AIMessage(content=full_response))
            if len(history) > 6:
                history = history[-6:]

        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    asyncio.run(interactive_chat())